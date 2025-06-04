from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from pathlib import Path
from enum import Enum
import json
from typing import Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM

load_dotenv()

app = FastAPI(title="RAG API", description="API for document ingestion and querying using RAG")

# Setup directories
BASE_DIR = Path(__file__).resolve().parent
FILES_DIR = BASE_DIR / "files"
CHROMA_PATH = BASE_DIR / "chroma"
FILES_DIR.mkdir(exist_ok=True)

# Store file-specific databases
DB_SESSIONS: Dict[str, Chroma] = {}

# Setup static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

PROMPT_TEMPLATE = """
You are a helpful assistant. Use only the information provided in the context below to answer the question. Do not include any information that is not in the context.

Context:
{context}

Question:
{question}
"""

class ModelType(str, Enum):
    DEEPSEEK = "deepseek"
    GPT = "gpt"

class QueryRequest(BaseModel):
    query: str
    model: ModelType
    file_name: str

class QueryResponse(BaseModel):
    response: str
    sources: List[Optional[str]]

class DatabaseResponse(BaseModel):
    message: str
    chunks_processed: int

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("files.html", {"request": request})

@app.get("/files")
async def files_page(request: Request):
    return templates.TemplateResponse("files.html", {"request": request})

@app.get("/chat/{file_name}")
async def chat_page(request: Request, file_name: str):
    file_path = FILES_DIR / file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/list-files")
async def list_files():
    files = [f.name for f in FILES_DIR.glob("*.pdf")]
    return files

@app.delete("/delete-file/{file_name}")
async def delete_file(file_name: str):
    try:
        # Delete PDF file
        file_path = FILES_DIR / file_name
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        file_path.unlink()

        # Delete Chroma database
        db_path = CHROMA_PATH / file_name.replace('.pdf', '')
        if db_path.exists():
            shutil.rmtree(db_path)

        # Remove database session
        if file_name in DB_SESSIONS:
            del DB_SESSIONS[file_name]

        return {"message": "File deleted successfully"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=DatabaseResponse)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save the uploaded file
        file_path = FILES_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load documents
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create file-specific Chroma directory
        db_path = CHROMA_PATH / file.filename.replace('.pdf', '')
        if db_path.exists():
            shutil.rmtree(db_path)
            
        db = Chroma.from_documents(
            documents=chunks,
            embedding=EMBEDDING_MODEL,
            persist_directory=str(db_path),
        )
        
        # Store the database session
        DB_SESSIONS[file.filename] = db
        
        return DatabaseResponse(
            message="Database created successfully",
            chunks_processed=len(chunks)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    try:
        if not request.file_name:
            raise HTTPException(status_code=400, detail="File name is required")
            
        # Get or load the database for this file
        if request.file_name not in DB_SESSIONS:
            db_path = CHROMA_PATH / request.file_name.replace('.pdf', '')
            if not db_path.exists():
                raise HTTPException(status_code=404, detail="File database not found")
            DB_SESSIONS[request.file_name] = Chroma(
                embedding_function=EMBEDDING_MODEL,
                persist_directory=str(db_path),
            )
            
        db = DB_SESSIONS[request.file_name]
        
        # Perform the query
        results = db.similarity_search_with_relevance_scores(request.query, k=10)
        if len(results) == 0 or results[0][1] < 0.5:
            raise HTTPException(status_code=404, detail="No relevant results found")
            
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=request.query)
        
        # Select model based on request
        if request.model == ModelType.DEEPSEEK:
            model = ChatLiteLLM(model="deepseek/deepseek-chat",api_key=os.getenv("DEEPSEEK_API_KEY"))
            response = model.invoke(prompt)
        else:  # GPT
            model = ChatLiteLLM(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
            response = model.invoke(prompt)
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        
        return QueryResponse(
            response=str(response_text),
            sources=[]
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)