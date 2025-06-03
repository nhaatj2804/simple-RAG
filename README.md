# simple-RAG

A Retrieval-Augmented Generation (RAG) application built with FastAPI and LangChain that allows users to query PDF documents using LLM models.

## Features

- Web interface for uploading and querying PDF documents
- CLI tool for direct document querying
- Support for multiple LLM models (DeepSeek and GPT-3.5)
- Document chunking and semantic search using Chroma vector store
- Embeddings using HuggingFace's E5-small model

## Prerequisites

- Python 3.8+
- API keys for DeepSeek and/or OpenAI (optional for GPT-3.5)

## Installation

1. Clone the repository and navigate to the project directory
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:

```
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key  # Optional for GPT-3.5
```

## Usage

### Web Interface

1. Start the FastAPI server:

```bash
python main.py
```

2. Open your browser and navigate to `http://localhost:8000`

3. Use the web interface to:
   - Upload PDF documents
   - Select documents for querying
   - Choose between DeepSeek and GPT-3.5 models
   - Ask questions about the document content

## Project Structure

```
simple-RAG/
├── main.py              # FastAPI web application
├── create_database.py   # PDF processing script
├── query_data.py        # CLI query tool
├── files/              # Directory for PDF documents
├── templates/          # HTML templates
├── static/            # Static files (CSS, JS)
└── chroma/            # Vector store directory
```

## Technical Details

- Document chunking: Uses RecursiveCharacterTextSplitter with 1000-character chunks and 200-character overlap
- Embeddings: HuggingFace E5-small-v2 model
- Vector Store: Chroma DB for efficient similarity search
- Web Framework: FastAPI with Jinja2 templates

## Limitations

- Currently supports PDF files only
- Requires API keys for LLM models
- Vector store is file-specific and stored locally

## Environment Variables

- `DEEPSEEK_API_KEY`: Required for using the DeepSeek model
- `OPENAI_API_KEY`: Required for using GPT-3.5-turbo model (optional)
