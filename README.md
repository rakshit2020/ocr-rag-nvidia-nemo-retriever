# OCR + NVIDIA NeMo Retriever RAG System

Production-ready document intelligence system combining Nanonets OCR with NVIDIA NeMo Retriever for semantic search and question answering.

## Features

- 🖼️ **OCR Extraction**: Extract text, tables, equations, and checkboxes from documents using Nanonets OCR2-3B
- 🔍 **Semantic Search**: NVIDIA NeMo Retriever embeddings for intelligent document retrieval
- 🎯 **Reranking**: Contextual reranking for improved answer accuracy
- 🚀 **FastAPI Backend**: Scalable REST API for document processing
- 💬 **Streamlit Frontend**: User-friendly chat interface
- 📊 **OCR Preview**: View extracted text before querying

## Prerequisites

- Python 3.10+
- NVIDIA API Key ([Get one here](https://build.nvidia.com/))
- vLLM server running with Nanonets OCR2-3B model (on `http://localhost:8000`)

## Installation

1. **Clone the repository**
git clone <your-repo-url>
cd ocr-rag-system

## Quick Start Commands

### Install everything
pip install -r requirements.txt

### Terminal 1: Start vLLM
vllm serve nanonets/Nanonets-OCR2-3B --port 8000

### Terminal 2: Start FastAPI
uvicorn FastAPI_OCR+RAG:app --host 0.0.0.0 --port 8080

### Terminal 3: Start Streamlit
streamlit run streamlit_app.py
