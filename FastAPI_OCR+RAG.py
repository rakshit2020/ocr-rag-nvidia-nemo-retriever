from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import base64
import os
from io import BytesIO
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA, NVIDIARerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI(title="OCR + RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vectordb = None
RETRIEVER = None
llm = None
ocr_client = None

# Configuration
OCR_MODEL = "nanonets/Nanonets-OCR2-3B"
LLM_MODEL = "meta/llama-3.3-70b-instruct"
EMBEDDING_MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"
RERANK_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"


# Request/Response models
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    num_sources: int


class OCRResponse(BaseModel):
    ocr_text: str
    preview: str
    num_chunks: int


# Initialize clients
def initialize_clients():
    global ocr_client, llm

    ocr_client = OpenAI(
        api_key="123",
        base_url="http://localhost:8000/v1"
    )

    os.environ["NVIDIA_API_KEY"] = "put your Nvidia API key"

    llm = ChatNVIDIA(
        model=LLM_MODEL,
        temperature=0.6,
        top_p=0.95,
        max_completion_tokens=8048
    )


@app.on_event("startup")
async def startup_event():
    """Initialize clients on startup"""
    initialize_clients()


def encode_image_bytes(image_bytes: bytes) -> str:
    """Encode image bytes to base64"""
    return base64.b64encode(image_bytes).decode("utf-8")


def ocr_image(img_base64: str) -> str:
    """Perform OCR on image using Nanonets"""
    response = ocr_client.chat.completions.create(
        model=OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes.",
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=15000
    )
    return response.choices[0].message.content


def create_vector_store(documents: List[str]):
    """Create FAISS vector store from documents"""
    global vectordb, RETRIEVER

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.create_documents(documents)

    # Create embeddings
    embeddings = NVIDIAEmbeddings(
        model=EMBEDDING_MODEL,
        truncate="END"
    )

    # Create vector store
    vectordb = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # Create retriever with reranker
    kb_retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    reranker = NVIDIARerank(model=RERANK_MODEL)

    RETRIEVER = ContextualCompressionRetriever(
        base_retriever=kb_retriever,
        base_compressor=reranker,
    )

    return len(chunks)


@app.post("/upload", response_model=OCRResponse)
async def upload_images(files: List[UploadFile] = File(...)):
    """
    Upload images for OCR processing and vector store creation
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        documents = []
        all_ocr_text = []

        for file in files:
            # Validate file type
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not an image"
                )

            # Read image
            image_bytes = await file.read()

            # Encode to base64
            img_base64 = encode_image_bytes(image_bytes)

            # Perform OCR
            ocr_text = ocr_image(img_base64)
            documents.append(ocr_text)
            all_ocr_text.append(ocr_text)

        # Create vector store
        num_chunks = create_vector_store(documents)

        # Create preview (first 500 chars)
        full_text = "\n\n".join(all_ocr_text)
        preview = full_text[:500] + "..." if len(full_text) > 500 else full_text

        return OCRResponse(
            ocr_text=full_text,
            preview=preview,
            num_chunks=num_chunks
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system
    """
    try:
        if RETRIEVER is None:
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded. Please upload images first."
            )

        # Retrieve relevant documents
        retrieved_docs = RETRIEVER.invoke(request.question)

        # Format context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Create prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. Answer the question using ONLY the context provided."
                    "If you cannot answer from the context, say 'I don't have enough information.'"
                    "The context you get is from the output of the OCR model."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {request.question}"
            }
        ]

        # Get response
        response = llm.invoke(messages)

        return QueryResponse(
            question=request.question,
            answer=response.content,
            num_sources=len(retrieved_docs)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_store_initialized": vectordb is not None
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

