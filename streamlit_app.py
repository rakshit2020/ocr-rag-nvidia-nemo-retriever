import streamlit as st
import requests
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="OCR + Nvidia Nemo Retriever",
    layout="wide"
)

# API configuration
API_URL = "http://localhost:8080"

# Initialize session state
if "ocr_preview" not in st.session_state:
    st.session_state.ocr_preview = None
if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title
st.title(" OCR + Nvidia Nemo Retriever")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose images",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if st.button("Process Images", type="primary", disabled=not uploaded_files):
        with st.spinner("Processing images with OCR..."):
            try:
                # Prepare files for upload
                files = []
                for uploaded_file in uploaded_files:
                    files.append(
                        ("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
                    )

                # Send to API
                response = requests.post(f"{API_URL}/upload", files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.session_state.ocr_preview = result["preview"]
                    st.session_state.documents_uploaded = True
                    st.session_state.num_chunks = result["num_chunks"]
                    st.success(f"✅ Processed {len(uploaded_files)} image(s) into {result['num_chunks']} chunks")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

            except Exception as e:
                st.error(f"Connection error: {str(e)}")

    # Show OCR preview in sidebar
    if st.session_state.ocr_preview:
        st.divider()
        st.subheader("OCR Preview")
        st.text_area(
            "Extracted Text",
            value=st.session_state.ocr_preview,
            height=300,
            disabled=True,
            label_visibility="collapsed"
        )

# Main chat interface
if st.session_state.documents_uploaded:
    st.subheader("Ask Questions")

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            st.caption(f"Sources used: {chat['num_sources']}")

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Display user message
        with st.chat_message("user"):
            st.write(question)

        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={"question": question}
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.write(result["answer"])
                        st.caption(f"Sources used: {result['num_sources']}")

                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": result["answer"],
                            "num_sources": result["num_sources"]
                        })
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

else:
    # Welcome message
    st.info("👈 Please upload images in the sidebar to get started")

    # Feature highlights
    col1, col2, col3 = st.columns(3)

    # with col1:
    #     st.markdown("### 🖼️ OCR Extraction")
    #     st.markdown("Extract text, tables, equations, and images from documents")
    #
    # with col2:
    #     st.markdown("### 🔍 Semantic Search")
    #     st.markdown("Powered by NVIDIA NeMo Retriever embeddings")
    #
    # with col3:
    #     st.markdown("### 🤖 AI Responses")
    #     st.markdown("Get accurate answers using Llama 3.3 70B")

# Footer
st.divider()
st.caption("Powered by NVIDIA NIM | Nanonets OCR ")
