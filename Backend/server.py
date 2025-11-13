# server.py
import os
import tempfile
from operator import itemgetter
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough  # Fixed import
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langserve import add_routes
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"

try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
except Exception as e:
    print(f"Error initializing models: {e}")
    llm = None
    embeddings = None

vector_store = None

def load_document_from_upload(uploaded_file: UploadFile):
    """Loads content from a FastAPI UploadFile."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.filename)[1]) as tmp_file:
            tmp_file.write(uploaded_file.file.read())
            tmp_file_path = tmp_file.name
        
        if uploaded_file.content_type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.content_type == "text/plain":
            loader = TextLoader(tmp_file_path, encoding="utf-8")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {uploaded_file.content_type}")
        
        documents = loader.load()
        
        for doc in documents:
            doc.metadata["source"] = uploaded_file.filename
        return documents

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading file {uploaded_file.filename}: {e}")
    finally:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def process_and_store_docs(documents: List):
    """Splits, embeds, and stores documents in FAISS."""
    global vector_store
    
    if not documents:
        return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(documents)
    
    current_vector_store = FAISS.from_documents(doc_splits, embeddings)
    current_vector_store.save_local(FAISS_INDEX_PATH)
    vector_store = current_vector_store
    print("Vector store created and saved.")

app = FastAPI(
    title="LangServe RAG Server",
    description="API for ingesting documents and answering questions."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API. Use /upload and /chat."}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Endpoint to upload documents."""
    all_documents = []
    for uploaded_file in files:
        docs = load_document_from_upload(uploaded_file)
        if docs:
            all_documents.extend(docs)
            
    if not all_documents:
        raise HTTPException(status_code=400, detail="No valid documents were processed.")
    
    process_and_store_docs(all_documents)
    
    return {"message": f"Successfully processed {len(files)} file(s). Ready to chat."}

def get_vector_store():
    """Helper to load the vector store."""
    global vector_store
    if vector_store is None:
        if os.path.exists(FAISS_INDEX_PATH):
            print("Loading existing vector store from disk.")
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            raise HTTPException(status_code=400, detail="Vector store not initialized. Please upload documents first via /upload.")
    return vector_store

def format_docs(docs: List) -> str:
    """Converts retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_sources(docs: List) -> List[str]:
    """Extracts source filenames from retrieved documents."""
    return list(set(doc.metadata.get("source", "Unknown") for doc in docs))

template = """
You are an assistant for answering questions about company policies.
Use ONLY the following context to answer the question.
If you don't know the answer from the context, just say that you don't know.
Do not use any other information.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Fixed: Use dictionaries instead of RunnableMap
class ChatInput(BaseModel):
    question: str

# Create chain lazily on each request
def final_chain_func(inputs):
    retriever = get_vector_store().as_retriever(search_kwargs={"k": 3})
    question = inputs["question"]
    docs = retriever.invoke(question)
    
    # Format context and get answer
    context = format_docs(docs)
    answer_input = {"context": context, "question": question}
    answer = (prompt | llm | StrOutputParser()).invoke(answer_input)
    
    sources = get_sources(docs)
    return {"answer": answer, "sources": sources}

from langchain_core.runnables import RunnableLambda
final_chain = RunnableLambda(final_chain_func).with_types(input_type=ChatInput)

add_routes(
    app,
    final_chain,
    path="/chat",
)

if __name__ == "__main__":
    print("Starting server... Go to http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)