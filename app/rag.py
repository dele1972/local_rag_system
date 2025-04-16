# rag.py
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from app.config import config
import requests

def check_ollama_connection():
    """Check if Ollama is accessible"""
    try:
        response = requests.get(f"{config.get_ollama_base_url()}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def build_qa_chain(vectorstore, model_name):
    """Build a question-answering chain with error handling"""
    if not check_ollama_connection():
        raise ConnectionError(f"Cannot connect to Ollama at {config.get_ollama_base_url()}")
    
    llm = OllamaLLM(
        model=model_name,
        base_url=config.get_ollama_base_url()
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True  # Include source documents in response
    )
    
    return qa