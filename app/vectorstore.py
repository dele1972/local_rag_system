# app/vectorstore.py
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from app.config import config

def build_vectorstore(documents, model_name):
    """Create a FAISS vector store from documents"""
    embeddings = OllamaEmbeddings(
        base_url=config.get_ollama_base_url(), 
        model=model_name
    )
    
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    return vectorstore

def save_vectorstore(vectorstore, path):
    """Save the vector store for future use"""
    vectorstore.save_local(path)

def load_vectorstore(path, model_name):
    """Load a previously saved vector store"""
    embeddings = OllamaEmbeddings(
        base_url=config.get_ollama_base_url(), 
        model=model_name
    )
    return FAISS.load_local(path, embeddings)