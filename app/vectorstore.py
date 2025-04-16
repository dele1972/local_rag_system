# app/vectorstore.py
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from app.config import config
import os
import shutil

def get_embeddings(model_name):
    """Erstellt ein Embedding-Objekt für das angegebene Modell"""
    return OllamaEmbeddings(
        base_url=config.get_ollama_base_url(), 
        model=model_name
    )

def build_vectorstore(documents, model_name, persist_directory=None):
    """Create a Chroma vector store from documents with persistence option
    
    Args:
        documents: Liste der zu vektorisierenden Dokumente
        model_name: Name des Ollama-Modells für Embeddings
        persist_directory: Optional - Pfad zum Speichern der Vektorendatenbank
    """
    embeddings = get_embeddings(model_name)
    
    if persist_directory:
        # Persistenter Vektorspeicher mit Chroma
        vectorstore = Chroma.from_documents(
            documents=documents, 
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()  # Sicherstellen, dass Änderungen geschrieben werden
    else:
        # In-Memory Vektorspeicher
        vectorstore = Chroma.from_documents(
            documents=documents, 
            embedding=embeddings
        )
    
    return vectorstore

def load_vectorstore(persist_directory, model_name):
    """Load a previously saved Chroma vector store
    
    Args:
        persist_directory: Pfad zur gespeicherten Vektorendatenbank
        model_name: Name des Ollama-Modells für Embeddings
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Vector store not found at {persist_directory}")
    
    embeddings = get_embeddings(model_name)
    
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def delete_vectorstore(persist_directory):
    """Delete a persisted vector store
    
    Args:
        persist_directory: Pfad zur zu löschenden Vektorendatenbank
    """
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        return True
    return False

def list_vectorstores(base_directory):
    """List all available vector stores in a directory
    
    Args:
        base_directory: Basisverzeichnis für die Suche nach Vektorspeichern
    
    Returns:
        Dict mit Name->Pfad der gefundenen Vektorspeicher
    """
    vectorstores = {}
    
    if not os.path.exists(base_directory):
        return vectorstores
    
    # Suche nach Unterverzeichnissen mit Chroma-Dateien
    for item in os.listdir(base_directory):
        item_path = os.path.join(base_directory, item)
        if os.path.isdir(item_path) and "chroma" in os.listdir(item_path):
            vectorstores[item] = item_path
    
    return vectorstores