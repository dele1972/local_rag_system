# app/loader.py
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import os

def load_documents_from_path(path):
    """Load documents from a directory path with improved logging"""
    docs = []
    loaded_files = []
    
    for file in Path(path).rglob("*"):
        if file.is_file():
            try:
                if file.suffix.lower() == ".pdf":
                    docs.extend(PyPDFLoader(str(file)).load())
                    loaded_files.append(str(file))
                elif file.suffix.lower() == ".txt":
                    docs.extend(TextLoader(str(file)).load())
                    loaded_files.append(str(file))
                elif file.suffix.lower() == ".md":
                    docs.extend(UnstructuredMarkdownLoader(str(file)).load())
                    loaded_files.append(str(file))
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(docs)
    
    return split_docs, loaded_files