# app/loader.py
# from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from pathlib import Path

def load_documents_from_path(path):
    docs = []
    for file in Path(path).rglob("*"):
        if file.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
        elif file.suffix == ".txt":
            docs.extend(TextLoader(str(file)).load())
        elif file.suffix == ".md":
            docs.extend(UnstructuredMarkdownLoader(str(file)).load())
    return docs