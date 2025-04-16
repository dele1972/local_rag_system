# app/vectorstore.py
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.embeddings import OllamaEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
import os

def build_vectorstore(documents, model_name):
    # embeddings = OllamaEmbeddings(model=model_name)
    embeddings = OllamaEmbeddings(base_url="http://host.docker.internal:11434", model=model_name)
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    return vectorstore