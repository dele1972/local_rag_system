# app/rag.py
from langchain.chains import RetrievalQA
# from langchain.llms import Ollama
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

def build_qa_chain(vectorstore, model_name):
    # llm = Ollama(model=model_name)
    # llm = OllamaLLM(model=model_name)
    llm = OllamaLLM(
        model=model_name,
        base_url="http://host.docker.internal:11434"
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    return qa