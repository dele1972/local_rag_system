# rag.py
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from app.config import config
import requests

def check_ollama_connection():
    """Check if Ollama is accessible"""
    try:
        response = requests.get(f"{config.get_ollama_base_url()}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_prompt_template():
    """Define a prompt template for RAG"""
    template = """Du bist ein hilfreicher Assistent, der Fragen basierend auf den bereitgestellten Dokumenten beantwortet.
    
    Kontext:
    {context}
    
    Frage: {question}
    
    Beantworte die Frage nur basierend auf dem gegebenen Kontext. Wenn die Antwort nicht im Kontext gefunden werden kann, 
    sage "Ich kann diese Frage nicht auf Basis der verfügbaren Dokumente beantworten." und gib keine Spekulationen ab.
    
    Deine Antwort:"""
    
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

def build_qa_chain(vectorstore, model_name, chain_type="stuff"):
    """Build a question-answering chain with error handling and customized chain type
    
    Args:
        vectorstore: Der Vektorspeicher mit Dokumenten
        model_name: Name des zu verwendenden Ollama-Modells
        chain_type: Art der Chain ('stuff', 'map_reduce', 'refine', 'map_rerank')
    """
    if not check_ollama_connection():
        raise ConnectionError(f"Cannot connect to Ollama at {config.get_ollama_base_url()}")
    
    llm = OllamaLLM(
        model=model_name,
        base_url=config.get_ollama_base_url()
    )
    
    # Verwende das angepasste Prompt-Template
    prompt = get_prompt_template()
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,  # Include source documents in response
        chain_type_kwargs={"prompt": prompt} if chain_type == "stuff" else {}
    )
    
    return qa

def get_chain_type_description(chain_type):
    """Gibt eine Beschreibung für jeden Chain-Typ zurück"""
    descriptions = {
        "stuff": "Standard für kleine Dokumente - Fügt alle Dokumente in einen Prompt ein",
        "map_reduce": "Für große Dokumente - Verarbeitet jedes Dokument einzeln und kombiniert die Ergebnisse",
        "refine": "Iterativer Ansatz - Verfeinert die Antwort schrittweise mit jedem Dokument",
        "map_rerank": "Bewertet Antworten - Ordnet Antworten nach Relevanz"
    }
    return descriptions.get(chain_type, "Keine Beschreibung verfügbar")