# app/ui.py
import gradio as gr
import os
from app.loader import load_documents_from_path
from app.vectorstore import (
    build_vectorstore, load_vectorstore, delete_vectorstore, 
    list_vectorstores
)
from app.rag import build_qa_chain, check_ollama_connection, get_chain_type_description
from app.config import config

qa_chain = None
current_vectorstore = None
current_model = None
current_chain_type = "stuff"

def setup_qa(model, doc_path, chain_type, vectorstore_name):
    global qa_chain, current_vectorstore, current_model, current_chain_type
    
    # Setze globale Variablen
    current_model = model
    current_chain_type = chain_type
    
    # Check Ollama connection
    if not check_ollama_connection():
        return f"⚠️ Fehler: Keine Verbindung zu Ollama unter {config.get_ollama_base_url()}"
    
    try:
        # Vector store Basis-Verzeichnis
        vs_base_dir = os.path.join(os.path.dirname(doc_path), ".vectorstores")
        os.makedirs(vs_base_dir, exist_ok=True)
        
        # Vollständiger Pfad zum Vektorspeicher
        vs_path = os.path.join(vs_base_dir, vectorstore_name)
        
        # Load documents and build vector store
        docs, loaded_files = load_documents_from_path(doc_path)
        if not docs:
            return "⚠️ Keine unterstützten Dokumente gefunden (.pdf, .txt, .md)"
        
        # Build vector store with Chroma
        current_vectorstore = build_vectorstore(docs, model, vs_path)
        
        # Build QA chain with selected chain type
        qa_chain = build_qa_chain(current_vectorstore, model, chain_type)
        
        chain_desc = get_chain_type_description(chain_type)
        
        return f"✅ Modell '{model}' mit Chain-Typ '{chain_type}' geladen\n" \
               f"ℹ️ {chain_desc}\n" \
               f"📚 {len(docs)} Dokumentenabschnitte aus {len(loaded_files)} Dateien verarbeitet.\n" \
               f"💾 Vektorstore '{vectorstore_name}' gespeichert unter {vs_path}"
    
    except Exception as e:
        return f"⚠️ Fehler: {str(e)}"

def load_existing_vectorstore(model, vectorstore_path, chain_type):
    global qa_chain, current_vectorstore, current_model, current_chain_type
    
    # Setze globale Variablen
    current_model = model
    current_chain_type = chain_type
    
    # Check Ollama connection
    if not check_ollama_connection():
        return f"⚠️ Fehler: Keine Verbindung zu Ollama unter {config.get_ollama_base_url()}"
    
    try:
        # Lade vorhandenen Vektorspeicher
        current_vectorstore = load_vectorstore(vectorstore_path, model)
        
        # Build QA chain with selected chain type
        qa_chain = build_qa_chain(current_vectorstore, model, chain_type)
        
        chain_desc = get_chain_type_description(chain_type)
        
        return f"✅ Vektorspeicher aus {vectorstore_path} geladen\n" \
               f"✅ Modell '{model}' mit Chain-Typ '{chain_type}' initialisiert\n" \
               f"ℹ️ {chain_desc}"
    
    except Exception as e:
        return f"⚠️ Fehler beim Laden des Vektorspeichers: {str(e)}"

def delete_vs(vectorstore_path):
    if delete_vectorstore(vectorstore_path):
        return f"✅ Vektorspeicher unter {vectorstore_path} wurde gelöscht"
    else:
        return f"⚠️ Vektorspeicher unter {vectorstore_path} nicht gefunden"

def refresh_vectorstores(doc_path):
    vs_base_dir = os.path.join(os.path.dirname(doc_path), ".vectorstores")
    vectorstores = list_vectorstores(vs_base_dir)
    
    vs_choices = []
    for name, path in vectorstores.items():
        vs_choices.append({"name": name, "path": path})
    
    return vs_choices

def ask_question(question):
    global qa_chain, current_model, current_chain_type
    
    if qa_chain is None:
        return "⚠️ Bitte zuerst Modell und Dokumente laden."
    
    try:
        result = qa_chain.invoke({"query": question})
        
        # Format answer with sources
        answer = result['result']
        sources = []
        
        # Extract source document information
        if 'source_documents' in result:
            for doc in result['source_documents']:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    sources.append(doc.metadata['source'])
        
        # Format the response
        model_info = f"Modell: {current_model}, Chain-Typ: {current_chain_type}"
        
        if sources:
            unique_sources = list(set(sources))
            source_text = "\n\nQuellen:\n" + "\n".join([f"- {s}" for s in unique_sources])
            return f"{answer}\n\n{source_text}\n\n{model_info}"
        else:
            return f"{answer}\n\n{model_info}"
    
    except Exception as e:
        return f"⚠️ Fehler bei der Verarbeitung der Frage: {str(e)}"

def create_vectorstore_name():
    """Erzeugt einen eindeutigen Namen für einen neuen Vektorspeicher"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"vectorstore_{timestamp}"

def start_ui():
    global current_model, current_chain_type
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Lokales RAG mit Ollama und LangChain")
        
        with gr.Tab("Einrichtung"):
            with gr.Row():
                with gr.Column():
                    model = gr.Dropdown(
                        choices=config.get_available_models(), 
                        label="Modellwahl",
                        value=config.get_available_models()[0] if config.get_available_models() else None
                    )
                    
                    chain_type = gr.Dropdown(
                        choices=["stuff", "map_reduce", "refine", "map_rerank"],
                        label="Chain-Typ",
                        value="stuff",
                        info="Wähle die Art der Verarbeitungskette"
                    )
                    
                    doc_path = gr.Textbox(
                        label="Pfad zur Dokumentenbasis", 
                        value=config.get_documents_path(),
                        placeholder="/pfad/zu/dokumenten"
                    )
                    
                    vectorstore_name = gr.Textbox(
                        label="Name für neuen Vektorspeicher",
                        value=create_vectorstore_name(),
                        placeholder="mein_vektorspeicher"
                    )
                    
                    with gr.Row():
                        load_btn = gr.Button("📚 Dokumente laden und vektorisieren", variant="primary")
                        refresh_btn = gr.Button("🔄 Vektorspeicher aktualisieren")
                
                with gr.Column():
                    status = gr.Textbox(label="Status", lines=5)
                    
                    # Show Ollama connection status
                    ollama_status = "✅ Verbunden" if check_ollama_connection() else "❌ Nicht verbunden"
                    gr.Markdown(f"**Ollama Status:** {ollama_status} ({config.get_ollama_base_url()})")
                    
                    # Vorhandene Vektorspeicher anzeigen
                    vectorstore_list = gr.Dropdown(
                        label="Vorhandene Vektorspeicher",
                        info="Wähle einen gespeicherten Vektorspeicher zum Laden",
                        type="index",
                        choices=[]
                    )
                    
                    with gr.Row():
                        load_vs_btn = gr.Button("📂 Vektorspeicher laden")
                        delete_vs_btn = gr.Button("🗑️ Vektorspeicher löschen", variant="stop")
        
        with gr.Tab("Fragen und Antworten"):
            question = gr.Textbox(
                label="Deine Frage an die Dokumente",
                placeholder="Was ist der Hauptinhalt dieser Dokumente?"
            )
            ask_btn = gr.Button("🔍 Frage stellen", variant="primary")
            answer = gr.Textbox(label="Antwort", lines=15)

        # Event Handler
        load_btn.click(setup_qa, inputs=[model, doc_path, chain_type, vectorstore_name], outputs=status)
        
        # Vectorstore management
        refresh_btn.click(
            refresh_vectorstores, 
            inputs=[doc_path], 
            outputs=vectorstore_list
        )
        
        # Handler für das Laden eines vorhandenen Vektorspeichers
        def load_selected_vectorstore(model, vectorstore_list, chain_type, doc_path):
            vs_base_dir = os.path.join(os.path.dirname(doc_path), ".vectorstores")
            vectorstores = list_vectorstores(vs_base_dir)
            vs_items = list(vectorstores.items())
            
            if vectorstore_list is not None and 0 <= vectorstore_list < len(vs_items):
                vs_name, vs_path = vs_items[vectorstore_list]
                return load_existing_vectorstore(model, vs_path, chain_type)
            return "⚠️ Bitte einen Vektorspeicher auswählen"
        
        load_vs_btn.click(
            load_selected_vectorstore,
            inputs=[model, vectorstore_list, chain_type, doc_path],
            outputs=status
        )
        
        # Handler für das Löschen eines Vektorspeichers
        def delete_selected_vectorstore(vectorstore_list, doc_path):
            vs_base_dir = os.path.join(os.path.dirname(doc_path), ".vectorstores")
            vectorstores = list_vectorstores(vs_base_dir)
            vs_items = list(vectorstores.items())
            
            if vectorstore_list is not None and 0 <= vectorstore_list < len(vs_items):
                vs_name, vs_path = vs_items[vectorstore_list]
                result = delete_vs(vs_path)
                # Aktualisiere die Liste nach dem Löschen
                return result
            return "⚠️ Bitte einen Vektorspeicher zum Löschen auswählen"
        
        delete_vs_btn.click(
            delete_selected_vectorstore,
            inputs=[vectorstore_list, doc_path],
            outputs=status
        )
        
        # Fragenverarbeitung
        ask_btn.click(ask_question, inputs=question, outputs=answer)
        question.submit(ask_question, inputs=question, outputs=answer)
        
        # Initialize by trying to load available vectorstores
        demo.load(
            refresh_vectorstores,
            inputs=[doc_path],
            outputs=[vectorstore_list]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)