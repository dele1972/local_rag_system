# app/ui.py
import gradio as gr
import os
from app.loader import load_documents_from_path
from app.vectorstore import build_vectorstore, save_vectorstore
from app.rag import build_qa_chain, check_ollama_connection
from app.config import config

qa_chain = None

def setup_qa(model, doc_path):
    global qa_chain
    
    # Check Ollama connection
    if not check_ollama_connection():
        return f"⚠️ Fehler: Keine Verbindung zu Ollama unter {config.get_ollama_base_url()}"
    
    try:
        # Load documents and build vector store
        docs, loaded_files = load_documents_from_path(doc_path)
        if not docs:
            return "⚠️ Keine unterstützten Dokumente gefunden (.pdf, .txt, .md)"
        
        # Build vector store and QA chain
        vs = build_vectorstore(docs, model)
        qa_chain = build_qa_chain(vs, model)
        
        # Save vector store for future use
        save_dir = os.path.join(os.path.dirname(doc_path), ".vectorstore")
        os.makedirs(save_dir, exist_ok=True)
        save_vectorstore(vs, save_dir)
        
        return f"✅ Modell '{model}' wurde geladen\n📚 {len(docs)} Dokumentenabschnitte aus {len(loaded_files)} Dateien verarbeitet.\nVektorstore gespeichert unter {save_dir}"
    
    except Exception as e:
        return f"⚠️ Fehler: {str(e)}"

def ask_question(question):
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
        if sources:
            unique_sources = list(set(sources))
            source_text = "\n\nQuellen:\n" + "\n".join([f"- {s}" for s in unique_sources])
            return answer + source_text
        else:
            return answer
    
    except Exception as e:
        return f"⚠️ Fehler bei der Verarbeitung der Frage: {str(e)}"

def start_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Lokales RAG mit Ollama und LangChain")
        
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    choices=config.get_available_models(), 
                    label="Modellwahl",
                    value=config.get_available_models()[0] if config.get_available_models() else None
                )
                doc_path = gr.Textbox(
                    label="Pfad zur Dokumentenbasis", 
                    value=config.get_documents_path(),
                    placeholder="/pfad/zu/dokumenten"
                )
                load_btn = gr.Button("📚 Laden und Vektorisieren", variant="primary")
            
            with gr.Column():
                status = gr.Textbox(label="Status", lines=5)
                
                # Show Ollama connection status
                ollama_status = "✅ Verbunden" if check_ollama_connection() else "❌ Nicht verbunden"
                gr.Markdown(f"**Ollama Status:** {ollama_status} ({config.get_ollama_base_url()})")
        
        gr.Markdown("## Fragen und Antworten")
        
        question = gr.Textbox(
            label="Deine Frage an die Dokumente",
            placeholder="Was ist der Hauptinhalt dieser Dokumente?"
        )
        ask_btn = gr.Button("🔍 Frage stellen", variant="primary")
        answer = gr.Textbox(label="Antwort", lines=10)

        load_btn.click(setup_qa, inputs=[model, doc_path], outputs=status)
        ask_btn.click(ask_question, inputs=question, outputs=answer)
        
        # Also allow submitting with Enter key
        question.submit(ask_question, inputs=question, outputs=answer)

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)