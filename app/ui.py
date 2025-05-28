# app/ui.py

import os
import datetime
import gradio as gr

from app.loader import load_documents_from_path
from app.vectorstore import (
    build_vectorstore, load_vectorstore, delete_vectorstore,
    list_vectorstores
)
from app.rag import build_qa_chain, check_ollama_connection, get_chain_type_description
from app.config import config

# === Globale Variablen ===
qa_chain = None
current_vectorstore = None
current_model = None
current_chain_type = "stuff"

# === Hilfsfunktionen ===

def create_vectorstore_name():
    """Erstellt einen eindeutigen Namen für einen neuen Vektorspeicher."""
    return f"vectorstore_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

def get_vectorstore_base_path():
    """Gibt den Basispfad für Vektorspeicher zurück."""
    return os.path.join(config.base_path, ".vectorstores")

def get_available_vectorstores():
    """Gibt eine Liste aller verfügbaren Vektorspeicher zurück."""
    vs_base_dir = get_vectorstore_base_path()
    vectorstores_dict = list_vectorstores(vs_base_dir)
    return list(vectorstores_dict.keys())

def update_vectorstore_dropdown_choices():
    """Aktualisiert die Dropdown-Auswahlmöglichkeiten für Vektorspeicher."""
    available_vs = get_available_vectorstores()
    
    if not available_vs:
        return gr.update(choices=[], value=None), "ℹ️ Kein Vektorspeicher geladen. Bitte laden Sie einen Vektorspeicher für die weitere Anwendung."
    
    return (
        gr.update(choices=available_vs, value=available_vs[0]), 
        f"✅ {len(available_vs)} Vektorspeicher gefunden, aber noch keiner geladen."
    )

# === Model Loading ===

def load_model_only(model, chain_type):
    """Lädt nur das Modell ohne Dokumente oder Vektorspeicher."""
    global current_model, current_chain_type
    
    if not check_ollama_connection():
        return f"⚠️ Fehler: Keine Verbindung zu Ollama unter {config.get_ollama_base_url()}"
    
    try:
        current_model = model
        current_chain_type = chain_type
        
        return f"""✅ Modell '{model}' mit Chain-Typ '{chain_type}' geladen
ℹ️ {get_chain_type_description(chain_type)}
⚠️ Hinweis: Zum Fragen beantworten muss noch ein Vektorspeicher geladen werden."""
        
    except Exception as e:
        return f"⚠️ Fehler beim Laden des Modells: {str(e)}"

# === Setup & Laden ===

def setup_qa(model, doc_path, chain_type, vectorstore_name):
    """Lädt Dokumente und erstellt einen neuen Vektorspeicher."""
    global qa_chain, current_vectorstore, current_model, current_chain_type
    current_model, current_chain_type = model, chain_type

    if not check_ollama_connection():
        return (
            f"⚠️ Fehler: Keine Verbindung zu Ollama unter {config.get_ollama_base_url()}",
            gr.update(),  # Dropdown bleibt unverändert
            gr.update()   # Status bleibt unverändert
        )

    try:
        vs_path = os.path.join(get_vectorstore_base_path(), vectorstore_name)
        docs, loaded_files = load_documents_from_path(doc_path)

        if not docs:
            return (
                "⚠️ Keine unterstützten Dokumente gefunden (.pdf, .txt, .md)",
                gr.update(),
                gr.update()
            )

        current_vectorstore = build_vectorstore(docs, model, vs_path)
        qa_chain = build_qa_chain(current_vectorstore, model, chain_type)

        # Nach erfolgreichem Erstellen: Dropdown aktualisieren
        available_vs = get_available_vectorstores()
        
        success_msg = f"""✅ Modell '{model}' mit Chain-Typ '{chain_type}' geladen
ℹ️ {get_chain_type_description(chain_type)}
📚 {len(docs)} Dokumentenabschnitte aus {len(loaded_files)} Dateien verarbeitet.
💾 Vektorspeicher '{vectorstore_name}' gespeichert unter {vs_path}"""

        return (
            success_msg,
            gr.update(choices=available_vs, value=vectorstore_name),
            f"✅ Vektorspeicher '{vectorstore_name}' wurde erstellt und geladen."
        )
    
    except Exception as e:
        return (
            f"⚠️ Fehler: {str(e)}",
            gr.update(),
            gr.update()
        )

def load_existing_vectorstore(model, selection, chain_type):
    """Lädt einen vorhandenen Vektorspeicher."""
    global qa_chain, current_vectorstore, current_model, current_chain_type
    current_model, current_chain_type = model, chain_type

    if not selection:
        return "⚠️ Bitte wähle einen Vektorspeicher aus"

    if not check_ollama_connection():
        return f"⚠️ Fehler: Keine Verbindung zu Ollama unter {config.get_ollama_base_url()}"

    try:
        vs_base_dir = get_vectorstore_base_path()
        vectorstores = list_vectorstores(vs_base_dir)
        path = vectorstores.get(selection)

        if not path:
            return f"⚠️ Vektorspeicher '{selection}' nicht gefunden"

        current_vectorstore = load_vectorstore(path, model)
        qa_chain = build_qa_chain(current_vectorstore, model, chain_type)

        return f"""✅ Vektorspeicher '{selection}' geladen
✅ Modell '{model}' mit Chain-Typ '{chain_type}' initialisiert
ℹ️ {get_chain_type_description(chain_type)}"""

    except Exception as e:
        return f"⚠️ Fehler beim Laden: {str(e)}"

# === Vektorspeicher-Verwaltung ===

def delete_selected_vectorstore(selection):
    """Löscht den ausgewählten Vektorspeicher und aktualisiert die Dropdown-Liste."""
    if not selection:
        return (
            "⚠️ Bitte einen Vektorspeicher auswählen",
            gr.update(),
            gr.update()
        )

    vs_base_dir = get_vectorstore_base_path()
    vectorstores = list_vectorstores(vs_base_dir)
    path = vectorstores.get(selection)

    if not path:
        return (
            f"⚠️ Vektorspeicher '{selection}' nicht gefunden",
            gr.update(),
            gr.update()
        )

    try:
        if delete_vectorstore(path):
            # Nach erfolgreichem Löschen: Dropdown aktualisieren
            available_vs = get_available_vectorstores()
            
            if available_vs:
                new_selection = available_vs[0]
                dropdown_update = gr.update(choices=available_vs, value=new_selection)
                vector_status = f"✅ Vektorspeicher '{selection}' gelöscht. Neuer Auswahl: '{new_selection}'"
            else:
                dropdown_update = gr.update(choices=[], value=None)
                vector_status = "ℹ️ Alle Vektorspeicher gelöscht. Bitte laden Sie einen Vektorspeicher für die weitere Anwendung."
            
            return (
                f"✅ Vektorspeicher '{selection}' erfolgreich gelöscht",
                dropdown_update,
                vector_status
            )
        else:
            return (
                f"⚠️ Fehler beim Löschen von '{selection}'",
                gr.update(),
                gr.update()
            )
    except Exception as e:
        return (
            f"⚠️ Fehler beim Löschen: {str(e)}",
            gr.update(),
            gr.update()
        )

def refresh_vectorstore_list():
    """Aktualisiert explizit die Vektorspeicher-Liste."""
    return update_vectorstore_dropdown_choices()

# === Fragen & Antworten ===

def ask_question(question):
    """Verarbeitet eine Frage über die geladenen Dokumente."""
    global qa_chain, current_model, current_chain_type

    if qa_chain is None:
        return "⚠️ Bitte zuerst Modell und Dokumente laden."

    if not question.strip():
        return "⚠️ Bitte eine Frage eingeben."

    try:
        result = qa_chain.invoke({"query": question})
        answer = result['result']
        
        # Quellen extrahieren
        sources = []
        for doc in result.get('source_documents', []):
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                sources.append(doc.metadata['source'])
        
        # Antwort formatieren
        source_info = ""
        if sources:
            unique_sources = list(set(sources))
            source_info = "\n\nQuellen:\n" + "\n".join(f"- {s}" for s in unique_sources)
        
        model_info = f"\n\nModell: {current_model}, Chain-Typ: {current_chain_type}"
        
        return f"{answer}{source_info}{model_info}"

    except Exception as e:
        return f"⚠️ Fehler bei der Frageverarbeitung: {str(e)}"

# === Benutzeroberfläche ===

def create_interface():
    """Erstellt die Gradio-Benutzeroberfläche."""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Lokales RAG System") as demo:
        gr.Markdown("# 🤖 Lokales RAG mit Ollama und LangChain")

        with gr.Tab("Einrichtung"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Modell-Konfiguration")
                    model = gr.Dropdown(
                        choices=config.get_available_models(),
                        label="Modellwahl",
                        value=config.get_available_models()[0] if config.get_available_models() else None
                    )
                    chain_type = gr.Dropdown(
                        choices=["stuff", "map_reduce", "refine", "map_rerank"],
                        label="Chain-Typ", 
                        value="stuff"
                    )
                    
                    # Neuer Button zum Laden nur des Modells
                    load_model_btn = gr.Button("🤖 Modell laden", variant="secondary")
                    
                    gr.Markdown("### Neue Dokumente laden")
                    doc_path = gr.Textbox(
                        label="Pfad zu Dokumenten", 
                        value=config.get_documents_path()
                    )
                    vectorstore_name = gr.Textbox(
                        label="Name für neuen Vektorspeicher", 
                        value=create_vectorstore_name()
                    )
                    
                    load_btn = gr.Button("📚 Dokumente laden & Vektorspeicher erstellen", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### System-Status")
                    ollama_connected = check_ollama_connection()
                    gr.Markdown(f"**Ollama Status:** {'✅ Verbunden' if ollama_connected else '❌ Nicht verbunden'}")
                    
                    status = gr.Textbox(
                        label="Status-Meldungen", 
                        lines=6,
                        value="Bereit für Konfiguration..." if ollama_connected else "⚠️ Ollama-Verbindung prüfen!"
                    )
                    
                    gr.Markdown("### Vektorspeicher-Verwaltung")
                    vectorstore_list = gr.Dropdown(
                        label="Vorhandene Vektorspeicher",
                        choices=[],
                        value=None,
                        interactive=True
                    )
                    
                    vector_status = gr.Textbox(
                        label="Vektorspeicher-Status", 
                        interactive=False,
                        value="Initialisiere..."
                    )
                    
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Liste aktualisieren")
                        load_vs_btn = gr.Button("📂 Vektorspeicher laden")
                        delete_vs_btn = gr.Button("🗑️ Löschen", variant="stop")

        with gr.Tab("Fragen und Antworten"):
            gr.Markdown("### Dokumente befragen")
            question = gr.Textbox(
                label="Frage an die Dokumente", 
                placeholder="Was ist der Hauptinhalt der Dokumente?",
                lines=2
            )
            ask_btn = gr.Button("🔍 Frage stellen", variant="primary")
            answer = gr.Textbox(
                label="Antwort", 
                lines=15,
                placeholder="Hier erscheint die Antwort..."
            )

        # === Event-Handler ===
        
        # Nur Modell laden (ohne Dokumente)
        load_model_btn.click(
            fn=load_model_only,
            inputs=[model, chain_type],
            outputs=[status]
        )
        
        # Dokumente laden und neuen Vektorspeicher erstellen
        load_btn.click(
            fn=setup_qa,
            inputs=[model, doc_path, chain_type, vectorstore_name],
            outputs=[status, vectorstore_list, vector_status]
        ).then(
            fn=lambda: create_vectorstore_name(),  # Neuen Namen generieren
            inputs=[],
            outputs=[vectorstore_name]
        )
        
        # Vektorspeicher-Liste aktualisieren
        refresh_btn.click(
            fn=refresh_vectorstore_list,
            inputs=[],
            outputs=[vectorstore_list, vector_status]
        )
        
        # Vorhandenen Vektorspeicher laden
        load_vs_btn.click(
            fn=load_existing_vectorstore,
            inputs=[model, vectorstore_list, chain_type],
            outputs=[status]
        )
        
        # Vektorspeicher löschen
        delete_vs_btn.click(
            fn=delete_selected_vectorstore,
            inputs=[vectorstore_list],
            outputs=[status, vectorstore_list, vector_status]
        )
        
        # Frage stellen
        ask_btn.click(
            fn=ask_question,
            inputs=[question],
            outputs=[answer]
        )
        
        question.submit(
            fn=ask_question,
            inputs=[question],
            outputs=[answer]
        )

        # Beim Laden der Seite: Vektorspeicher-Liste initialisieren
        demo.load(
            fn=update_vectorstore_dropdown_choices,
            inputs=[],
            outputs=[vectorstore_list, vector_status]
        )

    return demo

def start_ui():
    """Startet die Benutzeroberfläche."""
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)