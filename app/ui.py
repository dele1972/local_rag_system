# app/ui.py - Erweiterte Version mit Token-Tracking und robuster Ollama-Verbindung

import os
import datetime
import gradio as gr
import threading
import time
import re
import html
import markdown
from markdown.extensions import codehilite, fenced_code, tables, toc
import bleach

from app.loader import load_documents_from_path
from app.vectorstore import (
    build_vectorstore, load_vectorstore, delete_vectorstore,
    list_vectorstores
)
from app.rag import build_qa_chain, get_chain_type_description
from app.config import config
from app.connection_utils import (
    check_ollama_connection_with_retry, 
    wait_for_ollama_ready, 
    get_ollama_status
)
from app.token_tracker import token_tracker

# === Globale Variablen ===
qa_chain = None
current_vectorstore = None
current_model = None
current_chain_type = "stuff"
_ollama_status_cache = {'connected': False, 'last_check': 0, 'cache_duration': 10}

# === Sichere Markdown-Verarbeitung ===

# Erlaubte HTML-Tags und Attribute für Bleach
ALLOWED_TAGS = [
    'p', 'br', 'strong', 'b', 'em', 'i', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'blockquote', 'code', 'pre', 'a', 'table', 'thead', 'tbody',
    'tr', 'td', 'th', 'div', 'span', 'hr'
]

ALLOWED_ATTRIBUTES = {
    'a': ['href', 'title'],
    'code': ['class'],
    'pre': ['class'],
    'div': ['class'],
    'span': ['class'],
    'table': ['class'],
    'th': ['align'],
    'td': ['align']
}

def sanitize_and_format_text(text):
    """
    Konvertiert Markdown zu HTML und sanitized das Ergebnis für sichere Anzeige.
    
    Args:
        text (str): Der zu formatierende Text (kann Markdown enthalten)
        
    Returns:
        str: Sicherer HTML-Text für die Anzeige
    """
    if not text or not isinstance(text, str):
        return ""
    
    try:
        # 1. Markdown zu HTML konvertieren
        md = markdown.Markdown(
            extensions=[
                'codehilite',
                'fenced_code', 
                'tables',
                'toc',
                'nl2br'  # Zeilenumbrüche beibehalten
            ],
            extension_configs={
                'codehilite': {
                    'css_class': 'highlight',
                    'use_pygments': False  # Vermeidet externe Abhängigkeiten
                }
            }
        )
        
        html_content = md.convert(text)
        
        # 2. HTML sanitizen (entfernt potentiell schädliche Inhalte)
        clean_html = bleach.clean(
            html_content,
            tags=ALLOWED_TAGS,
            attributes=ALLOWED_ATTRIBUTES,
            strip=True  # Entfernt unerlaubte Tags komplett
        )
        
        # 3. Zusätzliche Bereinigung für bessere Darstellung
        clean_html = clean_html.replace('\n\n', '\n')  # Doppelte Zeilenumbrüche reduzieren
        
        return clean_html
        
    except Exception as e:
        # Fallback: Text escapen falls Markdown-Verarbeitung fehlschlägt
        print(f"Markdown-Verarbeitung fehlgeschlagen: {e}")
        return html.escape(text).replace('\n', '<br>')

def format_answer_with_sources_and_tokens(answer, sources, model_info, token_info):
    """
    Formatiert die Antwort mit Quellen, Modellinformationen und Token-Details als HTML.
    
    Args:
        answer (str): Die Hauptantwort
        sources (list): Liste der Quelldateien
        model_info (str): Informationen über das verwendete Modell
        token_info (str): Token-Verbrauchsinformationen
        
    Returns:
        str: Formatierte HTML-Antwort
    """
    # Hauptantwort formatieren
    formatted_answer = sanitize_and_format_text(answer)
    
    # Token-Informationen formatieren (Markdown zu HTML)
    formatted_token_info = sanitize_and_format_text(token_info)
    
    # Quellen-Sektion
    sources_html = ""
    if sources:
        unique_sources = list(set(sources))
        sources_list = []
        for source in unique_sources:
            # Nur den Dateinamen anzeigen, nicht den vollständigen Pfad
            filename = os.path.basename(source)
            sources_list.append(f"<li><code>{html.escape(filename)}</code></li>")
        
        sources_html = f"""
<hr>
<h4>📚 Quellen:</h4>
<ul>
{''.join(sources_list)}
</ul>
"""
    
    # Token-Sektion
    token_html = f"""
<hr>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007acc;">
{formatted_token_info}
</div>
"""
    
    # Modell-Info
    model_html = f"""
<hr>
<p><small><strong>🤖 Modell:</strong> {html.escape(model_info)}</small></p>
"""
    
    return f"{formatted_answer}{sources_html}{token_html}{model_html}"

# === Verbesserte Verbindungsprüfung ===

def get_cached_ollama_status():
    """
    Cached Ollama-Status um häufige Netzwerk-Calls zu vermeiden
    """
    current_time = time.time()
    
    # Cache für 10 Sekunden
    if (current_time - _ollama_status_cache['last_check']) < _ollama_status_cache['cache_duration']:
        return _ollama_status_cache['connected']
    
    # Status aktualisieren
    status = get_ollama_status()
    _ollama_status_cache['connected'] = status['connected']
    _ollama_status_cache['last_check'] = current_time
    
    return status['connected']

def check_ollama_with_user_feedback():
    """
    Ollama-Verbindung mit Benutzer-Feedback prüfen
    """
    status_info = get_ollama_status()
    
    if not status_info['connected']:
        # Retry-Versuch
        if check_ollama_connection_with_retry(max_retries=3, delay=1.0):
            return True, "✅ Verbindung zu Ollama hergestellt"
        else:
            return False, f"❌ {status_info['status_message']}\n💡 Tipp: Stelle sicher, dass Ollama läuft und erreichbar ist"
    
    return True, status_info['status_message']

# === Token-Statistik-Funktionen ===

def get_session_stats():
    """Gibt Session-Statistiken zurück"""
    stats = token_tracker.get_session_summary()
    
    return f"""📊 **Session-Statistiken:**
• Anfragen: {stats['total_requests']}
• Gesamt-Token: {stats['total_tokens']:,}
  - Input: {stats['total_input_tokens']:,}
  - Output: {stats['total_output_tokens']:,}
• Ø Token/Anfrage: {stats['average_tokens_per_request']}
• Ø Verarbeitungszeit: {stats['average_processing_time']}s
• Session-Dauer: {stats['session_duration']}
• Geschätzte Gesamtkosten: ${stats['estimated_total_cost']}"""

def get_recent_requests_table():
    """Erstellt eine HTML-Tabelle mit den letzten Anfragen"""
    recent = token_tracker.get_recent_requests(10)
    
    if not recent:
        return "<p><em>Noch keine Anfragen in dieser Session.</em></p>"
    
    table_rows = []
    for req in recent:
        table_rows.append(f"""
        <tr>
            <td>{req['timestamp']}</td>
            <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{html.escape(req['question'])}">{html.escape(req['question'][:50])}...</td>
            <td>{req['input_tokens']:,}</td>
            <td>{req['output_tokens']:,}</td>
            <td><strong>{req['total_tokens']:,}</strong></td>
            <td>{req['processing_time']}s</td>
            <td>${req['cost_estimate']}</td>
        </tr>
        """)
    
    return f"""
    <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
        <thead>
            <tr style="background-color: #f8f9fa;">
                <th style="border: 1px solid #ddd; padding: 8px;">Zeit</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Frage</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Input</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Output</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Gesamt</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Zeit</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Kosten</th>
            </tr>
        </thead>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>
    """

def reset_token_stats():
    """Setzt die Token-Statistiken zurück"""
    token_tracker.reset_session()
    return (
        get_session_stats(),
        get_recent_requests_table(),
        "🔄 Token-Statistiken wurden zurückgesetzt."
    )

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
        return gr.update(choices=[], value=None), "ℹ️ Kein Vektorspeicher gefunden."
    
    return (
        gr.update(choices=available_vs, value=available_vs[0]), 
        f"✅ {len(available_vs)} Vektorspeicher gefunden."
    )

# === Model Loading ===

def load_model_only(model, chain_type):
    """Lädt nur das Modell ohne Dokumente oder Vektorspeicher."""
    global current_model, current_chain_type
    
    connected, status_msg = check_ollama_with_user_feedback()
    if not connected:
        return f"⚠️ {status_msg}"
    
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
    """Lädt Dokumente und erstellt einen neuen Vektorspeicher mit verbesserter Verbindungsprüfung."""
    global qa_chain, current_vectorstore, current_model, current_chain_type
    current_model, current_chain_type = model, chain_type

    # Verbesserte Verbindungsprüfung
    connected, status_msg = check_ollama_with_user_feedback()
    if not connected:
        return (
            f"⚠️ {status_msg}",
            gr.update(),
            gr.update()
        )

    try:
        vs_path = os.path.join(get_vectorstore_base_path(), vectorstore_name)
        docs, loaded_files = load_documents_from_path(doc_path)

        if not docs:
            return (
                "⚠️ Keine unterstützten Dokumente gefunden",
                gr.update(),
                gr.update()
            )

        current_vectorstore = build_vectorstore(docs, model, vs_path)
        qa_chain = build_qa_chain(current_vectorstore, model, chain_type)

        # Nach erfolgreichem Erstellen: Dropdown aktualisieren
        available_vs = get_available_vectorstores()
        
        success_msg = f"""✅ Setup erfolgreich abgeschlossen!
🤖 Modell: {model} (Chain-Typ: {chain_type})
ℹ️ {get_chain_type_description(chain_type)}
📚 {len(docs)} Dokumentenabschnitte aus {len(loaded_files)} Dateien verarbeitet
💾 Vektorspeicher '{vectorstore_name}' gespeichert"""

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
    """Lädt einen vorhandenen Vektorspeicher mit verbesserter Verbindungsprüfung."""
    global qa_chain, current_vectorstore, current_model, current_chain_type
    current_model, current_chain_type = model, chain_type

    if not selection:
        return "⚠️ Bitte wähle einen Vektorspeicher aus"

    connected, status_msg = check_ollama_with_user_feedback()
    if not connected:
        return f"⚠️ {status_msg}"

    try:
        vs_base_dir = get_vectorstore_base_path()
        vectorstores = list_vectorstores(vs_base_dir)
        path = vectorstores.get(selection)

        if not path:
            return f"⚠️ Vektorspeicher '{selection}' nicht gefunden"

        current_vectorstore = load_vectorstore(path, model)
        qa_chain = build_qa_chain(current_vectorstore, model, chain_type)

        return f"""✅ Vektorspeicher '{selection}' erfolgreich geladen
🤖 Modell '{model}' mit Chain-Typ '{chain_type}' initialisiert
ℹ️ {get_chain_type_description(chain_type)}"""

    except Exception as e:
        return f"⚠️ Fehler beim Laden: {str(e)}"

# === System-Status-Check ===

def get_system_status():
    """Gibt aktuellen System-Status zurück"""
    status_info = get_ollama_status()
    
    ollama_status = status_info['status_message']
    
    if current_model and qa_chain:
        qa_status = f"✅ QA-System bereit (Modell: {current_model})"
    elif current_model:
        qa_status = "⚠️ Modell geladen, aber kein Vektorspeicher"
    else:
        qa_status = "⭕ Kein Modell geladen"
    
    return f"{ollama_status}\n{qa_status}"

def refresh_system_status():
    """Aktualisiert den System-Status"""
    # Cache leeren für frischen Status
    _ollama_status_cache['last_check'] = 0
    return get_system_status()

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
            available_vs = get_available_vectorstores()
            
            if available_vs:
                new_selection = available_vs[0]
                dropdown_update = gr.update(choices=available_vs, value=new_selection)
                vector_status = f"✅ '{selection}' gelöscht. Auswahl: '{new_selection}'"
            else:
                dropdown_update = gr.update(choices=[], value=None)
                vector_status = "ℹ️ Alle Vektorspeicher gelöscht."
            
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
    """Verarbeitet eine Frage über die geladenen Dokumente mit formatierter HTML-Ausgabe und Token-Tracking."""
    global qa_chain, current_model, current_chain_type

    if qa_chain is None:
        return (
            "⚠️ Bitte zuerst Modell und Vektorspeicher laden.",
            get_session_stats(),
            get_recent_requests_table()
        )

    if not question.strip():
        return (
            "⚠️ Bitte eine Frage eingeben.",
            get_session_stats(),
            get_recent_requests_table()
        )

    try:
        result = qa_chain.invoke({"query": question})
        answer = result['result']
        
        # Quellen extrahieren
        sources = []
        for doc in result.get('source_documents', []):
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                sources.append(doc.metadata['source'])
        
        # Modellinformationen
        model_info = f"{current_model}, Chain-Typ: {current_chain_type}"
        
        # Token-Informationen aus dem Ergebnis
        token_info = result.get('token_info', 'Token-Informationen nicht verfügbar')
        
        # Formatierte Antwort mit HTML erstellen
        formatted_answer = format_answer_with_sources_and_tokens(
            answer, sources, model_info, token_info
        )
        
        # Token-Statistiken aktualisieren
        session_stats = get_session_stats()
        recent_table = get_recent_requests_table()
        
        return formatted_answer, session_stats, recent_table

    except Exception as e:
        error_msg = f"⚠️ Fehler bei der Frageverarbeitung: {str(e)}"
        return (
            html.escape(error_msg),
            get_session_stats(),
            get_recent_requests_table()
        )

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
                    
                    status = gr.Textbox(
                        label="Status-Meldungen", 
                        lines=6,
                        value="Initialisiere System..."
                    )
                    
                    # Button zum manuellen Status-Refresh
                    refresh_status_btn = gr.Button("🔄 Status aktualisieren", size="sm")
                    
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
                        value="Lade Vektorspeicher-Liste..."
                    )
                    
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Liste aktualisieren")
                        load_vs_btn = gr.Button("📂 Vektorspeicher laden")
                        delete_vs_btn = gr.Button("🗑️ Löschen", variant="stop")

        with gr.Tab("Fragen und Antworten"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Dokumente befragen")
                    question = gr.Textbox(
                        label="Frage an die Dokumente", 
                        placeholder="Was ist der Hauptinhalt der Dokumente?",
                        lines=2
                    )
                    ask_btn = gr.Button("🔍 Frage stellen", variant="primary")
                    
                    # HTML-Komponente für formatierte Antworten verwenden
                    answer = gr.HTML(
                        label="Antwort",
                        value="<p><em>Hier erscheint die formatierte Antwort mit Token-Informationen...</em></p>"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Token-Statistiken")
                    
                    session_stats = gr.HTML(
                        label="Session-Übersicht",
                        value="<p><em>Noch keine Statistiken verfügbar.</em></p>"
                    )
                    
                    reset_stats_btn = gr.Button("🔄 Statistiken zurücksetzen", size="sm")
                    
                    gr.Markdown("### 📈 Letzte Anfragen")
                    recent_requests = gr.HTML(
                        label="Anfrage-Historie",
                        value="<p><em>Noch keine Anfragen.</em></p>"
                    )

        with gr.Tab("Token-Analytics"):
            gr.Markdown("### 📊 Detaillierte Token-Analyse")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Session-Übersicht")
                    detailed_stats = gr.HTML(
                        value="<p><em>Starte eine Unterhaltung, um Statistiken zu sehen.</em></p>"
                    )
                
                with gr.Column():
                    gr.Markdown("#### Anfrage-Historie")
                    detailed_history = gr.HTML(
                        value="<p><em>Anfrage-Historie wird hier angezeigt.</em></p>"
                    )
            
            refresh_analytics_btn = gr.Button("🔄 Analytics aktualisieren")

        # === Event-Handler ===
        
        # Status aktualisieren
        refresh_status_btn.click(
            fn=refresh_system_status,
            inputs=[],
            outputs=[status]
        )
        
        # Modell laden
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
            fn=lambda: create_vectorstore_name(),
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
            outputs=[answer, session_stats, recent_requests]
        )
        
        question.submit(
            fn=ask_question,
            inputs=[question],
            outputs=[answer, session_stats, recent_requests]
        )
        
        # Token-Statistiken zurücksetzen
        reset_stats_btn.click(
            fn=reset_token_stats,
            inputs=[],
            outputs=[session_stats, recent_requests, status]
        )
        
        # Analytics aktualisieren
        refresh_analytics_btn.click(
            fn=lambda: (get_session_stats(), get_recent_requests_table()),
            inputs=[],
            outputs=[detailed_stats, detailed_history]
        )

        # Beim Laden der Seite: Initialisierung mit Verzögerung
        demo.load(
            fn=lambda: (
                refresh_system_status(), 
                *update_vectorstore_dropdown_choices(),
                get_session_stats(),
                get_recent_requests_table()
            ),
            inputs=[],
            outputs=[status, vectorstore_list, vector_status, session_stats, recent_requests]
        )

    return demo

def start_ui():
    """Startet die Benutzeroberfläche mit Ollama-Warteschleife."""
    # Optional: Warte auf Ollama beim Start
    if not wait_for_ollama_ready(max_wait_time=10):
        print("⚠️ Warnung: Ollama scheint nicht bereit zu sein, starte UI trotzdem...")
    
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)