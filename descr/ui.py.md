# ui.py

## 📋 Übersicht

- **Datei:** `ui.py`
- **Zeilen:** 738
- **Analysiert:** 2025-06-17T14:46:23

## 📦 Imports

### Standard Library
- `import datetime`
- `import os`
- `import re`

### Third Party
- `from markdown.extensions import codehilite, fenced_code, tables, toc`
- `import bleach`
- `import gradio as gr`
- `import html`
- `import markdown`
- `import threading`
- `import time`

### Local/App
- `from app.config import config`
- `from app.connection_utils import check_ollama_connection_with_retry, wait_for_ollama_ready, get_ollama_status`
- `from app.loader import load_documents_from_path`
- `from app.rag import build_qa_chain, get_chain_type_description`
- `from app.token_tracker import token_tracker`
- `from app.vectorstore import build_vectorstore, load_vectorstore, delete_vectorstore, list_vectorstores`

## 🔧 Konstanten & Variablen

- 📝 **`qa_chain`** = `None`
- 📝 **`current_vectorstore`** = `None`
- 📝 **`current_model`** = `None`
- 📝 **`current_chain_type`** = `'stuff'`
- 📝 **`_ollama_status_cache`** = `{'connected': False, 'last_check': 0, 'cache_durat`
- 🔒 **`ALLOWED_TAGS`** = `['p', 'br', 'strong', 'b', 'em', 'i', 'u', 'h1', '`
- 🔒 **`ALLOWED_ATTRIBUTES`** = `{'a': ['href', 'title'], 'code': ['class'], 'pre':`

## ⚙️ Funktionen

### Öffentliche Funktionen

#### `sanitize_and_format_text(text)`

**Beschreibung:** Konvertiert Markdown zu HTML und sanitized das Ergebnis für sichere Anzeige.

#### `format_answer_with_sources_and_tokens(answer, sources, model_info, token_info)`

**Beschreibung:** Formatiert die Antwort mit Quellen, Modellinformationen und Token-Details als HTML.

#### `get_cached_ollama_status()`

**Beschreibung:** Cached Ollama-Status um häufige Netzwerk-Calls zu vermeiden

#### `check_ollama_with_user_feedback()`

**Beschreibung:** Ollama-Verbindung mit Benutzer-Feedback prüfen

#### `get_session_stats()`

**Beschreibung:** Gibt Session-Statistiken zurück

#### `get_recent_requests_table()`

**Beschreibung:** Erstellt eine HTML-Tabelle mit den letzten Anfragen

#### `reset_token_stats()`

**Beschreibung:** Setzt die Token-Statistiken zurück

#### `create_vectorstore_name()`

**Beschreibung:** Erstellt einen eindeutigen Namen für einen neuen Vektorspeicher.

#### `get_vectorstore_base_path()`

**Beschreibung:** Gibt den Basispfad für Vektorspeicher zurück.

#### `get_available_vectorstores()`

**Beschreibung:** Gibt eine Liste aller verfügbaren Vektorspeicher zurück.

#### `update_vectorstore_dropdown_choices()`

**Beschreibung:** Aktualisiert die Dropdown-Auswahlmöglichkeiten für Vektorspeicher.

#### `load_model_only(model, chain_type)`

**Beschreibung:** Lädt nur das Modell ohne Dokumente oder Vektorspeicher.

#### `setup_qa(model, doc_path, chain_type, vectorstore_name)`

**Beschreibung:** Lädt Dokumente und erstellt einen neuen Vektorspeicher mit verbesserter Verbindungsprüfung.

#### `load_existing_vectorstore(model, selection, chain_type)`

**Beschreibung:** Lädt einen vorhandenen Vektorspeicher mit verbesserter Verbindungsprüfung.

#### `get_system_status()`

**Beschreibung:** Gibt aktuellen System-Status zurück

#### `refresh_system_status()`

**Beschreibung:** Aktualisiert den System-Status

#### `delete_selected_vectorstore(selection)`

**Beschreibung:** Löscht den ausgewählten Vektorspeicher und aktualisiert die Dropdown-Liste.

#### `refresh_vectorstore_list()`

**Beschreibung:** Aktualisiert explizit die Vektorspeicher-Liste.

#### `ask_question(question)`

**Beschreibung:** Verarbeitet eine Frage über die geladenen Dokumente mit formatierter HTML-Ausgabe und Token-Tracking.

#### `create_interface()`

**Beschreibung:** Erstellt die Gradio-Benutzeroberfläche.

#### `start_ui()`

**Beschreibung:** Startet die Benutzeroberfläche mit Ollama-Warteschleife.
