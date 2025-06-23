# connection_utils.py

## 📋 Übersicht

- **Datei:** `connection_utils.py`
- **Zeilen:** 96
- **Analysiert:** 2025-06-23T13:19:59

## 📦 Imports

### Third Party
- `import logging`
- `import requests`
- `import time`

### Local/App
- `from app.config import config`

## 🔧 Konstanten & Variablen

- 📝 **`logger`** = `logging.getLogger(__name__)`

## ⚙️ Funktionen

### Öffentliche Funktionen

#### `check_ollama_connection_with_retry(max_retries = 3, delay = 1.0, timeout = 5)`

**Beschreibung:** Prüft Ollama-Verbindung mit Retry-Mechanismus

#### `wait_for_ollama_ready(max_wait_time = 30, check_interval = 2)`

**Beschreibung:** Wartet bis Ollama bereit ist oder Timeout erreicht wird

#### `get_ollama_status()`

**Beschreibung:** Gibt detaillierten Ollama-Status zurück
