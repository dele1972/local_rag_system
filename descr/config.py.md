# config.py

## 📋 Übersicht

- **Datei:** `config.py`
- **Zeilen:** 231
- **Analysiert:** 2025-06-17T14:46:22

## 📦 Imports

### Standard Library
- `from pathlib import Path`
- `import os`

### Third Party
- `import logging`

## 🔧 Konstanten & Variablen

- 📝 **`config`** = `ConfigManager()`

## 🏗️ Klassen

### `ConfigManager`

#### Methoden

- ⚙️ **`get_documents_path(self)`**
  - Führt aus: return self.documents_path...

- ⚙️ **`get_available_models(self)`**
  - Gibt Liste der verfügbaren Modellnamen zurück

- ⚙️ **`get_model_info(self, model_name)`**
  - Gibt detaillierte Informationen zu einem Modell zurück

- ⚙️ **`get_model_token_limit(self, model_name)`**
  - Gibt das Token-Limit für ein Modell zurück

- ⚙️ **`get_recommended_chain_type(self, model_name)`**
  - Gibt den empfohlenen Chain-Typ für ein Modell zurück

- ⚙️ **`get_models_by_capability(self)`**
  - Gruppiert Modelle nach ihren Fähigkeiten

- ⚙️ **`calculate_context_limits(self, model_name, prompt_overhead = None, answer_reserve = None)`**
  - Berechnet die verfügbaren Context-Token für ein Modell

- ⚙️ **`get_optimal_retrieval_k(self, model_name, file_count = 1)`**
  - Bestimmt optimale Anzahl von Dokumenten für Retrieval

- ⚙️ **`get_ollama_base_url(self)`**
  - Führt aus: return self.ollama_base_url...

- ⚙️ **`get_logger(self, name = None)`**
  - Gibt einen konfigurierten Logger zurück

- ⚙️ **`log_model_selection(self, model_name)`**
  - Loggt Informationen zur Modell-Auswahl

**Private Methoden:**
- `__init__()` - Führt aus: self.base_path = Pa...
- `_setup_logging()` - Konfiguriert das Logging-Syste...
