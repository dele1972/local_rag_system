# config.py

## 📋 Übersicht

- **Datei:** `config.py`
- **Zeilen:** 547
- **Analysiert:** 2025-06-24T10:56:33

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

- ⚙️ **`get_supported_file_types(self)`**
  - Gibt eine Liste der unterstützten Dateiformate zurück

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

- ⚙️ **`get_available_embedding_models(self)`**
  - Gibt alle verfügbaren Embedding-Modelle zurück

- ⚙️ **`get_embedding_model_info(self, model_name)`**
  - Gibt detaillierte Informationen zu einem Embedding-Modell zurück

- ⚙️ **`get_best_embedding_model_for_german(self)`**
  - Gibt das beste verfügbare Embedding-Modell für deutsche Texte zurück

- ⚙️ **`get_embedding_models_by_quality(self)`**
  - Gruppiert Embedding-Modelle nach deutscher Qualität

- ⚙️ **`get_similarity_threshold(self, embedding_model)`**
  - Gibt den empfohlenen Similarity-Threshold für ein Embedding-Modell zurück

- ⚙️ **`get_optimal_chunk_size(self, embedding_model, file_size_mb = None)`**
  - Bestimmt optimale Chunk-Größe basierend auf Embedding-Modell und Dateigröße

- ⚙️ **`get_retrieval_strategy(self, embedding_model, document_count = 1)`**
  - Bestimmt optimale Retrieval-Strategie

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

- ⚙️ **`log_embedding_selection(self, embedding_model)`**
  - Loggt Informationen zur Embedding-Modell-Auswahl

**Private Methoden:**
- `__init__()` - Führt aus: self.base_path = Pa...
- `_setup_logging()` - Konfiguriert das Logging-Syste...
