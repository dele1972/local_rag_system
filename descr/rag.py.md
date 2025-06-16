# rag.py

## 📋 Übersicht

- **Datei:** `rag.py`
- **Zeilen:** 1,114
- **Analysiert:** 2025-06-16T06:50:35

## 📦 Imports

### Standard Library
- `from datetime import datetime`
- `from typing import List, Dict, Any, Tuple`
- `import json`

### Third Party
- `from langchain.chains import RetrievalQA`
- `from langchain.chains.question_answering import load_qa_chain`
- `from langchain.prompts import PromptTemplate`
- `from langchain_ollama import OllamaLLM`
- `import requests`
- `import statistics`
- `import time`

### Local/App
- `from app.config import config`
- `from app.connection_utils import check_ollama_connection_with_retry`
- `from app.token_tracker import token_tracker, TokenUsage`

## 🔧 Konstanten & Variablen

- 📝 **`logger`** = `config.get_logger('RAG')`

## ⚙️ Funktionen

### Öffentliche Funktionen

#### `check_ollama_connection()`

**Beschreibung:** Einfache Ollama-Verbindungsprüfung (Kompatibilität)

#### `get_prompt_template()`

**Beschreibung:** Define a prompt template for RAG

#### `build_qa_chain(vectorstore, model_name, chain_type = 'stuff', enable_debug = True)`

**Beschreibung:** Build a question-answering chain with comprehensive debugging capabilities

#### `get_chain_type_description(chain_type)`

**Beschreibung:** Gibt eine erweiterte Beschreibung für jeden Chain-Typ zurück

#### `suggest_optimal_chain_type(model_name, document_count = None, file_size_mb = None)`

**Beschreibung:** Erweiterte Chain-Typ-Empfehlung basierend auf mehreren Faktoren

#### `debug_retrieval_quality(vectorstore, test_questions: List[str], k_values: List[int] = None) -> Dict`

**Beschreibung:** Umfassende Analyse der Retrieval-Qualität mit Test-Fragen

## 🏗️ Klassen

### `DocumentRetrievalDebugger`

**Beschreibung:** Detailliertes Debugging für Document Retrieval

#### Methoden

- ⚙️ **`analyze_retrieval(self, question: str, documents: List, vectorstore, k_values: List[int] = None) -> Dict`**
  - Analysiert Document Retrieval mit verschiedenen k-Werten

### `ChunkAnalyzer`

**Beschreibung:** Analysiert Chunk-Qualität und -Verteilung

#### Methoden

- ⚙️ **`analyze_chunks(self, vectorstore, sample_size: int = 100) -> Dict`**
  - Analysiert die Qualität der Chunks im Vektorspeicher

**Private Methoden:**
- `__init__()` - Führt aus: self.analysis_cache...
- `_compute_chunk_statistics()` - Berechnet detaillierte Chunk-S...
- `_generate_chunk_recommendations()` - Generiert Empfehlungen für Chu...

### `PerformanceProfiler`

**Beschreibung:** Profiling für RAG-Performance

#### Methoden

- ⚙️ **`start_profiling(self, operation: str) -> str`**
  - Startet Profiling für eine Operation

- ⚙️ **`log_stage(self, stage_name: str, **kwargs)`**
  - Loggt eine Stage im aktuellen Profiling

- ⚙️ **`end_profiling(self) -> Dict`**
  - Beendet das aktuelle Profiling und gibt Statistiken zurück

**Private Methoden:**
- `__init__()` - Führt aus: self.performance_lo...
- `_analyze_performance()` - Analysiert Performance-Daten...
- `_generate_performance_recommendations()` - Generiert Performance-Empfehlu...

### `SmartContextManager`

**Beschreibung:** Intelligentes Context-Management mit erweiterten Debugging-Funktionen

#### Methoden

- ⚙️ **`prepare_context(self, documents, question, debug_info = None)`**
  - Bereitet den Kontext vor mit erweiterten Debugging-Informationen

**Private Methoden:**
- `__init__()` - Führt aus: self.token_counter ...
- `_truncate_text_to_tokens()` - Erweiterte Text-Truncation mit...

### `TokenTrackingQA`

**Beschreibung:** Erweiterte QA-Chain mit umfassendem Debugging

#### Methoden

- ⚙️ **`invoke(self, query_dict, enable_debug = True)`**
  - Erweiterte QA-Ausführung mit umfassendem Debugging
