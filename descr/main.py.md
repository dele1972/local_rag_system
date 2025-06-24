# main.py

## 📋 Übersicht

- **Datei:** `main.py`
- **Zeilen:** 552
- **Analysiert:** 2025-06-24T10:56:33

## 📦 Imports

### Standard Library
- `from datetime import datetime`
- `import argparse`
- `import json`
- `import os`
- `import sys`

### Third Party
- `import traceback`

### Local/App
- `from app.config import config`
- `from app.rag import debug_retrieval_quality, suggest_optimal_chain_type, get_chain_type_description, DocumentRetrievalDebugger, ChunkAnalyzer, PerformanceProfiler`
- `from app.ui import start_ui`
- `from app.vectorstore import get_vectorstore`
- `from app.vectorstore import load_documents`

## 🔧 Konstanten & Variablen

- 📝 **`logger`** = `config.get_logger('MAIN')`

## ⚙️ Funktionen

### Öffentliche Funktionen

#### `setup_debug_logging()`

**Beschreibung:** Aktiviert erweiterte Debug-Ausgaben

#### `analyze_document_collection(documents_path)`

**Beschreibung:** Führt eine umfassende Analyse der Dokumentensammlung durch

#### `run_comprehensive_system_analysis(documents_path, model_name)`

**Beschreibung:** Führt eine umfassende System-Analyse durch

#### `save_analysis_report(analysis_results, output_file = None)`

**Beschreibung:** Speichert den Analyse-Bericht in eine JSON-Datei

#### `interactive_debugging_session()`

**Beschreibung:** Startet eine interaktive Debugging-Session

#### `main()`

**Beschreibung:** Führt aus: parser = argparse.ArgumentParser(description='Loka...
