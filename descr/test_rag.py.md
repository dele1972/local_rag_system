# test_rag.py

## 📋 Übersicht

- **Datei:** `test_rag.py`
- **Zeilen:** 709
- **Analysiert:** 2025-06-23T13:19:59

### Beschreibung

```
RAG System Test Framework
=========================

Systematische Tests für verschiedene Dokumentgrößen, Chunk-Größen und Modelle.
Speziell optimiert für deutsche Dokumente und große Dateien.

Autor: RAG System Optimizer
Datum: 2025-06-16
```

## 📦 Imports

### Standard Library
- `from datetime import datetime`
- `from pathlib import Path`
- `from typing import List, Dict, Any, Tuple, Optional`
- `import argparse`
- `import json`
- `import os`
- `import sys`

### Third Party
- `from dataclasses import dataclass, asdict`
- `import logging`
- `import statistics`
- `import time`
- `import traceback`
- `import traceback`

### Local/App
- `from app.config import config`
- `from app.connection_utils import check_ollama_connection_with_retry`
- `from app.rag import build_qa_chain, suggest_optimal_chain_type, debug_retrieval_quality`
- `from app.vectorstore import load_documents, get_vectorstore, build_vectorstore`

## 🔧 Konstanten & Variablen

- 📝 **`logger`** = `logging.getLogger(__name__)`

## ⚙️ Funktionen

### Öffentliche Funktionen

#### `main()`

**Beschreibung:** Hauptfunktion für CLI-Nutzung des Test-Frameworks

## 🏗️ Klassen

### `TestConfiguration`

**Beschreibung:** Konfiguration für einen einzelnen Test

### `TestResult`

**Beschreibung:** Ergebnis eines einzelnen Tests

### `RAGTestFramework`

**Beschreibung:** Hauptklasse für RAG-System Tests

#### Methoden

- ⚙️ **`create_test_configurations(self, document_paths: List[str]) -> List[TestConfiguration]`**
  - Erstellt Test-Konfigurationen für verschiedene Szenarien

- ⚙️ **`run_single_test(self, test_config: TestConfiguration) -> TestResult`**
  - Führt einen einzelnen Test durch

- ⚙️ **`run_test_suite(self, document_paths: List[str], output_file: str = None) -> Dict[str, Any]`**
  - Führt eine komplette Test-Suite durch

- ⚙️ **`run_quick_comparison(self, document_path: str, models: List[str] = None) -> Dict[str, Any]`**
  - Schneller Vergleich verschiedener Modelle für ein Dokument
