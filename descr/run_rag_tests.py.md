# run_rag_tests.py

## 📋 Übersicht

- **Datei:** `run_rag_tests.py`
- **Zeilen:** 564
- **Analysiert:** 2025-06-23T13:19:59

### Beschreibung

```
RAG System Test Runner
======================

Reparierte und optimierte Version des Test-Runners für das RAG-System.
Speziell für große deutsche Dokumente optimiert.

Verwendung:
    python run_rag_tests.py --documents path/to/docs --output results.json
    python run_rag_tests.py --quick-test path/to/single_doc.pdf
    python run_rag_tests.py --benchmark --models llama3.2,phi4-mini-reasoning:3.8b
```

## 📦 Imports

### Standard Library
- `from datetime import datetime`
- `from pathlib import Path`
- `from typing import List, Dict, Any, Optional`
- `import argparse`
- `import json`
- `import sys`

### Third Party
- `import logging`
- `import time`
- `import traceback`

### Local/App
- `from app.config import config`
- `from app.connection_utils import check_ollama_connection_with_retry`
- `from app.test_rag import RAGTestFramework, TestConfiguration`
- `from app.test_utils import PerformanceMonitor, TestResultVisualizer, BatchTestRunner, ReportGenerator`
- `from app.test_utils import quick_document_analysis, estimate_test_duration`

## 🔧 Konstanten & Variablen

- 📝 **`RAGTestFramework`** = `None`
- 📝 **`TestConfiguration`** = `None`
- 📝 **`quick_document_analysis`** = `None`
- 📝 **`estimate_test_duration`** = `None`
- 📝 **`PerformanceMonitor`** = `None`
- 📝 **`TestResultVisualizer`** = `None`
- 📝 **`BatchTestRunner`** = `None`
- 📝 **`ReportGenerator`** = `None`
- 📝 **`logger`** = `logging.getLogger(__name__)`
- 🔒 **`DEFAULT_MODELS`** = `['llama3.2', 'phi4-mini-reasoning:3.8b', 'mistral'`
- 🔒 **`DEFAULT_CHUNK_SIZES`** = `[500, 750, 1000, 1500, 2000]`
- 🔒 **`DEFAULT_OVERLAPS`** = `[50, 100, 150, 200]`
- 🔒 **`QUICK_TEST_CONFIG`** = `{'models': ['llama3.2'], 'chunk_sizes': [750, 1000`
- 🔒 **`BENCHMARK_CONFIG`** = `{'models': DEFAULT_MODELS, 'chunk_sizes': [500, 10`

## ⚙️ Funktionen

### Öffentliche Funktionen

#### `setup_test_environment() -> bool`

**Beschreibung:** Bereitet die Test-Umgebung vor und prüft alle Abhängigkeiten.

#### `run_document_analysis(document_paths: List[str]) -> Dict[str, Any]`

**Beschreibung:** Führt Vor-Analyse der Dokumente durch.

#### `run_quick_test(document_path: str, output_file: Optional[str] = None) -> Dict[str, Any]`

**Beschreibung:** Führt einen schnellen Test für ein einzelnes Dokument durch.

#### `run_full_benchmark(document_paths: List[str], output_file: Optional[str] = None) -> Dict[str, Any]`

**Beschreibung:** Führt vollständigen Benchmark durch.

#### `main()`

**Beschreibung:** Hauptfunktion für CLI-Nutzung
