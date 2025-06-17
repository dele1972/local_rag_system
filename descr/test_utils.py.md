# test_utils.py

## 📋 Übersicht

- **Datei:** `test_utils.py`
- **Zeilen:** 619
- **Analysiert:** 2025-06-17T14:46:23

### Beschreibung

```
RAG Test Framework - Utilities und Helper-Funktionen
====================================================

Zusätzliche Hilfsfunktionen für erweiterte Tests und Analysen.
Speziell für deutsche Dokumente und Performance-Optimierung.

Autor: RAG System Optimizer
Datum: 2025-06-16
```

## 📦 Imports

### Standard Library
- `from pathlib import Path`
- `from typing import List, Dict, Any, Tuple, Optional`
- `import json`
- `import os`
- `import re`

### Third Party
- `from collections import defaultdict`
- `from concurrent.futures import ThreadPoolExecutor, as_completed`
- `from dataclasses import dataclass`
- `import chardet`
- `import logging`
- `import matplotlib.pyplot as plt`
- `import numpy as np`
- `import pandas as pd`
- `import psutil`
- `import seaborn as sns`
- `import time`

### Local/App
- `from app.vectorstore import load_documents_from_file`

## 🔧 Konstanten & Variablen

- 📝 **`logger`** = `logging.getLogger(__name__)`

## ⚙️ Funktionen

### Öffentliche Funktionen

#### `quick_document_analysis(file_path: str) -> Dict[str, Any]`

**Beschreibung:** Schnelle Dokument-Analyse für Vor-Test-Bewertung

#### `estimate_test_duration(document_paths: List[str], models: List[str]) -> Dict[str, Any]`

**Beschreibung:** Schätzt Testdauer basierend auf Dokumenten und Modellen

## 🏗️ Klassen

### `DocumentMetrics`

**Beschreibung:** Metriken für Dokument-Analyse

### `DocumentAnalyzer`

**Beschreibung:** Analysiert Dokumente vor dem RAG-Test

#### Methoden

- ⚙️ **`analyze_document(self, file_path: str) -> DocumentMetrics`**
  - Führt umfassende Dokument-Analyse durch

### `PerformanceMonitor`

**Beschreibung:** Überwacht System-Performance während Tests

#### Methoden

- ⚙️ **`start_monitoring(self)`**
  - Startet Performance-Monitoring

- ⚙️ **`stop_monitoring(self) -> Dict[str, Any]`**
  - Stoppt Monitoring und gibt Zusammenfassung zurück

- ⚙️ **`collect_sample(self)`**
  - Sammelt eine Performance-Probe

**Private Methoden:**
- `__init__()` - Führt aus: self.monitoring = F...

### `TestResultVisualizer`

**Beschreibung:** Erstellt Visualisierungen für Test-Ergebnisse

#### Methoden

- ⚙️ **`create_performance_dashboard(self, results_file: str) -> str`**
  - Erstellt umfassendes Performance-Dashboard

- ⚙️ **`create_model_comparison_chart(self, results_file: str) -> str`**
  - Erstellt detaillierten Modell-Vergleich

**Private Methoden:**
- `__init__()` - Führt aus: self.output_dir = P...

### `BatchTestRunner`

**Beschreibung:** Führt Tests parallel aus für bessere Performance

#### Methoden

- ⚙️ **`run_parallel_tests(self, test_framework, configurations: List) -> List`**
  - Führt Tests parallel aus

**Private Methoden:**
- `__init__()` - Führt aus: self.max_workers = ...

### `ReportGenerator`

**Beschreibung:** Generiert detaillierte Test-Berichte

#### Methoden

- ⚙️ **`generate_html_report(self, results_file: str) -> str`**
  - Generiert HTML-Bericht

**Private Methoden:**
- `__init__()` - Führt aus: self.output_dir = P...
- `_create_html_template()` - Erstellt HTML-Template für Ber...
