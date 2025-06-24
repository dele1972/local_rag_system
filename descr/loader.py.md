# loader.py

## 📋 Übersicht

- **Datei:** `loader.py`
- **Zeilen:** 257
- **Analysiert:** 2025-06-24T10:56:33

## 📦 Imports

### Standard Library
- `from pathlib import Path`
- `import os`

### Third Party
- `from langchain.text_splitter import RecursiveCharacterTextSplitter`
- `from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader`

### Local/App
- `from app.config import config`
- `from app.vectorstore import load_documents_from_file`

## ⚙️ Funktionen

### Öffentliche Funktionen

#### `get_chunk_size_for_file(file_size_mb)`

**Beschreibung:** Bestimmt optimale Chunk-Größe basierend auf Dateigröße

#### `get_file_size_mb(file_path)`

**Beschreibung:** Gibt die Dateigröße in MB zurück

#### `create_adaptive_text_splitter(file_path)`

**Beschreibung:** Erstellt einen TextSplitter mit adaptiver Chunk-Größe

#### `load_documents_from_path(path)`

**Beschreibung:** Load documents from a directory path with support for multiple file formats

#### `load_single_file(file_path)`

**Beschreibung:** Load a single file with format detection and adaptive chunking

#### `get_file_stats(path)`

**Beschreibung:** Get detailed statistics about files in a directory
