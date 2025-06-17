# vectorstore.py

## 📋 Übersicht

- **Datei:** `vectorstore.py`
- **Zeilen:** 457
- **Analysiert:** 2025-06-17T14:46:23

## 📦 Imports

### Standard Library
- `import os`

### Third Party
- `from PIL import Image`
- `from langchain_chroma import Chroma`
- `from langchain_core.documents import Document`
- `from langchain_ollama.embeddings import OllamaEmbeddings`
- `from langchain_text_splitters import RecursiveCharacterTextSplitter`
- `from openpyxl import load_workbook`
- `from pptx import Presentation`
- `from pypdf import PdfReader`
- `import docx`
- `import fitz`
- `import io`
- `import logging`
- `import pandas as pd`
- `import pytesseract`
- `import shutil`
- `import subprocess`
- `import subprocess`
- `import tempfile`
- `import tempfile`
- `import xml.etree.ElementTree as ET`
- `import zipfile`

### Local/App
- `from app.config import config`

## 🔧 Konstanten & Variablen

- 📝 **`logger`** = `logging.getLogger(__name__)`

## ⚙️ Funktionen

### Öffentliche Funktionen

#### `get_embeddings(model_name)`

**Beschreibung:** Erstellt ein Embedding-Objekt für das angegebene Modell

#### `is_page_scannable(page)`

**Beschreibung:** Prüft, ob eine Seite hauptsächlich aus Bildern besteht (OCR erforderlich)

#### `extract_text_from_pdf_with_ocr(file_path)`

**Beschreibung:** Extrahiere Text aus PDF, mit OCR-Fallback für gescannte Seiten

#### `extract_text_from_docx(file_path)`

**Beschreibung:** Extrahiere Text aus Word-Dokument (.docx)

#### `extract_text_from_doc(file_path)`

**Beschreibung:** Extrahiere Text aus altem Word-Format (.doc) über LibreOffice

#### `extract_text_from_excel(file_path)`

**Beschreibung:** Extrahiere Text aus Excel-Datei (.xlsx, .xls)

#### `extract_text_from_powerpoint(file_path)`

**Beschreibung:** Extrahiere Text aus PowerPoint-Präsentation (.pptx)

#### `extract_text_from_ppt(file_path)`

**Beschreibung:** Extrahiere Text aus altem PowerPoint-Format (.ppt) über LibreOffice

#### `chunk_text(text, chunk_size = 1000, chunk_overlap = 200)`

**Beschreibung:** Teilt Text in kleinere Chunks für bessere Embedding-Performance

#### `load_documents_from_file(file_path, chunk_size = 1000, chunk_overlap = 200)`

**Beschreibung:** Lädt Text aus verschiedenen Dateiformaten

#### `load_documents(documents_path, chunk_size = 1000, chunk_overlap = 200)`

**Beschreibung:** Lädt alle Dokumente aus einem Pfad (Datei oder Verzeichnis)

#### `build_vectorstore(documents, model_name, persist_directory = None)`

**Beschreibung:** Erstelle einen Vektorspeicher aus LangChain-Dokumenten

#### `load_vectorstore(persist_directory, model_name)`

**Beschreibung:** Lade einen gespeicherten Vektorspeicher

#### `delete_vectorstore(persist_directory)`

**Beschreibung:** Lösche einen Vektorspeicher

#### `list_vectorstores(base_directory)`

**Beschreibung:** Liste alle verfügbaren Vektorspeicher in einem Verzeichnis auf

#### `get_supported_file_types()`

**Beschreibung:** Gibt eine Liste der unterstützten Dateiformate zurück

#### `get_vectorstore(documents, model_name = 'nomic-embed-text', persist_directory = None, chunk_size = 1000, chunk_overlap = 200)`

**Beschreibung:** Erstellt oder lädt einen Vektorspeicher
