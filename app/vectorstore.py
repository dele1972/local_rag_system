# app/vectorstore.py
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import config
import os
import shutil
from pypdf import PdfReader
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import zipfile
import xml.etree.ElementTree as ET
from openpyxl import load_workbook
import pandas as pd
from pptx import Presentation
import docx
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embeddings(model_name):
    """Erstellt ein Embedding-Objekt für das angegebene Modell"""
    return OllamaEmbeddings(
        base_url=config.get_ollama_base_url(), 
        model=model_name
    )

def is_page_scannable(page):
    """Prüft, ob eine Seite hauptsächlich aus Bildern besteht (OCR erforderlich)"""
    text = page.extract_text()
    return not text or len(text.strip()) < 50  # Schwellenwert anpassbar

def extract_text_from_pdf_with_ocr(file_path):
    """Extrahiere Text aus PDF, mit OCR-Fallback für gescannte Seiten"""
    reader = PdfReader(file_path)
    raw_text = ""

    for i, page in enumerate(reader.pages):
        try:
            if is_page_scannable(page):
                # OCR für gescannte Seiten
                try:
                    with fitz.open(file_path) as doc:
                        pix = doc[i].get_pixmap()
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        ocr_text = pytesseract.image_to_string(img, lang="deu")
                        if not ocr_text.strip():
                            logger.warning(f"OCR hat keinen Text auf Seite {i+1} erkannt")
                        raw_text += ocr_text + "\n"
                except pytesseract.TesseractError as e:
                    logger.error(f"OCR-Fehler auf Seite {i+1}: {e}")
                except Exception as e:
                    logger.error(f"Allgemeiner OCR-Fehler auf Seite {i+1}: {e}")
            else:
                # Normaler Text extrahierbar
                text = page.extract_text()
                raw_text += text + "\n"
        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten der Seite {i+1}: {e}")
    
    return raw_text

def extract_text_from_docx(file_path):
    """Extrahiere Text aus Word-Dokument (.docx)"""
    try:
        doc = docx.Document(file_path)
        text = ""
        
        # Haupttext
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # Text aus Tabellen
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + "\t"
                text += "\n"
        
        # Text aus Kopf- und Fußzeilen
        for section in doc.sections:
            if section.header:
                for paragraph in section.header.paragraphs:
                    text += paragraph.text + "\n"
            if section.footer:
                for paragraph in section.footer.paragraphs:
                    text += paragraph.text + "\n"
                    
        return text
    except Exception as e:
        logger.error(f"Fehler beim Extrahieren aus Word-Dokument {file_path}: {e}")
        return ""

def extract_text_from_doc(file_path):
    """Extrahiere Text aus altem Word-Format (.doc) über LibreOffice"""
    try:
        # Fallback für .doc Dateien - erfordert LibreOffice
        import subprocess
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Konvertiere .doc zu .txt mit LibreOffice
            output_file = os.path.join(temp_dir, "converted.txt")
            subprocess.run([
                "libreoffice", "--headless", "--convert-to", "txt",
                "--outdir", temp_dir, file_path
            ], check=True, capture_output=True)
            
            # Lese konvertierte Datei
            txt_file = os.path.join(temp_dir, os.path.splitext(os.path.basename(file_path))[0] + ".txt")
            if os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    return f.read()
            return ""
    except Exception as e:
        logger.error(f"Fehler beim Extrahieren aus .doc Datei {file_path}: {e}")
        logger.info("Hinweis: .doc Dateien benötigen LibreOffice für die Konvertierung")
        return ""

def extract_text_from_excel(file_path):
    """Extrahiere Text aus Excel-Datei (.xlsx, .xls)"""
    try:
        text = ""
        
        # Verwende pandas für bessere Kompatibilität
        if file_path.endswith('.xlsx'):
            # Lade alle Sheets
            xl_file = pd.ExcelFile(file_path)
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                text += f"\n=== Sheet: {sheet_name} ===\n"
                
                # Konvertiere DataFrame zu Text
                for index, row in df.iterrows():
                    row_text = []
                    for col in df.columns:
                        cell_value = str(row[col]) if pd.notna(row[col]) else ""
                        if cell_value:
                            row_text.append(f"{col}: {cell_value}")
                    if row_text:
                        text += " | ".join(row_text) + "\n"
        
        elif file_path.endswith('.xls'):
            # Für .xls Dateien
            df = pd.read_excel(file_path, engine='xlrd')
            for index, row in df.iterrows():
                row_text = []
                for col in df.columns:
                    cell_value = str(row[col]) if pd.notna(row[col]) else ""
                    if cell_value:
                        row_text.append(f"{col}: {cell_value}")
                if row_text:
                    text += " | ".join(row_text) + "\n"
        
        return text
    except Exception as e:
        logger.error(f"Fehler beim Extrahieren aus Excel-Datei {file_path}: {e}")
        return ""

def extract_text_from_powerpoint(file_path):
    """Extrahiere Text aus PowerPoint-Präsentation (.pptx)"""
    try:
        prs = Presentation(file_path)
        text = ""
        
        for slide_num, slide in enumerate(prs.slides, 1):
            text += f"\n=== Folie {slide_num} ===\n"
            
            # Text aus Shapes extrahieren
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text += shape.text + "\n"
                
                # Text aus Tabellen in Shapes
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = []
                        for cell in row.cells:
                            if cell.text.strip():
                                row_text.append(cell.text.strip())
                        if row_text:
                            text += " | ".join(row_text) + "\n"
            
            # Notizen extrahieren
            if slide.has_notes_slide:
                notes_slide = slide.notes_slide
                if notes_slide.notes_text_frame and notes_slide.notes_text_frame.text:
                    text += f"Notizen: {notes_slide.notes_text_frame.text}\n"
        
        return text
    except Exception as e:
        logger.error(f"Fehler beim Extrahieren aus PowerPoint-Datei {file_path}: {e}")
        return ""

def extract_text_from_ppt(file_path):
    """Extrahiere Text aus altem PowerPoint-Format (.ppt) über LibreOffice"""
    try:
        import subprocess
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Konvertiere .ppt zu .txt mit LibreOffice
            subprocess.run([
                "libreoffice", "--headless", "--convert-to", "txt",
                "--outdir", temp_dir, file_path
            ], check=True, capture_output=True)
            
            # Lese konvertierte Datei
            txt_file = os.path.join(temp_dir, os.path.splitext(os.path.basename(file_path))[0] + ".txt")
            if os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    return f.read()
            return ""
    except Exception as e:
        logger.error(f"Fehler beim Extrahieren aus .ppt Datei {file_path}: {e}")
        logger.info("Hinweis: .ppt Dateien benötigen LibreOffice für die Konvertierung")
        return ""

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Teilt Text in kleinere Chunks für bessere Embedding-Performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def load_documents_from_file(file_path, chunk_size=1000, chunk_overlap=200):
    """Lädt Text aus verschiedenen Dateiformaten"""
    content = ""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    logger.debug(f"[DEBUG] Starte Verarbeitung von Datei: {file_path}")
    try:
        if file_extension in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        
        elif file_extension == ".pdf":
            content = extract_text_from_pdf_with_ocr(file_path)
        
        elif file_extension == ".docx":
            content = extract_text_from_docx(file_path)
        
        elif file_extension == ".doc":
            content = extract_text_from_doc(file_path)
        
        elif file_extension in [".xlsx", ".xls"]:
            content = extract_text_from_excel(file_path)
        
        elif file_extension == ".pptx":
            content = extract_text_from_powerpoint(file_path)
        
        elif file_extension == ".ppt":
            content = extract_text_from_ppt(file_path)
        
        else:
            raise ValueError(f"Nicht unterstützter Dateityp: {file_extension}")
        
        if not content or len(content.strip()) == 0:
            logger.warning(f"Keine Inhalte aus Datei {file_path} extrahiert")
            return []
        
        # Text in Chunks aufteilen
        if len(content) > chunk_size:
            chunks = chunk_text(content, chunk_size, chunk_overlap)
            documents = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Nur nicht-leere Chunks
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "file_type": file_extension,
                            "chunk": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
            return documents
        else:
            return [Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "file_type": file_extension
                }
            )]
    
    except Exception as e:
        logger.error(f"❌ Fehler beim Laden der Datei {file_path}: {e}")
        return []

def load_documents(documents_path, chunk_size=1000, chunk_overlap=200):
    """
    Lädt alle Dokumente aus einem Pfad (Datei oder Verzeichnis)
    
    Args:
        documents_path: Pfad zu einer Datei oder einem Verzeichnis
        chunk_size: Größe der Text-Chunks  
        chunk_overlap: Überlappung zwischen Chunks
    
    Returns:
        Liste von LangChain-Dokumenten
    """
    documents = []
    supported_extensions = config.get_supported_file_types()
    
    if os.path.isfile(documents_path):
        # Einzelne Datei
        logger.info(f"Lade Datei: {documents_path}")
        documents.extend(load_documents_from_file(documents_path, chunk_size, chunk_overlap))
        
    elif os.path.isdir(documents_path):
        # Verzeichnis durchsuchen
        logger.info(f"Durchsuche Verzeichnis: {documents_path}")
        file_count = 0
        
        for root, dirs, files in os.walk(documents_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file)[1].lower()
                
                if file_extension in supported_extensions:
                    logger.info(f"Lade Datei: {file_path}")
                    file_documents = load_documents_from_file(file_path, chunk_size, chunk_overlap)
                    documents.extend(file_documents)
                    file_count += 1
                else:
                    logger.debug(f"Überspringe nicht unterstützte Datei: {file_path}")
        
        logger.info(f"Erfolgreich {file_count} Dateien geladen, insgesamt {len(documents)} Dokument-Chunks")
    else:
        raise ValueError(f"Pfad nicht gefunden: {documents_path}")
    
    if not documents:
        logger.warning("Keine Dokumente gefunden oder alle Dateien waren leer")
    
    return documents

def build_vectorstore(documents, model_name, persist_directory=None):
    """Erstelle einen Vektorspeicher aus LangChain-Dokumenten"""
    if not documents:
        raise ValueError("Keine Dokumente zum Erstellen des Vektorspeichers vorhanden")
    
    embeddings = get_embeddings(model_name)
    
    if persist_directory:
        return Chroma.from_documents(
            documents=documents, 
            embedding=embeddings,
            persist_directory=persist_directory
        )
    else:
        return Chroma.from_documents(
            documents=documents, 
            embedding=embeddings
        )

def load_vectorstore(persist_directory, model_name):
    """Lade einen gespeicherten Vektorspeicher"""
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Vector store not found at {persist_directory}")
    
    embeddings = get_embeddings(model_name)
    
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def delete_vectorstore(persist_directory):
    """Lösche einen Vektorspeicher"""
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        return True
    return False

def list_vectorstores(base_directory):
    """Liste alle verfügbaren Vektorspeicher in einem Verzeichnis auf"""
    vectorstores = {}
    
    if not os.path.exists(base_directory):
        return vectorstores
    
    for item in os.listdir(base_directory):
        item_path = os.path.join(base_directory, item)
        if os.path.isdir(item_path):
            chroma_files = ["chroma.sqlite3", "chroma_metadata.parquet"]
            has_chroma_files = any(os.path.exists(os.path.join(item_path, cf)) for cf in chroma_files)
            if has_chroma_files:
                vectorstores[item] = item_path
    
    return vectorstores

def get_vectorstore(documents, model_name="nomic-embed-text", persist_directory=None, chunk_size=1000, chunk_overlap=200):
    """
    Erstellt oder lädt einen Vektorspeicher
    
    Args:
        documents: Liste von LangChain-Dokumenten oder Pfad zu Dokumenten
        model_name: Name des Embedding-Modells
        persist_directory: Verzeichnis zum Speichern des Vektorspeichers
        chunk_size: Größe der Text-Chunks
        chunk_overlap: Überlappung zwischen Chunks
    
    Returns:
        Chroma-Vektorspeicher
    """
    
    # Falls documents ein String ist (Pfad), lade Dokumente aus dem Pfad
    if isinstance(documents, str):
        documents_path = documents
        documents = []
        
        # Lade alle unterstützten Dateien aus dem Pfad
        supported_extensions = config.get_supported_file_types()
        
        if os.path.isfile(documents_path):
            # Einzelne Datei
            documents.extend(load_documents_from_file(documents_path, chunk_size, chunk_overlap))
        elif os.path.isdir(documents_path):
            # Verzeichnis durchsuchen
            for root, dirs, files in os.walk(documents_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_extension = os.path.splitext(file)[1].lower()
                    
                    if file_extension in supported_extensions:
                        logger.info(f"Lade Datei: {file_path}")
                        file_documents = load_documents_from_file(file_path, chunk_size, chunk_overlap)
                        documents.extend(file_documents)
        else:
            raise ValueError(f"Pfad nicht gefunden: {documents_path}")
    
    if not documents:
        raise ValueError("Keine Dokumente zum Erstellen des Vektorspeichers gefunden")
    
    logger.info(f"Erstelle Vektorspeicher mit {len(documents)} Dokumenten")
    
    # Vektorspeicher erstellen
    return build_vectorstore(documents, model_name, persist_directory)