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
                            print(f"Warnung: OCR hat keinen Text auf Seite {i+1} erkannt")
                        raw_text += ocr_text + "\n"
                except pytesseract.TesseractError as e:
                    print(f"OCR-Fehler auf Seite {i+1}: {e}")
                except Exception as e:
                    print(f"Allgemeiner OCR-Fehler auf Seite {i+1}: {e}")
            else:
                # Normaler Text extrahierbar
                text = page.extract_text()
                raw_text += text + "\n"
        except Exception as e:
            print(f"Fehler beim Verarbeiten der Seite {i+1}: {e}")
    
    return raw_text

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
    """Lädt Text aus Datei (Text oder PDF), inkl. OCR bei gescannten PDFs"""
    # Korrektur: Tupel für endswith() verwenden
    if file_path.endswith((".txt", ".md")):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    elif file_path.endswith(".pdf"):
        content = extract_text_from_pdf_with_ocr(file_path)
    else:
        raise ValueError(f"Nicht unterstützter Dateityp: {file_path}")
    
    # Text in Chunks aufteilen, besonders wichtig für OCR-Texte
    if len(content) > chunk_size:
        chunks = chunk_text(content, chunk_size, chunk_overlap)
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": file_path,
                    "chunk": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        return documents
    else:
        return [Document(
            page_content=content,
            metadata={"source": file_path}
        )]

def build_vectorstore(documents, model_name, persist_directory=None):
    """Erstelle einen Vektorspeicher aus LangChain-Dokumenten"""
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