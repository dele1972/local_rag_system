# app/loader.py
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import os
from app.vectorstore import load_documents_from_file  # Unsere OCR-fähige Funktion importieren

def load_documents_from_path(path):
    """Load documents from a directory path with OCR support for PDFs"""
    docs = []
    loaded_files = []
    
    for file in Path(path).rglob("*"):
        if file.is_file():
            try:
                if file.suffix.lower() == ".pdf":
                    # Verwende unsere OCR-fähige Funktion statt PyPDFLoader
                    pdf_docs = load_documents_from_file(str(file))
                    docs.extend(pdf_docs)
                    loaded_files.append(str(file))
                    print(f"✅ PDF mit OCR-Support geladen: {file.name}")
                elif file.suffix.lower() == ".txt":
                    docs.extend(TextLoader(str(file)).load())
                    loaded_files.append(str(file))
                    print(f"✅ TXT-Datei geladen: {file.name}")
                elif file.suffix.lower() == ".md":
                    docs.extend(UnstructuredMarkdownLoader(str(file)).load())
                    loaded_files.append(str(file))
                    print(f"✅ Markdown-Datei geladen: {file.name}")
            except Exception as e:
                print(f"❌ Fehler beim Laden von {file}: {str(e)}")
    
    # Da unsere load_documents_from_file bereits chunking macht,
    # prüfen wir ob zusätzliches splitting nötig ist
    final_docs = []
    for doc in docs:
        # Wenn das Dokument bereits gechunkt ist (hat chunk metadata), behalten wir es
        if hasattr(doc, 'metadata') and 'chunk' in doc.metadata:
            final_docs.append(doc)
        else:
            # Ansonsten splitten wir es noch
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents([doc])
            final_docs.extend(split_docs)
    
    print(f"📊 Insgesamt {len(final_docs)} Dokument-Chunks aus {len(loaded_files)} Dateien geladen")
    
    return final_docs, loaded_files