# app/loader.py
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import os
from app.vectorstore import load_documents_from_file, get_supported_file_types

def load_documents_from_path(path):
    """Load documents from a directory path with support for multiple file formats"""
    docs = []
    loaded_files = []
    skipped_files = []
    supported_extensions = get_supported_file_types()
    
    for file in Path(path).rglob("*"):
        if file.is_file():
            file_extension = file.suffix.lower()
            
            try:
                if file_extension in supported_extensions:
                    # Verwende unsere universelle Funktion für alle unterstützten Formate
                    file_docs = load_documents_from_file(str(file))
                    if file_docs:  # Nur wenn Dokumente erfolgreich geladen wurden
                        docs.extend(file_docs)
                        loaded_files.append(str(file))
                        print(f"✅ {file_extension.upper()}-Datei geladen: {file.name} ({len(file_docs)} Chunks)")
                    else:
                        print(f"⚠️  Keine Inhalte aus {file.name} extrahiert")
                        skipped_files.append(str(file))
                
                elif file_extension == ".txt":
                    # Für .txt Dateien können wir auch den LangChain TextLoader verwenden
                    try:
                        langchain_docs = TextLoader(str(file)).load()
                        docs.extend(langchain_docs)
                        loaded_files.append(str(file))
                        print(f"✅ TXT-Datei geladen: {file.name}")
                    except:
                        # Fallback auf unsere eigene Funktion
                        file_docs = load_documents_from_file(str(file))
                        if file_docs:
                            docs.extend(file_docs)
                            loaded_files.append(str(file))
                            print(f"✅ TXT-Datei geladen (Fallback): {file.name}")
                
                elif file_extension == ".md":
                    # Für .md Dateien können wir auch den LangChain Loader verwenden
                    try:
                        langchain_docs = UnstructuredMarkdownLoader(str(file)).load()
                        docs.extend(langchain_docs)
                        loaded_files.append(str(file))
                        print(f"✅ Markdown-Datei geladen: {file.name}")
                    except:
                        # Fallback auf unsere eigene Funktion
                        file_docs = load_documents_from_file(str(file))
                        if file_docs:
                            docs.extend(file_docs)
                            loaded_files.append(str(file))
                            print(f"✅ Markdown-Datei geladen (Fallback): {file.name}")
                
                else:
                    # Nicht unterstützte Dateierweiterung
                    skipped_files.append(str(file))
                    print(f"⏭️  Übersprungen (nicht unterstützt): {file.name} ({file_extension})")
                    
            except Exception as e:
                print(f"❌ Fehler beim Laden von {file}: {str(e)}")
                skipped_files.append(str(file))
    
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
    
    # Zusammenfassung ausgeben
    print(f"\n📊 Lade-Zusammenfassung:")
    print(f"   ✅ Erfolgreich geladen: {len(loaded_files)} Dateien")
    print(f"   📄 Dokument-Chunks: {len(final_docs)}")
    if skipped_files:
        print(f"   ⏭️  Übersprungen: {len(skipped_files)} Dateien")
    
    # Unterstützte Dateiformate anzeigen
    print(f"\n🔧 Unterstützte Formate: {', '.join(supported_extensions)}")
    
    return final_docs, loaded_files

def load_single_file(file_path):
    """Load a single file with format detection"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Pfad ist keine Datei: {file_path}")
        
        file_extension = file_path.suffix.lower()
        supported_extensions = get_supported_file_types()
        
        if file_extension in supported_extensions:
            docs = load_documents_from_file(str(file_path))
            if docs:
                print(f"✅ Datei erfolgreich geladen: {file_path.name} ({len(docs)} Chunks)")
                return docs
            else:
                print(f"⚠️  Keine Inhalte aus Datei extrahiert: {file_path.name}")
                return []
        else:
            raise ValueError(f"Dateityp nicht unterstützt: {file_extension}")
            
    except Exception as e:
        print(f"❌ Fehler beim Laden der Datei {file_path}: {str(e)}")
        return []

def get_file_stats(path):
    """Get statistics about files in a directory"""
    stats = {}
    supported_extensions = get_supported_file_types()
    
    for file in Path(path).rglob("*"):
        if file.is_file():
            ext = file.suffix.lower()
            if ext in stats:
                stats[ext] += 1
            else:
                stats[ext] = 1
    
    supported_count = sum(count for ext, count in stats.items() if ext in supported_extensions)
    total_count = sum(stats.values())
    
    print(f"\n📈 Datei-Statistik für '{path}':")
    print(f"   📁 Gesamt: {total_count} Dateien")
    print(f"   ✅ Unterstützt: {supported_count} Dateien")
    print(f"   ❌ Nicht unterstützt: {total_count - supported_count} Dateien")
    
    if stats:
        print(f"\n📋 Dateitypen gefunden:")
        for ext, count in sorted(stats.items()):
            status = "✅" if ext in supported_extensions else "❌"
            print(f"   {status} {ext}: {count} Datei(en)")
    
    return stats