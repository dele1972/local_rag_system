# app/loader.py
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import os
from app.vectorstore import load_documents_from_file, get_supported_file_types

def get_chunk_size_for_file(file_size_mb):
    """Bestimmt optimale Chunk-Größe basierend auf Dateigröße"""
    if file_size_mb > 5:
        return 2000, 400  # chunk_size, chunk_overlap
    elif file_size_mb > 1:
        return 1500, 300
    else:
        return 1000, 200

def get_file_size_mb(file_path):
    """Gibt die Dateigröße in MB zurück"""
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except:
        return 0

def create_adaptive_text_splitter(file_path):
    """Erstellt einen TextSplitter mit adaptiver Chunk-Größe"""
    file_size_mb = get_file_size_mb(file_path)
    chunk_size, chunk_overlap = get_chunk_size_for_file(file_size_mb)
    
    print(f"📏 Datei {Path(file_path).name}: {file_size_mb:.2f}MB -> Chunk-Size: {chunk_size}")
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

def load_documents_from_path(path):
    """Load documents from a directory path with support for multiple file formats"""
    docs = []
    loaded_files = []
    skipped_files = []
    supported_extensions = get_supported_file_types()
    file_stats = {}  # Für Statistiken
    
    for file in Path(path).rglob("*"):
        if file.is_file():
            file_extension = file.suffix.lower()
            file_size_mb = get_file_size_mb(str(file))
            
            try:
                if file_extension in supported_extensions:
                    # Verwende unsere universelle Funktion für alle unterstützten Formate
                    file_docs = load_documents_from_file(str(file))
                    if file_docs:  # Nur wenn Dokumente erfolgreich geladen wurden
                        docs.extend(file_docs)
                        loaded_files.append(str(file))
                        file_stats[str(file)] = {
                            'size_mb': file_size_mb,
                            'chunks': len(file_docs),
                            'type': file_extension
                        }
                        print(f"✅ {file_extension.upper()}-Datei geladen: {file.name} ({file_size_mb:.2f}MB, {len(file_docs)} Chunks)")
                    else:
                        print(f"⚠️  Keine Inhalte aus {file.name} extrahiert")
                        skipped_files.append(str(file))
                
                elif file_extension == ".txt":
                    # Für .txt Dateien können wir auch den LangChain TextLoader verwenden
                    try:
                        langchain_docs = TextLoader(str(file)).load()
                        docs.extend(langchain_docs)
                        loaded_files.append(str(file))
                        file_stats[str(file)] = {
                            'size_mb': file_size_mb,
                            'chunks': len(langchain_docs),
                            'type': file_extension
                        }
                        print(f"✅ TXT-Datei geladen: {file.name} ({file_size_mb:.2f}MB)")
                    except:
                        # Fallback auf unsere eigene Funktion
                        file_docs = load_documents_from_file(str(file))
                        if file_docs:
                            docs.extend(file_docs)
                            loaded_files.append(str(file))
                            file_stats[str(file)] = {
                                'size_mb': file_size_mb,
                                'chunks': len(file_docs),
                                'type': file_extension
                            }
                            print(f"✅ TXT-Datei geladen (Fallback): {file.name} ({file_size_mb:.2f}MB)")
                
                elif file_extension == ".md":
                    # Für .md Dateien können wir auch den LangChain Loader verwenden
                    try:
                        langchain_docs = UnstructuredMarkdownLoader(str(file)).load()
                        docs.extend(langchain_docs)
                        loaded_files.append(str(file))
                        file_stats[str(file)] = {
                            'size_mb': file_size_mb,
                            'chunks': len(langchain_docs),
                            'type': file_extension
                        }
                        print(f"✅ Markdown-Datei geladen: {file.name} ({file_size_mb:.2f}MB)")
                    except:
                        # Fallback auf unsere eigene Funktion
                        file_docs = load_documents_from_file(str(file))
                        if file_docs:
                            docs.extend(file_docs)
                            loaded_files.append(str(file))
                            file_stats[str(file)] = {
                                'size_mb': file_size_mb,
                                'chunks': len(file_docs),
                                'type': file_extension
                            }
                            print(f"✅ Markdown-Datei geladen (Fallback): {file.name} ({file_size_mb:.2f}MB)")
                
                else:
                    # Nicht unterstützte Dateierweiterung
                    skipped_files.append(str(file))
                    print(f"⏭️  Übersprungen (nicht unterstützt): {file.name} ({file_extension}, {file_size_mb:.2f}MB)")
                    
            except Exception as e:
                print(f"❌ Fehler beim Laden von {file}: {str(e)}")
                skipped_files.append(str(file))
    
    # Adaptives Chunking für Dokumente ohne chunk metadata
    final_docs = []
    rechunked_count = 0
    
    for doc in docs:
        # Wenn das Dokument bereits gechunkt ist (hat chunk metadata), behalten wir es
        if hasattr(doc, 'metadata') and 'chunk' in doc.metadata:
            final_docs.append(doc)
        else:
            # Adaptives Splitting basierend auf der ursprünglichen Datei
            source_file = doc.metadata.get('source', '')
            if source_file and os.path.exists(source_file):
                text_splitter = create_adaptive_text_splitter(source_file)
                rechunked_count += 1
            else:
                # Fallback auf Standard-Chunking
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
            
            split_docs = text_splitter.split_documents([doc])
            final_docs.extend(split_docs)
    
    # Detaillierte Zusammenfassung ausgeben
    print(f"\n📊 Lade-Zusammenfassung:")
    print(f"   ✅ Erfolgreich geladen: {len(loaded_files)} Dateien")
    print(f"   📄 Dokument-Chunks: {len(final_docs)}")
    print(f"   🔄 Adaptiv gechunkt: {rechunked_count} Dokumente")
    if skipped_files:
        print(f"   ⏭️  Übersprungen: {len(skipped_files)} Dateien")
    
    # Dateigröße-Statistiken
    if file_stats:
        total_size = sum(stats['size_mb'] for stats in file_stats.values())
        large_files = [f for f, stats in file_stats.items() if stats['size_mb'] > 5]
        print(f"\n📈 Größen-Statistik:")
        print(f"   📦 Gesamtgröße: {total_size:.2f}MB")
        print(f"   📄 Große Dateien (>5MB): {len(large_files)}")
        
        if large_files:
            print(f"   🎯 Große Dateien Details:")
            for file_path in large_files[:5]:  # Nur die ersten 5 anzeigen
                stats = file_stats[file_path]
                filename = Path(file_path).name
                print(f"      • {filename}: {stats['size_mb']:.2f}MB, {stats['chunks']} Chunks")
    
    # Unterstützte Dateiformate anzeigen
    print(f"\n🔧 Unterstützte Formate: {', '.join(supported_extensions)}")
    
    return final_docs, loaded_files

def load_single_file(file_path):
    """Load a single file with format detection and adaptive chunking"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Pfad ist keine Datei: {file_path}")
        
        file_extension = file_path.suffix.lower()
        file_size_mb = get_file_size_mb(str(file_path))
        supported_extensions = get_supported_file_types()
        
        print(f"📁 Lade Einzeldatei: {file_path.name} ({file_size_mb:.2f}MB)")
        
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
    """Get detailed statistics about files in a directory"""
    stats = {}
    size_stats = {'total_mb': 0, 'large_files': 0, 'medium_files': 0, 'small_files': 0}
    supported_extensions = get_supported_file_types()
    
    for file in Path(path).rglob("*"):
        if file.is_file():
            ext = file.suffix.lower()
            file_size_mb = get_file_size_mb(str(file))
            
            if ext in stats:
                stats[ext]['count'] += 1
                stats[ext]['total_size_mb'] += file_size_mb
            else:
                stats[ext] = {'count': 1, 'total_size_mb': file_size_mb}
            
            # Größenkategorien
            size_stats['total_mb'] += file_size_mb
            if file_size_mb > 5:
                size_stats['large_files'] += 1
            elif file_size_mb > 1:
                size_stats['medium_files'] += 1
            else:
                size_stats['small_files'] += 1
    
    supported_count = sum(data['count'] for ext, data in stats.items() if ext in supported_extensions)
    total_count = sum(data['count'] for data in stats.values())
    
    print(f"\n📈 Detaillierte Datei-Statistik für '{path}':")
    print(f"   📁 Gesamt: {total_count} Dateien ({size_stats['total_mb']:.2f}MB)")
    print(f"   ✅ Unterstützt: {supported_count} Dateien")
    print(f"   ❌ Nicht unterstützt: {total_count - supported_count} Dateien")
    
    print(f"\n📊 Größen-Verteilung:")
    print(f"   🔴 Große Dateien (>5MB): {size_stats['large_files']}")
    print(f"   🟡 Mittlere Dateien (1-5MB): {size_stats['medium_files']}")
    print(f"   🟢 Kleine Dateien (<1MB): {size_stats['small_files']}")
    
    if stats:
        print(f"\n📋 Dateitypen gefunden:")
        for ext, data in sorted(stats.items()):
            status = "✅" if ext in supported_extensions else "❌"
            avg_size = data['total_size_mb'] / data['count']
            print(f"   {status} {ext}: {data['count']} Datei(en), ⌀{avg_size:.2f}MB")
    
    return stats, size_stats