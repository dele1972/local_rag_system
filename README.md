# local_rag_system
Lokales RAG-System mit Ollama und LangChain. Viel Spaß beim lokalen Fragenstellen! 🤓

# RAG System mit Ollama

## 🏗️ System-Architektur

[architecture](docs/architecture.md)

## 📦 Komponenten-Übersicht

| Komponente          | Datei                 | Zweck                        |
|---------------------|-----------------------|------------------------------|
| **UI**              | `ui.py`               | Gradio Web-Interface         |
| **RAG Engine**      | `rag.py`              | Frage-Antwort-System         |
| **Document Loader** | `loader.py`           | Multi-Format-Dokumentenlader |
| **Vector Store**    | `vectorstore.py`      | Embeddings & ChromaDB        |
| **Config**          | `config.py`           | Zentrale Konfiguration       |
| **Connection**      | `connection_utils.py` | Ollama-Verbindungsmanagement |

## 🔄 Datenfluss

1. **Dokumenten-Upload** → Format-Erkennung → Text-Extraktion → Chunking → Embeddings
2. **Frage-Input** → Vektor-Suche → Kontext-Assembly → LLM-Query → Antwort + Quellen

## 📋 Unterstützte Formate

|    | Format         | Zweck                                      |
|----|----------------|--------------------------------------------|
| 📄 | **Text**       | `.txt`, `.md`                              |
| 📕 | **PDF**        | Mit OCR-Support für gescannte Dokumente    |
| 📝 | **Word**       | `.docx`, `.doc` (LibreOffice erforderlich) |
| 📊 | **Excel**      | `.xlsx`, `.xls`                            |
| 📈 | **PowerPoint** | `.pptx`, `.ppt` (LibreOffice erforderlich) |

## 🚀 Quick Start

### Voraussetzungen

**Hinweis**: Es wird vorausgesetzt, dass Ollama korrekt läuft und die Modelle geladen werden können. Die Modelle müssen vorher über `ollama pull` heruntergeladen werden.

- Python 3.11 (Docker Container)
- Ollama (lokal installiert)
    - mit Modellen wie
        - [phi4-mini:3.8b](https://ollama.com/library/phi4-mini:3.8b)
        - [llama3](https://ollama.com/library/llama3.2:3b),
        - [mistral](https://ollama.com/library/mistral),
        - [deepseek-r1](https://ollama.com/library/deepseek-r1)
        - siehe [Perplexity Empfehlung](https://www.perplexity.ai/search/welches-ollama-modell-mit-bis-IK_81RgRRlGGwBkk38vR7w)
    - mit `ollama list` die Liste der installierten Modelle anzeigen
    - in `self.available_models = ["llama3.2", "mistral", "deepseek-r1"]` die Liste entsprechend anpassen

### 1. Start Backend (Docker)

#### 0 - Erstelle den Docker Container (sofern noch nicht vorhanden)
```
docker build -t lokales-rag-claude_v2 .
```

#### 1A - Starte das Backend (Variante A - per Docker Compose)
```
docker-compose up
```

#### 1B - Starte das Backend (Variante B - manuell)
```
docker run --add-host=host.docker.internal:host-gateway -v "./documents:/app/documents" -p 7860:7860 lokales-rag-claude_v2
```

### 2. Oberfläche aufrufen

Dann öffne deinen Browser unter: http://localhost:7860

## Nutzung
1. Wähle ein Modell aus (z. B. llama3)
2. Gib den Pfad zu deinen Dokumenten an (z. B. `./documents`)
3. Klicke auf "Laden"
4. Stelle eine Frage zu deinen Dokumenten

## Sonstiges

### Test, ob Ollama im Host erreichbar ist

```
curl http://localhost:11434/api/tags
```

### Docker Compose nach Änderungen

```
docker-compose down
```

```
docker-compose build --no-cache
```

```
docker-compose up
```

### Sonstige Docker Aktivitäten

#### Docker auf Urzustand setzen (Alle Container, Images, etc. löschen - bis auf aktive)

```
docker system prune -a --volumes
```

#### Alle Docker Container löschen

Wenn zu viele verschiedene Docker Container mit Docker run gebildet wurden, können alle Docker Container mit diesem Befehl gelöscht werden:
```
docker builder prune --all --force
```

#### Container neu bauen:

```
docker build -t lokales-rag .
```
-> Danach wieder mit run starten


#### Docker
```
docker ps
docker stop <container-id>
docker rm <container-id>
docker run -v ... lokales-rag
```

## Changelog

[changelog](docs/changelog.md)