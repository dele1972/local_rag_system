# local_rag_system
Lokales RAG-System mit Ollama und LangChain

## Voraussetzungen
- Python 3.11
- Ollama lokal installiert mit Modellen wie llama3, mistral, phi


## 1. Start Backend (Docker)

### 0 - Erstelle den Docker Container (sofern noch nicht vorhanden)
```
docker build -t lokales-rag-claude_v2 .
```

### 1A - Starte das Backend (Variante A - per Docker Compose)
```
docker-compose up
```

### 1B - Starte das Backend (Variante B - manuell)
```
docker run --add-host=host.docker.internal:host-gateway -v "./documents:/app/documents" -p 7860:7860 lokales-rag-claude_v2
```

## 2. Oberfläche aufrufen
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

### Nach Änderungen, den Container neu bauen:
```
docker build -t lokales-rag .
```
-> Danach wieder mit run starten


### Docker auf Urzustand (Alle Container, Images, etc. löschen - bis auf aktive)
```
docker system prune -a --volumes
```

### Alle Docker Container löschen
Wenn zu viele verschiedene Docker Container mit Docker run gebildet wurden, können alle Docker Container mit diesem Befehl gelöscht werden:
```
docker builder prune --all --force
```

### Docker foo

#### Docker Compose nach Änderungen

```
docker-compose down
```

```
docker-compose build --no-cache
```

```
docker-compose up
```

#### Docker
```
docker stop <container-id>
docker rm <container-id>
docker run -v ... lokales-rag
```

## Changehistory

### V2

#### Wichtigste Änderungen

Diese Änderungen bieten dir ein erheblich verbessertes RAG-System mit:

Höherer Antwortqualität durch angepasste Prompts und optimierte Chain-Typen
Persistenter Speicherung von Embeddings mit Chroma
Einer intuitiveren Benutzeroberfläche mit mehr Funktionen
Besserer Verwaltung von Vektorspeichern für verschiedene Dokumentensammlungen

#### 1. In app/rag.py

- **Prompt-Templates**: Ich habe ein anpassbares Prompt-Template hinzugefügt, das die Qualität der Antworten verbessert, indem es explizite Anweisungen für das LLM enthält.
- **Chain-Typen**: Die Funktion build_qa_chain unterstützt jetzt verschiedene Chain-Typen:
	- `stuff`: Standardmethode für kleinere Dokumentensammlungen
	- `map_reduce`: Besser für große Dokumentenmengen, verarbeitet jeden Chunk separat
	- `refine`: Iterativer Ansatz mit schrittweiser Verfeinerung der Antwort
	- `map_rerank`: Ordnet Antworten nach Relevanz

#### 2. In app/vectorstore.py

- **Chroma statt FAISS**: Ersetzt FAISS durch Chroma, das eine persistente Speicherung ermöglicht
- **Bessere Vektorstore-Verwaltung**:
	- Funktionen zum Auflisten, Laden und Löschen von Vektorspeichern
	- Verbesserte Fehlerbehandlung und Rückgabewerte
	- Strukturierte Verzeichnisorganisation für Vektorspeicher

#### 3. In app/ui.py

- **Tab-basierte UI**: Trennung von Einrichtung und Fragebereich für bessere Übersicht
- **Chain-Typ-Auswahl**: Dropdown zur Auswahl des gewünschten Chain-Typs
- **Vektorspeicher-Management**:
	- Anzeige und Auswahl vorhandener Vektorspeicher
	- Buttons zum Laden und Löschen von Vektorspeichern
	- Refresh-Funktion für die Vektorspeicher-Liste
- **Verbesserte Statusanzeigen**: Ausführlichere Informationen über den aktuellen Zustand des Systems

#### 4. In requirements.txt

- **Aktualisierte Abhängigkeiten**:
	- Hinzugefügt: `chromadb` für persistente Vektorspeicherung
	- Versionsangaben für bessere Kompatibilität
	- `sentence-transformers` als optionale Alternative zu Ollama-Embeddings

---

**Hinweis**: Es wird vorausgesetzt, dass Ollama korrekt läuft und die Modelle geladen werden können. Die Modelle müssen vorher über `ollama pull` heruntergeladen werden.

Viel Spaß beim lokalen Fragenstellen! 🤓