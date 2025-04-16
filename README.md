# local_rag_system
Lokales RAG-System mit Ollama und LangChain

## Voraussetzungen
- Python 3.11
- Ollama lokal installiert mit Modellen wie llama3, mistral, phi

## Start (lokal auf Windows ohne Docker)
```
pip install -r requirements.txt
python app/main.py
```

## Start mit Docker (Backend)
```
docker build -t lokales-rag .
docker run -v "${PWD.Path}/documents:/app/documents" -p 7860:7860 lokales-rag
docker run --name rag-system -v "${PWD.Path}/documents:/app/documents" -p 7860:7860 lokales-rag

```

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

```
docker stop <container-id>
docker rm <container-id>
docker run -v ... lokales-rag
```
---

**Hinweis**: Es wird vorausgesetzt, dass Ollama korrekt läuft und die Modelle geladen werden können. Die Modelle müssen vorher über `ollama pull` heruntergeladen werden.

Viel Spaß beim lokalen Fragenstellen! 🤓