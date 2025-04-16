# local_rag_system
Lokales RAG-System mit Ollama und LangChain

## Voraussetzungen
- Python 3.11
- Ollama lokal installiert mit Modellen wie llama3, mistral, phi


## 1. Start Backend (Docker)

### 0 - Erstelle den Docker Container (sofern noch nicht vorhanden)
```
docker build -t lokales-rag-claude .
```

### 1A - Starte das Backend (Variante A - per Docker Compose)
```
docker-compose up
```

### 1B - Starte das Backend (Variante B - manuell)
```
docker run --add-host=host.docker.internal:host-gateway -v "./documents:/app/documents" -p 7860:7860 lokales-rag-claude
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

```
docker stop <container-id>
docker rm <container-id>
docker run -v ... lokales-rag
```
---

**Hinweis**: Es wird vorausgesetzt, dass Ollama korrekt läuft und die Modelle geladen werden können. Die Modelle müssen vorher über `ollama pull` heruntergeladen werden.

Viel Spaß beim lokalen Fragenstellen! 🤓