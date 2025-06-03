# Benutzerhandbuch: RAG mit Ollama und LangChain

Dieses Benutzerhandbuch erklärt die Verwendung des verbesserten RAG-Systems mit Ollama und LangChain.

## Inhaltsverzeichnis
1. [Überblick](#überblick)
2. [Modellauswahl](#modellauswahl)
3. [Chain-Typen](#chain-typen)
4. [Arbeiten mit Vektorspeichern](#arbeiten-mit-vektorspeichern)
5. [Workflow mit neuen Dokumenten](#workflow-mit-neuen-dokumenten)
6. [Fragen stellen und Antworten erhalten](#fragen-stellen-und-antworten-erhalten)
7. [Fehlerbehebung](#fehlerbehebung)

## Überblick

Das System besteht aus zwei Hauptbereichen:
- **Einrichtung**: Hier werden Modelle, Dokumente und Vektorspeicher verwaltet
- **Fragen und Antworten**: Hier können Fragen an die Dokumente gestellt werden

## Modellauswahl

- Das System lädt standardmäßig das erste verfügbare Modell aus der Konfiguration (in der Regel "llama3.2")
- Sie können jederzeit ein anderes Modell aus dem Dropdown "Modellwahl" auswählen
- Die Modellauswahl wirkt sich auf die Qualität der Antworten und die Geschwindigkeit des Systems aus

## Chain-Typen

Der Chain-Typ bestimmt, wie das System mit den Dokumenten arbeitet, **nachdem** sie bereits vektorisiert wurden:

- **stuff** (Standard): Fügt alle relevanten Dokumentenchunks in einen Prompt ein
  - Geeignet für: Kleinere Dokumente, einfache Fragen
  - Schnellste Methode

- **map_reduce**: Verarbeitet jeden Chunk separat und kombiniert die Ergebnisse
  - Geeignet für: Große Dokumentenmengen
  - Kann mit mehr Kontext arbeiten als "stuff"

- **refine**: Verfeinert die Antwort schrittweise mit jedem Dokument
  - Geeignet für: Komplexe Fragen, die eine nuancierte Antwort erfordern
  - Langsamer, aber oft präziser

- **map_rerank**: Bewertet Antworten nach ihrer Relevanz
  - Geeignet für: Situationen, in denen Ranking wichtig ist
  - Hilfreich bei mehrdeutigen Fragen

**Wichtig**: Der Chain-Typ beeinflusst **nicht** die Art der Vektorspeicherung selbst, sondern nur wie die gespeicherten Dokumente bei einer Anfrage verarbeitet werden. Sie können jederzeit zwischen verschiedenen Chain-Typen wechseln, ohne den Vektorspeicher neu erstellen zu müssen.

## Arbeiten mit Vektorspeichern

### Neuen Vektorspeicher erstellen:

1. Geben Sie den Pfad zu Ihren Dokumenten im Feld "Pfad zur Dokumentenbasis" ein
2. Im Feld "Name für neuen Vektorspeicher" wird automatisch ein Name generiert (Format: `vectorstore_ZEITSTEMPEL`)
   - Sie können diesen Namen bei Bedarf ändern
3. Klicken Sie auf "📚 Dokumente laden und vektorisieren"
   - Dies erstellt einen neuen Vektorspeicher mit dem angegebenen Namen
   - Der Status zeigt an, wie viele Dokumente verarbeitet wurden

### Vorhandene Vektorspeicher verwalten:

1. Klicken Sie auf "🔄 Vektorspeicher aktualisieren", um die Liste der verfügbaren Vektorspeicher zu aktualisieren
   - Diese Funktion lädt die Liste aller Vektorspeicher aus dem Verzeichnis `.vectorstores` im Dokumentenpfad
   - **Wichtig**: Nach dem Erstellen eines neuen Vektorspeichers müssen Sie auf diese Schaltfläche klicken, um ihn in der Liste zu sehen

2. Wählen Sie einen Vektorspeicher aus dem Dropdown "Vorhandene Vektorspeicher"

3. Klicken Sie auf "📂 Vektorspeicher laden", um den ausgewählten Vektorspeicher zu verwenden
   - Sie können dabei ein anderes Modell oder einen anderen Chain-Typ auswählen

4. Verwenden Sie "🗑️ Vektorspeicher löschen", um nicht mehr benötigte Vektorspeicher zu entfernen

## Workflow mit neuen Dokumenten

Wenn Sie neue Dokumente hinzufügen möchten, haben Sie zwei Möglichkeiten:

### Option 1: Mit laufendem System
1. Fügen Sie neue Dokumente dem Dokumentenverzeichnis hinzu
2. Erstellen Sie einen neuen Vektorspeicher mit "📚 Dokumente laden und vektorisieren"
   - Vergeben Sie einen aussagekräftigen Namen, um ihn später identifizieren zu können
3. Klicken Sie auf "🔄 Vektorspeicher aktualisieren", um die Liste zu aktualisieren
4. Wählen Sie den neuen Vektorspeicher aus der Liste und laden Sie ihn

### Option 2: Nach Container-Neustart
1. Fügen Sie neue Dokumente dem Host-Dokumentenverzeichnis hinzu
2. Starten Sie den Container neu (wenn nötig)
3. Folgen Sie den Schritten unter "Neuen Vektorspeicher erstellen"

**Hinweis**: Ein Container-Neustart ist nicht erforderlich, solange das Dokumentenverzeichnis im Container verfügbar ist (z.B. durch Volume-Mapping in Docker Compose).

## Fragen stellen und Antworten erhalten

1. Wechseln Sie zum Tab "Fragen und Antworten"
2. Geben Sie Ihre Frage in das Textfeld ein
3. Klicken Sie auf "🔍 Frage stellen" oder drücken Sie Enter
4. Die Antwort wird unterhalb angezeigt und enthält:
   - Die eigentliche Antwort auf Ihre Frage
   - Quellen, aus denen die Antwort generiert wurde
   - Informationen über das verwendete Modell und den Chain-Typ

## Fehlerbehebung

### Problem: Keine Verbindung zu Ollama
- Stellen Sie sicher, dass Ollama auf dem Host-System läuft
- Überprüfen Sie die Ollama-URL in der Konfiguration (`config.py`)
- Bei Verwendung von Docker: Stellen Sie sicher, dass `host.docker.internal` korrekt aufgelöst wird

### Problem: Vektorspeicher erscheint nicht in der Liste
- Klicken Sie auf "🔄 Vektorspeicher aktualisieren"
- Überprüfen Sie, ob der in "Pfad zur Dokumentenbasis" angegebene Pfad korrekt ist
- Prüfen Sie, ob im `.vectorstores`-Unterverzeichnis Daten angelegt wurden

### Problem: Leere oder falsche Antworten
- Probieren Sie einen anderen Chain-Typ, z.B. "refine" für komplexere Fragen
- Versuchen Sie es mit einem anderen Modell (z.B. "deepseek-r1" statt "llama3.2")
- Formulieren Sie Ihre Frage spezifischer oder verwenden Sie Begriffe aus den Dokumenten