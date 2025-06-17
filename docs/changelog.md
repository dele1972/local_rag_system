# Changelog

## Entwickler Version

### RAG Test Framework Suite - Vollständige Test-Komponente

#### 📦 Komponenten hinzugefügt:
- **test_rag.py** - Core Test Framework
- **test_utils.py** - Analysis & Visualization Utilities  
- **run_rag_tests.py** - CLI Interface & Orchestration

#### ✨ Gesamt-Features:
- **Systematische RAG-Evaluation:** Multi-Parameter Test-Matrix, Automatische Kombination von Modellen, Chunk-Größen und Dokumenten
- **Deutsche Dokument-Optimierung:** Speziell angepasste Parameter für deutsche Texte
- **Große Datei-Support:** Speicher-optimiert für 6.5MB+ Dokumente
- **Performance-Monitoring:** Detaillierte Messung von Response-Zeit und Ressourcenverbrauch, CPU/RAM-Überwachung während Tests
- **Visualisierung:** Interactive Dashboards & HTML-Reports
- **Parallelisierung:** Batch-Test-Ausführung für bessere Performance
- **Flexible Test-Modi:** Analyse, Quick-Test, Vollständiger Benchmark
- **JSON-Export:** Strukturierte Ergebnisse für weitere Analyse

#### 🎯 Verwendungszwecke:
- **Performance-Baseline:** Systematische RAG-System-Bewertung und RAG-Performance-Evaluation
- **Parameter-Optimierung:** Beste Chunk-Größe/Modell-Kombination finden, Optimierung für große deutsche Dokumente
- **Regression-Testing:** Continuous Integration für RAG-Änderungen
- **Capacity-Planning:** Resource-Verbrauch für große Dokumente, Performance-Baseline-Erstellung
- **Model-Selection:** Datenbasierte Modell-Auswahl

#### 🏗️ Architektur-Highlights:
- **RAGTestFramework:** Haupt-Engine mit 4 Kern-Methoden
- **3-Schichten-Design:** Separation of Concerns
- **Datenklassen-basiert:** Strukturierte Test-Konfiguration
- **Robuste Fehlerbehandlung:** Graceful Degradation bei Test-Fehlern
- **Modular aufgebaut:** Einzelne Komponenten verwendbar
- **CLI + Programmatic API:** Flexible Nutzung
- **TestConfiguration:** Datenstruktur für Test-Parameter
- **TestResult:** Strukturierte Ergebnis-Sammlung  

#### 🔧 Deutsche RAG-Optimierungen:
- **Chunk-Größen:** 500-2000 Token (deutsche Satzlängen)
- **Overlaps:** 50-200 Token (deutsche Grammatik-Strukturen)  
- **Encoding:** Robuste UTF-8/Umlaut-Behandlung
- **Model-Support:** Alle 4 Hauptmodelle integriert

### Automatisierte Dokumentation der Python Scripte (`/scripts/docgen.py`)
- Das Script erstellt automatisch kompakte Markdown-Dokumentation aus den Python-Dateien.

### Optimierungsprozess - Neue Debugging Funktionen (`rag.py`, `main.py`)
- Erweiterte Token-Logging-Funktion
- Document-Retrieval-Debugging
- Performance-Metriken sammeln
- Chunk-Analyse-Tools

### I001: MarkDown und HTML Formatierungen aus KI Antworten übernehmen bzw. darstellen
**Ziel:** Antworten sollen entsprechend ihrer Formatierungsbefehle formatiert angezeigt werden

**Betroffene Dateien:**
- app/ui.py
- Dockerfile
- requirements.txt
  - [bleach](https://github.com/mozilla/bleach)

**Config.py Verbesserungen**:

✅ Token-Limits konfigurierbar: Alle Modelle mit spezifischen Token-Limits
✅ Erweiterte Modell-Metadaten: Display-Namen, Beschreibungen, Chain-Empfehlungen
✅ Intelligente Berechnungen: Context-Limits mit Sicherheitspuffern
✅ Verbessertes Logging: Strukturiertes Logging mit konfigurierbaren Leveln
✅ Utility-Funktionen: Modell-Gruppierung, optimale Retrieval-Einstellungen

**RAG.py Anpassungen**:

✅ Config-Integration: Verwendet Token-Limits aus der Config
✅ Verbessertes Logging: Nutzt konfigurierten Logger
✅ Intelligente Empfehlungen: Chain-Type-Empfehlungen basierend auf Modell

---

## Version 1.2

- Unterstützung von weiteren Dokumentformaten
    - Bislang wurden nur reguläre PDF- und reine Textdateien unterstützt. In dieser Version wurden weitere Formate hinzugefügt die nun eingelesen werden können:
    - PDF/OCR - Gescannte PDF Dokumente werden nun erkannt und per OCR eingelesen
        - mittels Googles [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract)
    - DOC/DOCX - Word Dateien
    - XLS/XLSX - Excel Dateien
    - PPT/PPTX - Powerpoint Dateien
    - MD - Markdown Textdateien
- Fehlerbehebung
    - Vektorstore lässt sich laden und löschen (inkl. richtige Darstellung und refresh in der DropDown Liste)
    - Beim initialen Aufruf des Frontends konnte es zu Zeitproblemen kommen, was dazu führte, dass ein Vektorstore nicht auf Anhieb erstellt werden konnte

## Version 1.1

### Wichtigste Änderungen

Diese Änderungen bieten dir ein erheblich verbessertes RAG-System mit:

- Höherer Antwortqualität durch angepasste Prompts und optimierte Chain-Typen
- Persistenter Speicherung von Embeddings mit Chroma
- Einer intuitiveren Benutzeroberfläche mit mehr Funktionen
- Besserer Verwaltung von Vektorspeichern für verschiedene Dokumentensammlungen

### 1. In app/rag.py

- **Prompt-Templates**: Ich habe ein anpassbares Prompt-Template hinzugefügt, das die Qualität der Antworten verbessert, indem es explizite Anweisungen für das LLM enthält.
- **Chain-Typen**: Die Funktion build_qa_chain unterstützt jetzt verschiedene Chain-Typen:
	- `stuff`: Standardmethode für kleinere Dokumentensammlungen
	- `map_reduce`: Besser für große Dokumentenmengen, verarbeitet jeden Chunk separat
	- `refine`: Iterativer Ansatz mit schrittweiser Verfeinerung der Antwort
	- `map_rerank`: Ordnet Antworten nach Relevanz

### 2. In app/vectorstore.py

- **Chroma statt FAISS**: Ersetzt FAISS durch Chroma, das eine persistente Speicherung ermöglicht
- **Bessere Vektorstore-Verwaltung**:
	- Funktionen zum Auflisten, Laden und Löschen von Vektorspeichern
	- Verbesserte Fehlerbehandlung und Rückgabewerte
	- Strukturierte Verzeichnisorganisation für Vektorspeicher

### 3. In app/ui.py

- **Tab-basierte UI**: Trennung von Einrichtung und Fragebereich für bessere Übersicht
- **Chain-Typ-Auswahl**: Dropdown zur Auswahl des gewünschten Chain-Typs
- **Vektorspeicher-Management**:
	- Anzeige und Auswahl vorhandener Vektorspeicher
	- Buttons zum Laden und Löschen von Vektorspeichern
	- Refresh-Funktion für die Vektorspeicher-Liste
- **Verbesserte Statusanzeigen**: Ausführlichere Informationen über den aktuellen Zustand des Systems

### 4. In requirements.txt

- **Aktualisierte Abhängigkeiten**:
	- Hinzugefügt: `chromadb` für persistente Vektorspeicherung
	- Versionsangaben für bessere Kompatibilität
	- `sentence-transformers` als optionale Alternative zu Ollama-Embeddings
