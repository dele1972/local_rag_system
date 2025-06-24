# token_tracker.py

## 📋 Übersicht

- **Datei:** `token_tracker.py`
- **Zeilen:** 239
- **Analysiert:** 2025-06-24T10:56:33

## 📦 Imports

### Standard Library
- `from datetime import datetime`
- `from typing import Dict, List, Optional, Tuple`
- `import re`

### Third Party
- `from dataclasses import dataclass, field`
- `import tiktoken`
- `import time`

## 🔧 Konstanten & Variablen

- 📝 **`token_tracker`** = `TokenTracker()`

## 🏗️ Klassen

### `TokenUsage`

**Beschreibung:** Datenklasse für Token-Verbrauch pro Anfrage

#### Methoden

- 🔧 **`cost_estimate(self) -> float`**
  - Schätzt die Kosten basierend auf lokalen Ressourcen (CPU-Zeit)

### `SessionStats`

**Beschreibung:** Session-weite Token-Statistiken

#### Methoden

- 🔧 **`total_tokens(self) -> int`**
  - Führt aus: return self.total_input_tokens + self.total_output...

- 🔧 **`average_tokens_per_request(self) -> float`**
  - Führt aus: return self.total_tokens / max(1, self.total_reque...

- 🔧 **`average_processing_time(self) -> float`**
  - Führt aus: return self.total_processing_time / max(1, self.to...

- 🔧 **`total_estimated_cost(self) -> float`**
  - Führt aus: return sum((req.cost_estimate for req in self.requ...

### `TokenCounter`

**Beschreibung:** Token-Counter für verschiedene Modelle

#### Methoden

- ⚙️ **`count_tokens(self, text: str, model_name: str = 'default') -> int`**
  - Zählt Token in einem Text

**Private Methoden:**
- `__init__()` - Führt aus: self.tiktoken_encod...
- `_try_init_tiktoken()` - Versucht tiktoken zu initialis...
- `_get_model_key()` - Extrahiert den Modell-Typ aus ...

### `TokenTracker`

**Beschreibung:** Haupt-Token-Tracking-Klasse

#### Methoden

- ⚙️ **`start_request(self)`**
  - Startet die Zeitmessung für eine neue Anfrage

- ⚙️ **`track_request(self, question: str, answer: str, context: str, model_name: str) -> TokenUsage`**
  - Verfolgt eine komplette Anfrage und aktualisiert Statistiken

- ⚙️ **`get_session_summary(self) -> Dict`**
  - Gibt eine Zusammenfassung der Session-Statistiken zurück

- ⚙️ **`get_recent_requests(self, limit: int = 10) -> List[Dict]`**
  - Gibt die letzten N Anfragen zurück

- ⚙️ **`reset_session(self)`**
  - Setzt die Session-Statistiken zurück

- ⚙️ **`format_token_info(self, usage: TokenUsage) -> str`**
  - Formatiert Token-Informationen für die Anzeige

**Private Methoden:**
- `__init__()` - Führt aus: self.counter = Toke...
- `_update_session_stats()` - Aktualisiert die Session-weite...
