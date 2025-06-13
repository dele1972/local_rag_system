# app/token_tracker.py - Token-Counting und Statistiken für Ollama-Modelle

import time
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import tiktoken

@dataclass
class TokenUsage:
    """Datenklasse für Token-Verbrauch pro Anfrage"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_name: str
    timestamp: datetime
    question: str
    processing_time: float = 0.0
    
    @property
    def cost_estimate(self) -> float:
        """Schätzt die Kosten basierend auf lokalen Ressourcen (CPU-Zeit)"""
        # Vereinfachte Kostenschätzung basierend auf Token-Anzahl und Verarbeitungszeit
        base_cost_per_1k_tokens = 0.001  # Symbolischer Wert für lokale Verarbeitung
        return (self.total_tokens / 1000) * base_cost_per_1k_tokens

@dataclass 
class SessionStats:
    """Session-weite Token-Statistiken"""
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_processing_time: float = 0.0
    requests: List[TokenUsage] = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.now)
    
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens
    
    @property
    def average_tokens_per_request(self) -> float:
        return self.total_tokens / max(1, self.total_requests)
    
    @property
    def average_processing_time(self) -> float:
        return self.total_processing_time / max(1, self.total_requests)
    
    @property
    def total_estimated_cost(self) -> float:
        return sum(req.cost_estimate for req in self.requests)

class TokenCounter:
    """Token-Counter für verschiedene Modelle"""
    
    # Modell-spezifische Token-Schätzungen (falls tiktoken nicht verfügbar)
    MODEL_TOKEN_RATIOS = {
        'llama2': 4.0,      # ~4 Zeichen pro Token
        'llama3': 3.8,      # Etwas effizienter
        'mistral': 4.2,
        'codellama': 3.5,   # Code-Modelle sind oft effizienter
        'phi': 4.1,
        'gemma': 3.9,
        'default': 4.0
    }
    
    def __init__(self):
        self.tiktoken_encoder = None
        self._try_init_tiktoken()
    
    def _try_init_tiktoken(self):
        """Versucht tiktoken zu initialisieren (falls verfügbar)"""
        try:
            self.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # tiktoken nicht verfügbar, verwende Fallback
            self.tiktoken_encoder = None
    
    def count_tokens(self, text: str, model_name: str = "default") -> int:
        """
        Zählt Token in einem Text
        
        Args:
            text: Der zu analysierende Text
            model_name: Name des Modells für modell-spezifische Schätzungen
            
        Returns:
            Geschätzte Anzahl Token
        """
        if not text:
            return 0
        
        # Tiktoken verwenden falls verfügbar (genauer)
        if self.tiktoken_encoder is not None:
            try:
                return len(self.tiktoken_encoder.encode(text))
            except Exception:
                pass
        
        # Fallback: Modell-spezifische Schätzung
        model_key = self._get_model_key(model_name)
        char_per_token = self.MODEL_TOKEN_RATIOS.get(model_key, 4.0)
        
        # Berücksichtige Whitespace und Sonderzeichen
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        estimated_tokens = len(cleaned_text) / char_per_token
        
        # Mindestens 1 Token, aufgerundet
        return max(1, int(estimated_tokens + 0.5))
    
    def _get_model_key(self, model_name: str) -> str:
        """Extrahiert den Modell-Typ aus dem vollständigen Modellnamen"""
        model_name_lower = model_name.lower()
        
        for key in self.MODEL_TOKEN_RATIOS.keys():
            if key in model_name_lower:
                return key
        
        return 'default'

class TokenTracker:
    """Haupt-Token-Tracking-Klasse"""
    
    def __init__(self):
        self.counter = TokenCounter()
        self.session_stats = SessionStats()
        self._current_request_start = None
    
    def start_request(self):
        """Startet die Zeitmessung für eine neue Anfrage"""
        self._current_request_start = time.time()
    
    def track_request(self, 
                     question: str, 
                     answer: str, 
                     context: str, 
                     model_name: str) -> TokenUsage:
        """
        Verfolgt eine komplette Anfrage und aktualisiert Statistiken
        
        Args:
            question: Die gestellte Frage
            answer: Die generierte Antwort
            context: Der verwendete Kontext aus den Dokumenten
            model_name: Name des verwendeten Modells
            
        Returns:
            TokenUsage-Objekt mit den Details
        """
        # Token zählen
        input_tokens = self.counter.count_tokens(f"{context}\n{question}", model_name)
        output_tokens = self.counter.count_tokens(answer, model_name)
        total_tokens = input_tokens + output_tokens
        
        # Verarbeitungszeit berechnen
        processing_time = 0.0
        if self._current_request_start is not None:
            processing_time = time.time() - self._current_request_start
            self._current_request_start = None
        
        # TokenUsage-Objekt erstellen
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model_name=model_name,
            timestamp=datetime.now(),
            question=question[:100] + "..." if len(question) > 100 else question,
            processing_time=processing_time
        )
        
        # Session-Statistiken aktualisieren
        self._update_session_stats(usage)
        
        return usage
    
    def _update_session_stats(self, usage: TokenUsage):
        """Aktualisiert die Session-weiten Statistiken"""
        self.session_stats.total_requests += 1
        self.session_stats.total_input_tokens += usage.input_tokens
        self.session_stats.total_output_tokens += usage.output_tokens
        self.session_stats.total_processing_time += usage.processing_time
        self.session_stats.requests.append(usage)
        
        # Begrenze gespeicherte Anfragen (Memory-Management)
        if len(self.session_stats.requests) > 100:
            self.session_stats.requests = self.session_stats.requests[-50:]
    
    def get_session_summary(self) -> Dict:
        """Gibt eine Zusammenfassung der Session-Statistiken zurück"""
        stats = self.session_stats
        
        return {
            'total_requests': stats.total_requests,
            'total_tokens': stats.total_tokens,
            'total_input_tokens': stats.total_input_tokens,
            'total_output_tokens': stats.total_output_tokens,
            'average_tokens_per_request': round(stats.average_tokens_per_request, 1),
            'average_processing_time': round(stats.average_processing_time, 2),
            'total_processing_time': round(stats.total_processing_time, 2),
            'session_duration': str(datetime.now() - stats.session_start).split('.')[0],
            'estimated_total_cost': round(stats.total_estimated_cost, 4)
        }
    
    def get_recent_requests(self, limit: int = 10) -> List[Dict]:
        """Gibt die letzten N Anfragen zurück"""
        recent = self.session_stats.requests[-limit:] if self.session_stats.requests else []
        
        return [
            {
                'timestamp': req.timestamp.strftime('%H:%M:%S'),
                'question': req.question,
                'input_tokens': req.input_tokens,
                'output_tokens': req.output_tokens,
                'total_tokens': req.total_tokens,
                'processing_time': round(req.processing_time, 2),
                'model': req.model_name,
                'cost_estimate': round(req.cost_estimate, 4)
            }
            for req in reversed(recent)
        ]
    
    def reset_session(self):
        """Setzt die Session-Statistiken zurück"""
        self.session_stats = SessionStats()
    
    def format_token_info(self, usage: TokenUsage) -> str:
        """Formatiert Token-Informationen für die Anzeige"""
        return f"""📊 **Token-Verbrauch:**
• Input: {usage.input_tokens:,} Token
• Output: {usage.output_tokens:,} Token  
• Gesamt: {usage.total_tokens:,} Token
• Verarbeitungszeit: {usage.processing_time:.2f}s
• Modell: {usage.model_name}
• Geschätzte Kosten: ${usage.cost_estimate:.4f}"""

# Globale Instanz für die Anwendung
token_tracker = TokenTracker()