# config.py
import os
import logging
from pathlib import Path

class ConfigManager:
    def __init__(self):
        self.base_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
        self.documents_path = os.path.join(self.base_path, "documents")
        
        # Verfügbare Modelle mit Token-Limits und Metadaten
        self.model_config = {
            "llama3.2": {
                "display_name": "Llama 3.2",
                "token_limit": 8192,
                "recommended_chain": "stuff",
                "description": "Neuestes Llama-Modell mit hoher Kapazität"
            },
            "gemma3:1b": {
                "display_name": "Gemma 3 (1B)",
                "token_limit": 2048,
                "recommended_chain": "map_reduce",
                "description": "Kleines, schnelles Modell"
            },
            "gemma3:12b": {
                "display_name": "Gemma 3 (12B)",
                "token_limit": 4096,
                "recommended_chain": "stuff",
                "description": "Größeres Gemma-Modell mit besserer Qualität"
            },
            "qwen3:4b": {
                "display_name": "Qwen 3 (4B)",
                "token_limit": 8192,
                "recommended_chain": "stuff",
                "description": "Mittelgroßes Qwen-Modell"
            },
            "qwen3:8b": {
                "display_name": "Qwen 3 (8B)",
                "token_limit": 8192,
                "recommended_chain": "stuff",
                "description": "Größeres Qwen-Modell mit hoher Leistung"
            },
            "mistral": {
                "display_name": "Mistral 7B",
                "token_limit": 4096,
                "recommended_chain": "stuff",
                "description": "Vielseitiges Modell mit guter Balance"
            },
            "phi4-mini-reasoning:3.8b": {
                "display_name": "Phi-4 Mini Reasoning",
                "token_limit": 4096,
                "recommended_chain": "stuff",
                "description": "Spezialisiert auf logisches Denken"
            },
            "deepseek-r1:8b": {
                "display_name": "DeepSeek R1-0528",
                "token_limit": 8192,
                "recommended_chain": "stuff",
                "description": "Neuestes DeepSeek-Modell"
            },
            "deepseek-r1": {
                "display_name": "DeepSeek R1",
                "token_limit": 8192,
                "recommended_chain": "stuff",
                "description": "Vorläufer DeepSeek-Modell"
            }
        }
        
        # Detect if running in Docker
        self.in_docker = os.path.exists("/.dockerenv")
        
        # Set the Ollama base URL based on environment
        if self.in_docker:
            self.ollama_base_url = "http://host.docker.internal:11434"
        else:
            self.ollama_base_url = "http://localhost:11434"
        
        # Logging-Konfiguration
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Token-Management Einstellungen
        self.token_safety_margin = 0.1  # 10% Sicherheitspuffer
        self.min_context_tokens = 100   # Minimum für sinnvollen Context
        self.default_answer_reserve = 500  # Token für Antwort reservieren
        
        # Setup Logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Konfiguriert das Logging-System"""
        # Root Logger konfigurieren
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=self.log_format,
            handlers=[
                logging.StreamHandler(),  # Console output
            ]
        )
        
        # App-spezifische Logger
        self.logger = logging.getLogger("RAG-App")
        
        # Externe Libraries weniger verbose machen
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
        self.logger.info(f"Logging konfiguriert: Level={self.log_level}, Docker={self.in_docker}")
    
    def get_documents_path(self):
        return self.documents_path
    
    def get_available_models(self):
        """Gibt Liste der verfügbaren Modellnamen zurück"""
        return list(self.model_config.keys())
    
    def get_model_info(self, model_name):
        """Gibt detaillierte Informationen zu einem Modell zurück"""
        return self.model_config.get(model_name, {
            "display_name": model_name,
            "token_limit": 1024,  # Konservativer Fallback
            "recommended_chain": "map_reduce",
            "description": "Unbekanntes Modell - konservative Einstellungen"
        })
    
    def get_model_token_limit(self, model_name):
        """Gibt das Token-Limit für ein Modell zurück"""
        model_info = self.get_model_info(model_name)
        return model_info["token_limit"]
    
    def get_recommended_chain_type(self, model_name):
        """Gibt den empfohlenen Chain-Typ für ein Modell zurück"""
        model_info = self.get_model_info(model_name)
        return model_info["recommended_chain"]
    
    def get_models_by_capability(self):
        """Gruppiert Modelle nach ihren Fähigkeiten"""
        small_models = []    # < 2048 tokens
        medium_models = []   # 2048-4096 tokens
        large_models = []    # > 4096 tokens
        
        for model_name, info in self.model_config.items():
            token_limit = info["token_limit"]
            model_entry = {
                "name": model_name,
                "display_name": info["display_name"],
                "token_limit": token_limit,
                "description": info["description"]
            }
            
            if token_limit < 2048:
                small_models.append(model_entry)
            elif token_limit <= 4096:
                medium_models.append(model_entry)
            else:
                large_models.append(model_entry)
        
        return {
            "small": small_models,
            "medium": medium_models,
            "large": large_models
        }
    
    def calculate_context_limits(self, model_name, prompt_overhead=None, answer_reserve=None):
        """Berechnet die verfügbaren Context-Token für ein Modell"""
        token_limit = self.get_model_token_limit(model_name)
        
        # Standardwerte oder übergebene Werte verwenden
        if prompt_overhead is None:
            prompt_overhead = 150
        if answer_reserve is None:
            answer_reserve = self.default_answer_reserve
        
        # Sicherheitspuffer anwenden
        effective_limit = int(token_limit * (1 - self.token_safety_margin))
        context_limit = effective_limit - prompt_overhead - answer_reserve
        
        return {
            "total_limit": token_limit,
            "effective_limit": effective_limit,
            "context_limit": max(self.min_context_tokens, context_limit),
            "prompt_overhead": prompt_overhead,
            "answer_reserve": answer_reserve,
            "safety_margin": self.token_safety_margin
        }
    
    def get_optimal_retrieval_k(self, model_name, file_count=1):
        """Bestimmt optimale Anzahl von Dokumenten für Retrieval"""
        token_limit = self.get_model_token_limit(model_name)
        
        # Basis-k basierend auf Token-Limit
        if token_limit < 2048:
            base_k = 3
        elif token_limit < 4096:
            base_k = 5
        elif token_limit < 8192:
            base_k = 8
        else:
            base_k = 10
        
        # Erhöhe k für große/viele Dateien
        if file_count > 10:
            base_k = min(base_k + 3, 15)
        
        return base_k    

    def get_ollama_base_url(self):
        return self.ollama_base_url
    
    def get_logger(self, name=None):
        """Gibt einen konfigurierten Logger zurück"""
        if name:
            return logging.getLogger(f"RAG-App.{name}")
        return self.logger
    
    def log_model_selection(self, model_name):
        """Loggt Informationen zur Modell-Auswahl"""
        model_info = self.get_model_info(model_name)
        context_limits = self.calculate_context_limits(model_name)
        
        self.logger.info(f"=== Modell-Konfiguration ===")
        self.logger.info(f"Gewähltes Modell: {model_info['display_name']} ({model_name})")
        self.logger.info(f"Token-Limit: {model_info['token_limit']:,}")
        self.logger.info(f"Empfohlener Chain-Typ: {model_info['recommended_chain']}")
        self.logger.info(f"Context-Limit: {context_limits['context_limit']:,} Token")
        self.logger.info(f"Retrieval-K: {self.get_optimal_retrieval_k(model_name)}")
        self.logger.info(f"Beschreibung: {model_info['description']}")

# Create a singleton instance
config = ConfigManager()