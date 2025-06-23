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
                "description": "Neuestes Llama-Modell mit hoher Kapazität",
                "language_support": "multilingual"
            },
            "gemma3:1b": {
                "display_name": "Gemma 3 (1B)",
                "token_limit": 2048,
                "recommended_chain": "map_reduce",
                "description": "Kleines, schnelles Modell",
                "language_support": "multilingual"
            },
            "gemma3:12b": {
                "display_name": "Gemma 3 (12B)",
                "token_limit": 4096,
                "recommended_chain": "stuff",
                "description": "Größeres Gemma-Modell mit besserer Qualität",
                "language_support": "multilingual"
            },
            "qwen3:4b": {
                "display_name": "Qwen 3 (4B)",
                "token_limit": 8192,
                "recommended_chain": "stuff",
                "description": "Mittelgroßes Qwen-Modell",
                "language_support": "multilingual"
            },
            "qwen3:8b": {
                "display_name": "Qwen 3 (8B)",
                "token_limit": 8192,
                "recommended_chain": "stuff",
                "description": "Größeres Qwen-Modell mit hoher Leistung",
                "language_support": "multilingual"
            },
            "mistral": {
                "display_name": "Mistral 7B",
                "token_limit": 4096,
                "recommended_chain": "stuff",
                "description": "Vielseitiges Modell mit guter Balance",
                "language_support": "multilingual"
            },
            "phi4-mini-reasoning:3.8b": {
                "display_name": "Phi-4 Mini Reasoning",
                "token_limit": 4096,
                "recommended_chain": "stuff",
                "description": "Spezialisiert auf logisches Denken",
                "language_support": "multilingual"
            },
            "deepseek-r1:8b": {
                "display_name": "DeepSeek R1-0528",
                "token_limit": 8192,
                "recommended_chain": "stuff",
                "description": "Neuestes DeepSeek-Modell",
                "language_support": "multilingual"
            },
            "deepseek-r1": {
                "display_name": "DeepSeek R1",
                "token_limit": 8192,
                "recommended_chain": "stuff",
                "description": "Vorläufer DeepSeek-Modell",
                "language_support": "multilingual"
            }
        }
        
        # Erweiterte Embedding-Konfiguration
        self.embedding_config = {
            # Ollama-basierte Embeddings
            "ollama": {
                "nomic-embed-text": {
                    "display_name": "Nomic Embed Text",
                    "type": "ollama",
                    "model_size": "137MB",
                    "dimensions": 768,
                    "max_tokens": 2048,
                    "language_support": "multilingual",
                    "german_quality": "medium",
                    "description": "Standard Ollama Embedding-Modell"
                },
                "mxbai-embed-large": {
                    "display_name": "MxBai Embed Large",
                    "type": "ollama", 
                    "model_size": "669MB",
                    "dimensions": 1024,
                    "max_tokens": 2048,
                    "language_support": "multilingual",
                    "german_quality": "good",
                    "description": "Größeres Ollama Embedding-Modell mit besserer Qualität"
                },
                "all-minilm": {
                    "display_name": "All-MiniLM-L6-v2",
                    "type": "ollama",
                    "model_size": "80MB",
                    "dimensions": 384,
                    "max_tokens": 512,
                    "language_support": "multilingual",
                    "german_quality": "medium",
                    "description": "Schnelles, kompaktes Embedding-Modell"
                }
            },
            
            # Hugging Face Embeddings - Optimiert für Deutsch
            "huggingface": {
                "intfloat/multilingual-e5-large": {
                    "display_name": "Multilingual E5 Large",
                    "type": "huggingface",
                    "model_size": "2.24GB",
                    "dimensions": 1024,
                    "max_tokens": 512,
                    "language_support": "multilingual",
                    "german_quality": "excellent",
                    "description": "Hochqualitatives multilinguale Embedding-Modell, exzellent für Deutsch"
                },
                "intfloat/multilingual-e5-base": {
                    "display_name": "Multilingual E5 Base",
                    "type": "huggingface",
                    "model_size": "1.11GB",
                    "dimensions": 768,
                    "max_tokens": 512,
                    "language_support": "multilingual",
                    "german_quality": "very_good",
                    "description": "Gutes multilinguale Embedding-Modell mit besserer Performance"
                },
                "intfloat/multilingual-e5-small": {
                    "display_name": "Multilingual E5 Small",
                    "type": "huggingface",
                    "model_size": "471MB",
                    "dimensions": 384,
                    "max_tokens": 512,
                    "language_support": "multilingual",
                    "german_quality": "good",
                    "description": "Kompakte Version des E5-Modells"
                },
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
                    "display_name": "Paraphrase Multilingual MiniLM",
                    "type": "huggingface",
                    "model_size": "471MB",
                    "dimensions": 384,
                    "max_tokens": 512,
                    "language_support": "multilingual",
                    "german_quality": "good",
                    "description": "Spezialisiert auf Paraphrasierung, gut für deutsche Texte"
                },
                "distiluse-base-multilingual-cased": {
                    "display_name": "DistilUSE Multilingual",
                    "type": "huggingface",
                    "model_size": "492MB",
                    "dimensions": 512,
                    "max_tokens": 512,
                    "language_support": "multilingual",
                    "german_quality": "good",
                    "description": "Distilled Universal Sentence Encoder, multilingual"
                }
            }
        }
        
        # Standard-Embedding-Modell für deutsche Texte
        self.default_embedding_model = "intfloat/multilingual-e5-base"
        self.fallback_embedding_model = "mxbai-embed-large"
        
        # Detect if running in Docker
        self.in_docker = os.path.exists("/.dockerenv")
        
        # Set the Ollama base URL based on environment
        if self.in_docker:
            self.ollama_base_url = "http://host.docker.internal:11434"
        else:
            self.ollama_base_url = "http://localhost:11434"
        
        # Logging-Konfiguration
        # (!) Hier wird das Log-Level für die gesamte App gesetzt
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Token-Management Einstellungen
        self.token_safety_margin = 0.1  # 10% Sicherheitspuffer
        self.min_context_tokens = 100   # Minimum für sinnvollen Context
        self.default_answer_reserve = 500  # Token für Antwort reservieren
        
        # Erweiterte Embedding-Einstellungen
        self.embedding_settings = {
            "chunk_size_mapping": {
                "small": {"chunk_size": 512, "chunk_overlap": 100},
                "medium": {"chunk_size": 1000, "chunk_overlap": 200}, 
                "large": {"chunk_size": 1500, "chunk_overlap": 300}
            },
            "similarity_thresholds": {
                "high_quality": 0.8,    # Für E5-Modelle
                "medium_quality": 0.7,  # Für Standard-Modelle
                "low_quality": 0.6      # Für einfache Modelle
            },
            "retrieval_strategies": {
                "precise": {"k": 3, "fetch_k": 10},
                "balanced": {"k": 5, "fetch_k": 20},
                "comprehensive": {"k": 10, "fetch_k": 50}
            }
        }
        
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
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        
        self.logger.info(f"Logging konfiguriert: Level={self.log_level}, Docker={self.in_docker}")
    
    def get_documents_path(self):
        return self.documents_path

    def get_supported_file_types(self):
        """Gibt eine Liste der unterstützten Dateiformate zurück"""
        return [
            ".txt", ".md",           # Text
            ".pdf",                  # PDF
            ".docx", ".doc",         # Word
            ".xlsx", ".xls",         # Excel
            ".pptx", ".ppt"          # PowerPoint
        ]

    def get_available_models(self):
        """Gibt Liste der verfügbaren Modellnamen zurück"""
        return list(self.model_config.keys())
    
    def get_model_info(self, model_name):
        """Gibt detaillierte Informationen zu einem Modell zurück"""
        return self.model_config.get(model_name, {
            "display_name": model_name,
            "token_limit": 1024,  # Konservativer Fallback
            "recommended_chain": "map_reduce",
            "description": "Unbekanntes Modell - konservative Einstellungen",
            "language_support": "unknown"
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
                "description": info["description"],
                "language_support": info.get("language_support", "unknown")
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
    
    def get_available_embedding_models(self):
        """Gibt alle verfügbaren Embedding-Modelle zurück"""
        all_models = {}
        
        # Ollama-Modelle
        for model_name, info in self.embedding_config["ollama"].items():
            all_models[model_name] = {
                **info,
                "full_name": model_name
            }
        
        # Hugging Face Modelle
        for model_name, info in self.embedding_config["huggingface"].items():
            all_models[model_name] = {
                **info,
                "full_name": model_name
            }
        
        return all_models
    
    def get_embedding_model_info(self, model_name):
        """Gibt detaillierte Informationen zu einem Embedding-Modell zurück"""
        # Suche in Ollama-Modellen
        if model_name in self.embedding_config["ollama"]:
            return self.embedding_config["ollama"][model_name]
        
        # Suche in Hugging Face Modellen
        if model_name in self.embedding_config["huggingface"]:
            return self.embedding_config["huggingface"][model_name]
        
        # Fallback für unbekannte Modelle
        return {
            "display_name": model_name,
            "type": "unknown",
            "model_size": "unknown",
            "dimensions": 768,
            "max_tokens": 512,
            "language_support": "unknown",
            "german_quality": "unknown",
            "description": f"Unbekanntes Embedding-Modell: {model_name}"
        }
    
    def get_best_embedding_model_for_german(self):
        """Gibt das beste verfügbare Embedding-Modell für deutsche Texte zurück"""
        # Priorisiere Modelle nach deutscher Qualität
        german_models = []
        
        for provider in ["huggingface", "ollama"]:
            for model_name, info in self.embedding_config[provider].items():
                quality_score = {
                    "excellent": 5,
                    "very_good": 4,
                    "good": 3,
                    "medium": 2,
                    "low": 1,
                    "unknown": 0
                }.get(info.get("german_quality", "unknown"), 0)
                
                german_models.append({
                    "name": model_name,
                    "quality_score": quality_score,
                    "info": info
                })
        
        # Sortiere nach Qualität
        german_models.sort(key=lambda x: x["quality_score"], reverse=True)
        
        if german_models:
            return german_models[0]["name"]
        else:
            return self.fallback_embedding_model
    
    def get_embedding_models_by_quality(self):
        """Gruppiert Embedding-Modelle nach deutscher Qualität"""
        quality_groups = {
            "excellent": [],
            "very_good": [],
            "good": [],
            "medium": [],
            "low": []
        }
        
        for provider in ["huggingface", "ollama"]:
            for model_name, info in self.embedding_config[provider].items():
                quality = info.get("german_quality", "unknown")
                if quality in quality_groups:
                    quality_groups[quality].append({
                        "name": model_name,
                        "display_name": info["display_name"],
                        "type": info["type"],
                        "model_size": info["model_size"],
                        "dimensions": info["dimensions"],
                        "description": info["description"]
                    })
        
        return quality_groups
    
    def get_similarity_threshold(self, embedding_model):
        """Gibt den empfohlenen Similarity-Threshold für ein Embedding-Modell zurück"""
        model_info = self.get_embedding_model_info(embedding_model)
        german_quality = model_info.get("german_quality", "medium")
        
        threshold_mapping = {
            "excellent": self.embedding_settings["similarity_thresholds"]["high_quality"],
            "very_good": self.embedding_settings["similarity_thresholds"]["high_quality"],
            "good": self.embedding_settings["similarity_thresholds"]["medium_quality"],
            "medium": self.embedding_settings["similarity_thresholds"]["medium_quality"],
            "low": self.embedding_settings["similarity_thresholds"]["low_quality"]
        }
        
        return threshold_mapping.get(german_quality, 0.6)
    
    def get_optimal_chunk_size(self, embedding_model, file_size_mb=None):
        """Bestimmt optimale Chunk-Größe basierend auf Embedding-Modell und Dateigröße"""
        model_info = self.get_embedding_model_info(embedding_model)
        max_tokens = model_info.get("max_tokens", 512)
        
        # Basis-Chunk-Größe basierend auf Modell-Kapazität
        if max_tokens >= 2048:
            size_category = "large"
        elif max_tokens >= 1024:
            size_category = "medium"
        else:
            size_category = "small"
        
        chunk_config = self.embedding_settings["chunk_size_mapping"][size_category]
        
        # Anpassung basierend auf Dateigröße
        if file_size_mb:
            if file_size_mb > 10:  # Große Dateien
                chunk_config["chunk_size"] = min(chunk_config["chunk_size"] * 1.5, max_tokens * 2)
                chunk_config["chunk_overlap"] = min(chunk_config["chunk_overlap"] * 1.5, chunk_config["chunk_size"] * 0.3)
            elif file_size_mb < 1:  # Kleine Dateien
                chunk_config["chunk_size"] = max(chunk_config["chunk_size"] * 0.7, 200)
                chunk_config["chunk_overlap"] = max(chunk_config["chunk_overlap"] * 0.7, 50)
        
        return chunk_config
    
    def get_retrieval_strategy(self, embedding_model, document_count=1):
        """Bestimmt optimale Retrieval-Strategie"""
        model_info = self.get_embedding_model_info(embedding_model)
        german_quality = model_info.get("german_quality", "medium")
        
        # Basis-Strategie basierend auf Modell-Qualität
        if german_quality in ["excellent", "very_good"]:
            base_strategy = "precise"
        elif german_quality == "good":
            base_strategy = "balanced"
        else:
            base_strategy = "comprehensive"
        
        strategy = self.embedding_settings["retrieval_strategies"][base_strategy].copy()
        
        # Anpassung basierend auf Dokumentenanzahl
        if document_count > 20:
            strategy["k"] = min(strategy["k"] * 2, 15)
            strategy["fetch_k"] = min(strategy["fetch_k"] * 2, 100)
        elif document_count < 5:
            strategy["k"] = max(strategy["k"] // 2, 2)
            strategy["fetch_k"] = max(strategy["fetch_k"] // 2, 5)
        
        return strategy
    
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
    
    def log_embedding_selection(self, embedding_model):
        """Loggt Informationen zur Embedding-Modell-Auswahl"""
        model_info = self.get_embedding_model_info(embedding_model)
        similarity_threshold = self.get_similarity_threshold(embedding_model)
        
        self.logger.info(f"=== Embedding-Konfiguration ===")
        self.logger.info(f"Gewähltes Embedding-Modell: {model_info['display_name']} ({embedding_model})")
        self.logger.info(f"Typ: {model_info['type']}")
        self.logger.info(f"Modell-Größe: {model_info['model_size']}")
        self.logger.info(f"Dimensionen: {model_info['dimensions']}")
        self.logger.info(f"Max. Token: {model_info['max_tokens']}")
        self.logger.info(f"Deutsche Qualität: {model_info['german_quality']}")
        self.logger.info(f"Similarity-Threshold: {similarity_threshold}")
        self.logger.info(f"Beschreibung: {model_info['description']}")

# Create a singleton instance
config = ConfigManager()