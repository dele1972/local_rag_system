# config.py
import os
from pathlib import Path

class ConfigManager:
    def __init__(self):
        self.base_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
        self.documents_path = os.path.join(self.base_path, "documents")
        self.available_models = ["llama3.2", "mistral", "deepseek-r1"]
        
        # Detect if running in Docker
        self.in_docker = os.path.exists("/.dockerenv")
        
        # Set the Ollama base URL based on environment
        if self.in_docker:
            self.ollama_base_url = "http://host.docker.internal:11434"
        else:
            self.ollama_base_url = "http://localhost:11434"
    
    def get_documents_path(self):
        return self.documents_path
    
    def get_available_models(self):
        return self.available_models
    
    def get_ollama_base_url(self):
        return self.ollama_base_url

# Create a singleton instance
config = ConfigManager()