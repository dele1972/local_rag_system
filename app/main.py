# app/main.py
from app.ui import start_ui
from app.config import config
import argparse

def main():
    parser = argparse.ArgumentParser(description='Lokales RAG mit Ollama und LangChain')
    parser.add_argument('--path', type=str, help='Pfad zur Dokumentenbasis')
    parser.add_argument('--model', type=str, choices=config.get_available_models(), 
                        help='Ollama-Modell für RAG')
    args = parser.parse_args()
    
    # Update config with command line arguments if provided
    if args.path:
        config.documents_path = args.path
    
    start_ui()

if __name__ == "__main__":
    main()