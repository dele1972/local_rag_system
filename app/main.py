# app/main.py - Erweitert für RAG-Debugging und Performance-Analyse
from app.ui import start_ui
from app.config import config
from app.rag import (
    debug_retrieval_quality, 
    suggest_optimal_chain_type, 
    get_chain_type_description,
    DocumentRetrievalDebugger,
    ChunkAnalyzer,
    PerformanceProfiler
)
from app.vectorstore import get_vectorstore
# from app.loader import load_documents
from app.vectorstore import load_documents
import argparse
import sys
import os
from datetime import datetime
import json

# Logger aus Config verwenden
logger = config.get_logger("MAIN")

def setup_debug_logging():
    """Aktiviert erweiterte Debug-Ausgaben"""
    # Debugging für alle wichtigen Module aktivieren
    config.set_log_level("DEBUG")
    
    logger.info("🔍 ERWEITERTE DEBUGGING-FUNKTIONEN AKTIVIERT")
    logger.info("=" * 60)
    logger.info(f"Session gestartet: {datetime.now().isoformat()}")
    logger.info(f"System: Windows 11, 32GB RAM, Ollama 0.9.0")
    logger.info(f"Haupt-Probleme: Große deutsche Dokumente (6.5MB)")
    logger.info("=" * 60)

def analyze_document_collection(documents_path):
    """
    Führt eine umfassende Analyse der Dokumentensammlung durch
    
    Args:
        documents_path: Pfad zur Dokumentenbasis
        
    Returns:
        Dict mit Analyse-Ergebnissen
    """
    logger.info(f"🔍 DOKUMENT-ANALYSE: {documents_path}")
    
    if not os.path.exists(documents_path):
        logger.error(f"Dokumentenpfad nicht gefunden: {documents_path}")
        return {"error": "Path not found"}
    
    analysis = {
        "path": documents_path,
        "timestamp": datetime.now().isoformat(),
        "files": [],
        "total_size_mb": 0,
        "large_files": [],
        "recommendations": []
    }
    
    try:
        # Dateien analysieren
        for root, dirs, files in os.walk(documents_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    file_info = {
                        "name": file,
                        "path": file_path,
                        "size_mb": round(file_size, 2),
                        "extension": os.path.splitext(file)[1].lower()
                    }
                    analysis["files"].append(file_info)
                    analysis["total_size_mb"] += file_size
                    
                    # Große Dateien identifizieren (>5MB)
                    if file_size > 5:
                        analysis["large_files"].append(file_info)
                        logger.warning(f"📄 GROSSE DATEI: {file} ({file_size:.1f}MB)")
                        
                except OSError as e:
                    logger.warning(f"Kann Datei nicht analysieren: {file} - {str(e)}")
        
        # Statistiken
        file_count = len(analysis["files"])
        avg_size = analysis["total_size_mb"] / file_count if file_count > 0 else 0
        large_file_count = len(analysis["large_files"])
        
        logger.info(f"📊 SAMMLUNG-STATISTIKEN:")
        logger.info(f"   Dateien: {file_count}")
        logger.info(f"   Gesamtgröße: {analysis['total_size_mb']:.1f}MB")
        logger.info(f"   Durchschnittsgröße: {avg_size:.2f}MB")
        logger.info(f"   Große Dateien (>5MB): {large_file_count}")
        
        # Empfehlungen generieren
        if large_file_count > 0:
            analysis["recommendations"].append(f"🚨 {large_file_count} große Dateien detected - map_reduce Chain empfohlen")
            analysis["recommendations"].append("💡 Feineres Chunking für große deutsche Dokumente konfigurieren")
        
        if analysis["total_size_mb"] > 50:
            analysis["recommendations"].append("⚠️ Sehr große Dokumentenbasis - Performance-Optimierung erforderlich")
            analysis["recommendations"].append("💡 Erwägen Sie Batch-Processing oder Index-Caching")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Dokument-Analyse fehlgeschlagen: {str(e)}")
        return {"error": str(e)}

def run_comprehensive_system_analysis(documents_path, model_name):
    """
    Führt eine umfassende System-Analyse durch
    
    Args:
        documents_path: Pfad zu den Dokumenten
        model_name: Zu analysierendes Modell
    """
    logger.info("🔬 STARTE UMFASSENDE SYSTEM-ANALYSE")
    logger.info("=" * 80)
    
    analysis_results = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "model": model_name,
            "documents_path": documents_path,
            "ollama_url": config.get_ollama_base_url()
        }
    }
    
    try:
        # 1. Dokument-Analyse
        logger.info("📄 PHASE 1: DOKUMENT-ANALYSE")
        doc_analysis = analyze_document_collection(documents_path)
        analysis_results["document_analysis"] = doc_analysis
        
        # 2. Dokumente laden für weitere Analyse
        logger.info("📚 PHASE 2: DOKUMENTE LADEN")
        try:
            documents = load_documents(documents_path)
            logger.info(f"✅ {len(documents)} Dokumente geladen")
            analysis_results["loaded_documents"] = len(documents)
        except Exception as e:
            logger.error(f"❌ Dokumenten-Laden fehlgeschlagen: {str(e)}")
            analysis_results["document_loading_error"] = str(e)
            return analysis_results
        
        # 3. Vektorspeicher erstellen
        logger.info("🔍 PHASE 3: VEKTORSPEICHER-ERSTELLUNG")
        try:
            vectorstore = get_vectorstore(documents)
            logger.info("✅ Vektorspeicher erfolgreich erstellt")
        except Exception as e:
            logger.error(f"❌ Vektorspeicher-Erstellung fehlgeschlagen: {str(e)}")
            analysis_results["vectorstore_error"] = str(e)
            return analysis_results
        
        # 4. Chunk-Analyse
        logger.info("🧩 PHASE 4: CHUNK-ANALYSE")
        chunk_analyzer = ChunkAnalyzer()
        chunk_analysis = chunk_analyzer.analyze_chunks(vectorstore, sample_size=100)
        analysis_results["chunk_analysis"] = chunk_analysis
        
        # Chunk-Ergebnisse loggen
        chunk_count = chunk_analysis.get("chunk_count", 0)
        avg_length = chunk_analysis.get("length_stats", {}).get("mean", 0)
        logger.info(f"📊 Chunk-Statistiken: {chunk_count} Chunks, ⌀ {avg_length:.0f} Zeichen")
        
        for rec in chunk_analysis.get("recommendations", [])[:3]:
            logger.warning(f"💡 CHUNK-EMPFEHLUNG: {rec}")
        
        # 5. Chain-Typ-Empfehlung
        logger.info("⛓️ PHASE 5: CHAIN-TYP-ANALYSE")
        recommended_chain, explanation, confidence = suggest_optimal_chain_type(
            model_name, 
            document_count=chunk_count,
            file_size_mb=doc_analysis.get("total_size_mb", 0)
        )
        
        analysis_results["chain_recommendation"] = {
            "recommended": recommended_chain,
            "explanation": explanation,
            "confidence": confidence
        }
        
        logger.info(f"🎯 EMPFOHLENER CHAIN-TYP: {recommended_chain} (Konfidenz: {confidence:.1%})")
        logger.info(explanation)
        
        # 6. Retrieval-Qualitäts-Test (falls möglich)
        logger.info("🎯 PHASE 6: RETRIEVAL-QUALITÄTS-TEST")
        test_questions = [
            "Was ist das Hauptthema des Dokuments?",
            "Welche wichtigen Punkte werden erwähnt?",
            "Gibt es spezifische Details oder Zahlen?",
            "Wer sind die beteiligten Personen oder Organisationen?",
            "Welche Schlussfolgerungen werden gezogen?"
        ]
        
        try:
            retrieval_analysis = debug_retrieval_quality(vectorstore, test_questions, k_values=[1, 3, 5, 10, 15, 20])
            analysis_results["retrieval_analysis"] = retrieval_analysis
            
            # Retrieval-Ergebnisse loggen
            avg_similarity = retrieval_analysis.get("aggregate_analysis", {}).get("avg_similarity_across_questions", 0)
            optimal_k = retrieval_analysis.get("optimal_k", "N/A")
            
            logger.info(f"📈 Retrieval-Qualität: ⌀ Similarity {avg_similarity:.3f}, Optimal k={optimal_k}")
            
            for rec in retrieval_analysis.get("recommendations", [])[:3]:
                logger.warning(f"🔍 RETRIEVAL-EMPFEHLUNG: {rec}")
                
        except Exception as e:
            logger.warning(f"Retrieval-Test fehlgeschlagen: {str(e)}")
            analysis_results["retrieval_test_error"] = str(e)
        
        # 7. Zusammenfassung und kritische Warnungen
        logger.info("📋 PHASE 7: KRITISCHE ANALYSE")
        
        critical_issues = []
        warnings = []
        recommendations = []
        
        # Große Dateien
        large_files = doc_analysis.get("large_files", [])
        if large_files:
            critical_issues.append(f"🚨 {len(large_files)} große Dateien (>5MB) detected")
            recommendations.append("💡 Zwingend map_reduce Chain verwenden")
            recommendations.append("💡 Chunk-Größe für deutsche Texte optimieren")
        
        # Token-Limits
        model_info = config.get_model_info(model_name)
        token_limit = model_info.get("token_limit", 4096)
        if chunk_count * 500 > token_limit:  # Grobe Schätzung
            critical_issues.append(f"⚠️ Zu viele Chunks für Token-Limit ({token_limit:,})")
            recommendations.append("💡 Retrieval-k reduzieren oder map_reduce verwenden")
        
        # Deutsche Sprache
        german_score = chunk_analysis.get("german_language_score", 0)
        if german_score < 0.05:
            warnings.append("🇩🇪 Niedrige deutsche Sprach-Indikatoren")
            recommendations.append("💡 Deutsche Embedding-Modelle erwägen")
        
        # Retrieval-Qualität
        if "retrieval_analysis" in analysis_results:
            avg_sim = analysis_results["retrieval_analysis"].get("aggregate_analysis", {}).get("avg_similarity_across_questions", 0)
            if avg_sim < 0.5:
                critical_issues.append("🚨 Sehr niedrige Retrieval-Qualität")
                recommendations.append("💡 Embedding-Modell wechseln oder Chunking verbessern")
        
        # Finale Zusammenfassung
        analysis_results["final_assessment"] = {
            "critical_issues": critical_issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "overall_severity": "HIGH" if critical_issues else ("MEDIUM" if warnings else "LOW")
        }
        
        # Finale Ausgabe
        logger.info("=" * 80)
        logger.info("🎯 FINALE SYSTEM-BEWERTUNG")
        logger.info("=" * 80)
        
        severity = analysis_results["final_assessment"]["overall_severity"]
        logger.info(f"Gesamt-Schweregrad: {severity}")
        
        if critical_issues:
            logger.error("🚨 KRITISCHE PROBLEME:")
            for issue in critical_issues:
                logger.error(f"   {issue}")
        
        if warnings:
            logger.warning("⚠️ WARNUNGEN:")
            for warning in warnings:
                logger.warning(f"   {warning}")
        
        if recommendations:
            logger.info("💡 EMPFEHLUNGEN:")
            for rec in recommendations[:5]:  # Top 5
                logger.info(f"   {rec}")
        
        logger.info("=" * 80)
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"System-Analyse fehlgeschlagen: {str(e)}")
        analysis_results["system_analysis_error"] = str(e)
        return analysis_results

def save_analysis_report(analysis_results, output_file=None):
    """
    Speichert den Analyse-Bericht in eine JSON-Datei
    
    Args:
        analysis_results: Analyse-Ergebnisse
        output_file: Ausgabe-Datei (optional)
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"rag_analysis_report_{timestamp}.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 Analyse-Bericht gespeichert: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Konnte Analyse-Bericht nicht speichern: {str(e)}")
        return None

def interactive_debugging_session():
    """Startet eine interaktive Debugging-Session"""
    logger.info("🔧 INTERAKTIVE DEBUGGING-SESSION")
    print("\n" + "="*60)
    print("🔧 RAG-SYSTEM INTERAKTIVE DEBUGGING-SESSION")
    print("="*60)
    
    print("""
Verfügbare Debugging-Optionen:
1. 📄 Dokument-Sammlung analysieren
2. 🔍 Vollständige System-Analyse
3. ⛓️ Chain-Typ-Empfehlungen anzeigen
4. 🎯 Retrieval-Qualitäts-Test
5. 📊 Aktuelle Konfiguration anzeigen
6. 🚀 Standard-UI starten
7. ❌ Beenden
""")
    
    while True:
        try:
            choice = input("\nWählen Sie eine Option (1-7): ").strip()
            
            if choice == "1":
                path = input("Pfad zur Dokumentenbasis: ").strip()
                if path:
                    analyze_document_collection(path)
                else:
                    path = config.documents_path
                    if path:
                        analyze_document_collection(path)
                    else:
                        print("❌ Kein Pfad angegeben")
            
            elif choice == "2":
                path = input("Pfad zur Dokumentenbasis (oder Enter für Standard): ").strip()
                if not path:
                    path = config.documents_path
                
                model = input("Modell-Name (oder Enter für Standard): ").strip()
                if not model:
                    available_models = config.get_available_models()
                    if available_models:
                        model = available_models[0]
                    else:
                        model = "llama3.2"
                
                if path:
                    analysis = run_comprehensive_system_analysis(path, model)
                    
                    # Frage nach Speicherung
                    save_choice = input("\nAnalyse-Bericht speichern? (j/n): ").strip().lower()
                    if save_choice in ['j', 'ja', 'y', 'yes']:
                        report_file = save_analysis_report(analysis)
                        if report_file:
                            print(f"✅ Bericht gespeichert: {report_file}")
                else:
                    print("❌ Kein Pfad angegeben")
            
            elif choice == "3":
                model = input("Modell-Name (oder Enter für llama3.2): ").strip()
                if not model:
                    model = "llama3.2"
                
                doc_count = input("Anzahl Dokumente (optional): ").strip()
                file_size = input("Dateigröße in MB (optional): ").strip()
                
                doc_count = int(doc_count) if doc_count.isdigit() else None
                file_size = float(file_size) if file_size.replace('.', '').isdigit() else None
                
                recommended, explanation, confidence = suggest_optimal_chain_type(
                    model, doc_count, file_size
                )
                
                print(f"\n🎯 EMPFEHLUNG: {recommended}")
                print(explanation)
                
                # Beschreibungen aller Chain-Typen
                print("\n📚 CHAIN-TYP-BESCHREIBUNGEN:")
                for chain_type in ["stuff", "map_reduce", "refine", "map_rerank"]:
                    print(f"\n{chain_type.upper()}:")
                    print(get_chain_type_description(chain_type))
            
            elif choice == "4":
                print("🎯 Retrieval-Test benötigt geladenen Vektorspeicher...")
                print("💡 Verwenden Sie Option 2 für vollständige Analyse")
            
            elif choice == "5":
                print("\n📊 AKTUELLE KONFIGURATION:")
                print(f"Ollama URL: {config.get_ollama_base_url()}")
                print(f"Dokumente Pfad: {config.documents_path}")
                print(f"Verfügbare Modelle: {', '.join(config.get_available_models())}")
                
                # Zeige Modell-Informationen
                for model in ["llama3.2", "phi4-mini-reasoning:3.8b", "mistral:latest", "deepseek-r1:8b"]:
                    try:
                        model_info = config.get_model_info(model)
                        print(f"\n{model}:")
                        print(f"  Token-Limit: {model_info['token_limit']:,}")
                        print(f"  Empfohlene Chain: {model_info['recommended_chain']}")
                        print(f"  Optimal k: {config.get_optimal_retrieval_k(model)}")
                    except:
                        print(f"{model}: Nicht verfügbar")
            
            elif choice == "6":
                print("🚀 Starte Standard-UI...")
                break
            
            elif choice == "7":
                print("👋 Debugging-Session beendet")
                return False
            
            else:
                print("❌ Ungültige Auswahl. Bitte wählen Sie 1-7.")
                
        except KeyboardInterrupt:
            print("\n👋 Debugging-Session unterbrochen")
            return False
        except Exception as e:
            print(f"❌ Fehler: {str(e)}")
    
    return True  # UI starten

def main():
    parser = argparse.ArgumentParser(
        description='Lokales RAG-System mit erweiterten Debugging-Funktionen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DEBUGGING-MODI:
  --debug-interactive    Startet interaktive Debugging-Session
  --analyze-system      Vollständige System-Analyse
  --analyze-docs        Nur Dokument-Sammlung analysieren
  --chain-recommendation Zeigt Chain-Typ-Empfehlungen

BEISPIELE:
  python main.py --debug-interactive
  python main.py --analyze-system --path ./docs --model llama3.2
  python main.py --chain-recommendation --model phi4-mini-reasoning:3.8b
        """
    )
    
    # Bestehende Argumente
    parser.add_argument('--path', type=str, help='Pfad zur Dokumentenbasis')
    parser.add_argument('--model', type=str, choices=config.get_available_models(), 
                        help='Ollama-Modell für RAG')
    
    # Neue Debugging-Argumente
    parser.add_argument('--debug-interactive', action='store_true',
                        help='Startet interaktive Debugging-Session')
    parser.add_argument('--analyze-system', action='store_true',
                        help='Führt vollständige System-Analyse durch')
    parser.add_argument('--analyze-docs', action='store_true',
                        help='Analysiert nur die Dokument-Sammlung')
    parser.add_argument('--chain-recommendation', action='store_true',
                        help='Zeigt Chain-Typ-Empfehlungen für Modell')
    parser.add_argument('--save-report', type=str,
                        help='Speichert Analyse-Bericht in angegebene Datei')
    parser.add_argument('--verbose', action='store_true',
                        help='Aktiviert erweiterte Debug-Ausgaben')
    
    args = parser.parse_args()
    
    # Update config with command line arguments if provided
    if args.path:
        config.documents_path = args.path
    
    # Erweiterte Debug-Ausgaben aktivieren
    if args.verbose:
        setup_debug_logging()
    
    # Standard-Modell setzen falls nicht angegeben
    model_name = args.model if args.model else "llama3.2"
    
    try:
        # Debugging-Modi
        if args.debug_interactive:
            setup_debug_logging()
            should_start_ui = interactive_debugging_session()
            if not should_start_ui:
                return
        
        elif args.analyze_system:
            setup_debug_logging()
            if not config.documents_path:
                logger.error("❌ Kein Dokumentenpfad konfiguriert. Verwenden Sie --path")
                return
            
            analysis = run_comprehensive_system_analysis(config.documents_path, model_name)
            
            if args.save_report:
                save_analysis_report(analysis, args.save_report)
            else:
                # Frage interaktiv nach Speicherung
                try:
                    save_choice = input("\nAnalyse-Bericht speichern? (j/n): ").strip().lower()
                    if save_choice in ['j', 'ja', 'y', 'yes']:
                        report_file = save_analysis_report(analysis)
                        if report_file:
                            print(f"✅ Bericht gespeichert: {report_file}")
                except:
                    pass  # Non-interactive Umgebung
            
            return
        
        elif args.analyze_docs:
            setup_debug_logging()
            if not config.documents_path:
                logger.error("❌ Kein Dokumentenpfad konfiguriert. Verwenden Sie --path")
                return
            
            analyze_document_collection(config.documents_path)
            return
        
        elif args.chain_recommendation:
            print(f"\n⛓️ CHAIN-TYP-EMPFEHLUNG FÜR: {model_name}")
            print("=" * 50)
            
            recommended, explanation, confidence = suggest_optimal_chain_type(model_name)
            print(f"🎯 EMPFEHLUNG: {recommended} (Konfidenz: {confidence:.1%})")
            print(explanation)
            
            print(f"\n📚 BESCHREIBUNG VON '{recommended.upper()}':")
            print(get_chain_type_description(recommended))
            
            return
        
        # Standard-Verhalten: UI starten
        logger.info("🚀 Starte RAG-System UI...")
        start_ui()
        
    except KeyboardInterrupt:
        logger.info("👋 System durch Benutzer beendet")
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()