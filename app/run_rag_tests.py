#!/usr/bin/env python3
# run_rag_tests.py
"""
RAG System Test Runner
======================

Reparierte und optimierte Version des Test-Runners für das RAG-System.
Speziell für große deutsche Dokumente optimiert.

Verwendung:
    python run_rag_tests.py --documents path/to/docs --output results.json
    python run_rag_tests.py --quick-test path/to/single_doc.pdf
    python run_rag_tests.py --benchmark --models llama3.2,phi4-mini-reasoning:3.8b
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

# Lokale Imports mit fallback für optionale Module
try:
    from app.config import config
    from app.connection_utils import check_ollama_connection_with_retry
    print("✅ Core RAG-Module geladen")
except ImportError as e:
    print(f"❌ Core Import-Fehler: {e}")
    sys.exit(1)

# Test-Module mit Fallback laden
RAGTestFramework = None
TestConfiguration = None
quick_document_analysis = None
estimate_test_duration = None
PerformanceMonitor = None
TestResultVisualizer = None
BatchTestRunner = None
ReportGenerator = None

try:
    from app.test_rag import RAGTestFramework, TestConfiguration
    print("✅ Test-Framework geladen")
except ImportError as e:
    print(f"⚠️ Test-Framework nicht verfügbar: {e}")

try:
    from app.test_utils import quick_document_analysis, estimate_test_duration
    print("✅ Basic Test-Utils geladen")
except ImportError as e:
    print(f"⚠️ Basic Test-Utils nicht verfügbar: {e}")

# Optionale Visualisierungs-Module
try:
    from app.test_utils import PerformanceMonitor, TestResultVisualizer, BatchTestRunner, ReportGenerator
    print("✅ Erweiterte Test-Utils geladen")
except ImportError as e:
    print(f"⚠️ Erweiterte Test-Utils nicht verfügbar (Visualisierung deaktiviert): {e}")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_tests.log')
    ]
)
logger = logging.getLogger(__name__)

# Test-Konfigurationen für deutsche Dokumente
DEFAULT_MODELS = [
    "llama3.2",                    # Hauptmodell - 2.0GB, 8192 Token
    "phi4-mini-reasoning:3.8b",    # Reasoning - 3.2GB, 4096 Token  
    "mistral",              # Backup - 4.1GB, 4096 Token
    "deepseek-r1:8b"              # Komplex - 5.2GB, 8192 Token
]

DEFAULT_CHUNK_SIZES = [500, 750, 1000, 1500, 2000]  # Optimiert für deutsche Texte
DEFAULT_OVERLAPS = [50, 100, 150, 200]  # Deutsche Sätze sind länger

QUICK_TEST_CONFIG = {
    "models": ["llama3.2"],
    "chunk_sizes": [750, 1000],
    "overlaps": [100],
    "questions_per_test": 3
}

BENCHMARK_CONFIG = {
    "models": DEFAULT_MODELS,
    "chunk_sizes": [500, 1000, 1500, 2000],
    "overlaps": [100, 200],
    "questions_per_test": 5,
    "include_performance_monitoring": True
}


def setup_test_environment() -> bool:
    """
    Bereitet die Test-Umgebung vor und prüft alle Abhängigkeiten.
    
    Returns:
        bool: True wenn alles bereit ist
    """
    logger.info("🔧 Bereite Test-Umgebung vor...")
    
    # 1. Ollama-Verbindung prüfen
    if not check_ollama_connection_with_retry():
        logger.error("❌ Ollama-Verbindung fehlgeschlagen")
        return False
    
    # 2. Verfügbare Modelle prüfen
    try:
        available_models = config.get_available_models()
        logger.info(f"📊 Verfügbare Modelle: {available_models}")
        
        missing_models = [m for m in DEFAULT_MODELS if m not in available_models]
        if missing_models:
            logger.warning(f"⚠️ Fehlende Modelle: {missing_models}")
    except Exception as e:
        logger.warning(f"⚠️ Konnte Modelle nicht prüfen: {e}")
    
    # 3. Output-Verzeichnisse erstellen
    Path("test_results").mkdir(exist_ok=True)
    Path("test_reports").mkdir(exist_ok=True)
    Path("test_visualizations").mkdir(exist_ok=True)
    
    logger.info("✅ Test-Umgebung bereit")
    return True


def run_document_analysis(document_paths: List[str]) -> Dict[str, Any]:
    """
    Führt Vor-Analyse der Dokumente durch.
    
    Args:
        document_paths: Liste der zu analysierenden Dokumente
        
    Returns:
        Dict mit Analyse-Ergebnissen
    """
    logger.info("🔍 Führe Dokument-Analyse durch...")
    
    analysis_results = {}
    total_size = 0

    logger.debug(f"[DEBUG] Analysiert werden (run_rag_tests.py:run_document_analysis()): {document_paths}")
    
    for doc_path in document_paths:
        logger.info(f"[TEST] quick_document_analysis vorhanden? {'JA' if quick_document_analysis else 'NEIN'}")
        try:
            if quick_document_analysis:
                try:
                    analysis = quick_document_analysis(doc_path)
                    logger.debug(f"🔍 Analyse-Ergebnis: {analysis}")
                except Exception as qe:
                    logger.exception(f"[ERROR] quick_document_analysis() fehlgeschlagen mit Exception:")
                    import traceback
                    traceback.print_exc()
                    analysis = {
                        'file_size_mb': 0.0,
                        'estimated_tokens': 0,
                        'language': 'unbekannt',
                        'analysis_method': 'error_in_quick_analysis',
                        'error': str(qe)
                    }
            else:
                # Einfache Fallback-Analyse
                file_path = Path(doc_path)
                file_size_mb = file_path.stat().st_size / (1024*1024)
                analysis = {
                    'file_size_mb': file_size_mb,
                    'estimated_tokens': int(file_size_mb * 1000 * 0.75),  # Grosse Schätzung
                    'language': 'deutsch (geschätzt)',
                    'analysis_method': 'fallback'
                }

            # Extra flachere Kopie für Anzeigezwecke
            flat = {}
            if "metrics" in analysis:
                metrics = analysis["metrics"]
                flat = {
                    "file_size_mb": getattr(metrics, "file_size_mb", 0),
                    "estimated_tokens": getattr(metrics, "word_count", 0),
                    "language": getattr(metrics, "language_detected", "unbekannt")
                }

            analysis_results[doc_path] = {**analysis, **flat}

            total_size += analysis.get('file_size_mb', 0)

            if "metrics" in analysis:
                m = analysis["metrics"]
                logger.info(f"📄 {Path(doc_path).name}: "
                            f"{getattr(m, 'file_size_mb', 0):.1f}MB, "
                            f"~{getattr(m, 'word_count', 0):,} Tokens "
                            f"(Sprache: {getattr(m, 'language_detected', 'unbekannt')})")
            else:
                logger.warning(f"📄 {Path(doc_path).name}: Keine Analyse-Metriken verfügbar.")

        except Exception as e:
            logger.error(f"❌ Fehler bei Analyse von {doc_path}: {e}")
            analysis_results[doc_path] = {"error": str(e)}

    # Geschätzte Testdauer berechnen
    total_tokens = 0
    for doc_path, result in analysis_results.items():
        if doc_path.startswith("_") or "error" in result:
            continue
        total_tokens += result.get("estimated_tokens", 0)

    # Konfig: 3 Fragen pro Modell, ca. 1.5s pro 1000 Token + Rechenzeit
    models = DEFAULT_MODELS
    questions_per_model = 3
    tokens_pro_test = total_tokens
    test_anzahl = len(models) * questions_per_model
    minuten_schätzung = max(1, int((tokens_pro_test / 1000 * 1.5 * test_anzahl) / 60))  # ganz grob

    analysis_results["_duration_estimate"] = {
        "total_minutes": minuten_schätzung,
        "method": "token_based_estimate"
    }
    logger.info(f"⏱️ Geschätzte Testdauer: {minuten_schätzung} Minuten")

    analysis_results['_summary'] = {
        'total_documents': len(document_paths),
        'total_size_mb': total_size,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    return analysis_results


def run_quick_test(document_path: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Führt einen schnellen Test für ein einzelnes Dokument durch.
    
    Args:
        document_path: Pfad zum Dokument
        output_file: Optional - Ausgabedatei für Ergebnisse
        
    Returns:
        Dict mit Test-Ergebnissen
    """
    logger.info(f"🚀 Starte Quick-Test für: {document_path}")
    
    # Prüfe ob Test-Framework verfügbar ist
    if not RAGTestFramework:
        logger.error("❌ Test-Framework nicht verfügbar - test_rag.py nicht geladen")
        return {"error": "Test-Framework nicht verfügbar"}
    
    # Dokument-Analyse
    if quick_document_analysis:
        doc_analysis = quick_document_analysis(document_path)
    else:
        # Fallback-Analyse
        file_path = Path(document_path)
        file_size_mb = file_path.stat().st_size / (1024*1024)
        doc_analysis = {
            'file_size_mb': file_size_mb,
            'estimated_tokens': int(file_size_mb * 1000 * 0.75),
            'language': 'deutsch (geschätzt)',
            'analysis_method': 'fallback'
        }
    
    logger.info(f"📊 Dokument: {doc_analysis.get('file_size_mb', 0):.1f}MB, "
               f"Sprache: {doc_analysis.get('language', 'unbekannt')}")
    
    # Test-Framework initialisieren
    test_framework = RAGTestFramework()
    
    # Performance-Monitoring (optional)
    monitor = None
    if PerformanceMonitor:
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
    
    try:
        # Quick-Test durchführen
        results = test_framework.run_quick_comparison(
            document_path, 
            models=QUICK_TEST_CONFIG["models"]
        )
        
        # Performance-Daten hinzufügen (wenn verfügbar)
        if monitor:
            performance_data = monitor.stop_monitoring()
            results['performance_monitoring'] = performance_data
        
        results['document_analysis'] = doc_analysis
        results['test_config'] = QUICK_TEST_CONFIG
        
        # Ergebnisse speichern
        if output_file:
            output_path = Path(output_file)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"test_results/quick_test_{timestamp}.json")
        
        # Verzeichnis erstellen falls nicht vorhanden
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Ergebnisse gespeichert: {output_path}")
        
        # Kurze Zusammenfassung
        if 'results' in results:
            best_result = min(results['results'], 
                            key=lambda x: x.get('response_time', float('inf')))
            logger.info(f"🏆 Bestes Ergebnis: {best_result.get('model', 'unbekannt')} "
                       f"({best_result.get('response_time', 0):.1f}s)")
        
        return results
        
    except Exception as e:
        if monitor:
            monitor.stop_monitoring()
        logger.error(f"❌ Quick-Test fehlgeschlagen: {e}")
        return {"error": str(e), "document_analysis": doc_analysis}


def run_full_benchmark(document_paths: List[str], output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Führt vollständigen Benchmark durch.
    
    Args:
        document_paths: Liste der zu testenden Dokumente
        output_file: Optional - Ausgabedatei für Ergebnisse
        
    Returns:
        Dict mit Benchmark-Ergebnissen
    """
    logger.info(f"🏁 Starte vollständigen Benchmark für {len(document_paths)} Dokumente")
    
    # Dokument-Analyse
    doc_analysis = run_document_analysis(document_paths)
    
    # Test-Framework und Tools initialisieren
    test_framework = RAGTestFramework()
    batch_runner = BatchTestRunner(max_workers=2)  # Konservativ für Speicher
    monitor = PerformanceMonitor()
    
    monitor.start_monitoring()
    
    try:
        # Test-Konfigurationen erstellen
        test_configs = test_framework.create_test_configurations(document_paths)
        logger.info(f"📋 Erstellt {len(test_configs)} Test-Konfigurationen")
        
        # Parallel ausführen
        results = batch_runner.run_parallel_tests(test_framework, test_configs)
        
        # Performance-Daten
        performance_data = monitor.stop_monitoring()
        
        # Ergebnisse zusammenfassen
        benchmark_results = {
            'results': results,
            'document_analysis': doc_analysis,
            'performance_monitoring': performance_data,
            'benchmark_config': BENCHMARK_CONFIG,
            'test_summary': {
                'total_tests': len(test_configs),
                'completed_tests': len([r for r in results if not r.get('error')]),
                'failed_tests': len([r for r in results if r.get('error')]),
                'total_duration_minutes': performance_data.get('total_duration_minutes', 0),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Speichern
        if output_file:
            output_path = Path(output_file)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"test_results/benchmark_{timestamp}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Benchmark-Ergebnisse gespeichert: {output_path}")
        
        # Visualisierungen erstellen
        try:
            visualizer = TestResultVisualizer()
            dashboard_path = visualizer.create_performance_dashboard(str(output_path))
            comparison_path = visualizer.create_model_comparison_chart(str(output_path))
            
            logger.info(f"📊 Dashboard erstellt: {dashboard_path}")
            logger.info(f"📈 Vergleichsdiagramm erstellt: {comparison_path}")
        except Exception as e:
            logger.warning(f"⚠️ Konnte Visualisierungen nicht erstellen: {e}")
        
        # HTML-Report generieren
        try:
            report_gen = ReportGenerator()
            report_path = report_gen.generate_html_report(str(output_path))
            logger.info(f"📄 HTML-Bericht erstellt: {report_path}")
        except Exception as e:
            logger.warning(f"⚠️ Konnte HTML-Bericht nicht erstellen: {e}")
        
        return benchmark_results
        
    except Exception as e:
        monitor.stop_monitoring()
        logger.error(f"❌ Benchmark fehlgeschlagen: {e}")
        return {"error": str(e), "document_analysis": doc_analysis}


def main():
    """Hauptfunktion für CLI-Nutzung"""
    parser = argparse.ArgumentParser(
        description="RAG System Test Runner - Optimiert für deutsche Dokumente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python run_rag_tests.py --documents documents/ --output results.json
  python run_rag_tests.py --quick-test documents/large_doc.pdf
  python run_rag_tests.py --benchmark --models llama3.2,phi4-mini-reasoning:3.8b
  python run_rag_tests.py --analyze documents/
        """
    )
    
    # Hauptoptionen
    parser.add_argument('--documents', '-d', type=str,
                       help='Pfad zu Dokumenten (Datei oder Verzeichnis)')
    parser.add_argument('--output', '-o', type=str,
                       help='Ausgabedatei für Ergebnisse')
    
    # Test-Modi
    parser.add_argument('--quick-test', type=str,
                       help='Schneller Test für ein einzelnes Dokument')
    parser.add_argument('--benchmark', action='store_true',
                       help='Vollständiger Benchmark-Test')
    parser.add_argument('--analyze', type=str,
                       help='Nur Dokument-Analyse durchführen')
    
    # Konfiguration
    parser.add_argument('--models', type=str,
                       help='Komma-getrennte Liste der zu testenden Modelle')
    parser.add_argument('--chunk-sizes', type=str,
                       help='Komma-getrennte Liste der Chunk-Größen')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Ausführliche Ausgabe')
    
    args = parser.parse_args()
    
    # Logging-Level setzen
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Test-Umgebung vorbereiten
    if not setup_test_environment():
        sys.exit(1)
    
    try:
        # Nur Analyse
        if args.analyze:
            doc_paths = []
            analyze_path = Path(args.analyze)
            
            if analyze_path.is_file():
                doc_paths = [str(analyze_path)]
            elif analyze_path.is_dir():
                supported_types = config.get_supported_file_types()
                doc_paths = []
                for ext in supported_types:
                    doc_paths.extend(str(p) for p in analyze_path.rglob(f'*{ext}'))
            
            if not doc_paths:
                logger.error("❌ Keine Dokumente gefunden")
                sys.exit(1)
            
            analysis = run_document_analysis(doc_paths)
            
            # Analyse-Ergebnisse ausgeben
            print("\n📊 DOKUMENT-ANALYSE")
            print("=" * 50)
            for doc_path, data in analysis.items():
                if doc_path.startswith('_'):
                    continue
                print(f"\n📄 {Path(doc_path).name}")
                if 'error' in data:
                    print(f"   ❌ Fehler: {data['error']}")
                else:
                    print(f"   📏 Größe: {data.get('file_size_mb', 0):.1f} MB")
                    print(f"   📝 Tokens: ~{data.get('estimated_tokens', 0):,}")
                    print(f"   🗣️ Sprache: {data.get('language', 'unbekannt')}")
                    if data.get('analysis_method') == 'fallback':
                        print(f"   ⚠️ Einfache Analyse (test_utils.py nicht verfügbar)")
            
            if '_duration_estimate' in analysis:
                est = analysis['_duration_estimate']
                print(f"\n⏱️ Geschätzte Testdauer: {est.get('total_minutes', 'unbekannt')} Minuten")
                if est.get('method') == 'fallback_estimate':
                    print("   ⚠️ Grobe Schätzung (erweiterte Utils nicht verfügbar)")
            
            return
        
        # Quick Test
        if args.quick_test:
            if not Path(args.quick_test).exists():
                logger.error(f"❌ Dokument nicht gefunden: {args.quick_test}")
                sys.exit(1)
            
            results = run_quick_test(args.quick_test, args.output)
            
            if 'error' not in results:
                print("\n🚀 QUICK-TEST ABGESCHLOSSEN")
                print("=" * 30)
                if 'results' in results:
                    for result in results['results']:
                        print(f"📊 {result.get('model', 'unbekannt')}: "
                             f"{result.get('response_time', 0):.1f}s")
            return
        
        # Vollständiger Benchmark
        if args.benchmark or args.documents:
            doc_paths = []
            
            if args.documents:
                docs_path = Path(args.documents)
                if docs_path.is_file():
                    doc_paths = [str(docs_path)]
                elif docs_path.is_dir():
                    supported_types = config.get_supported_file_types()
                    doc_paths = []
                    for ext in supported_types:
                        doc_paths.extend(str(p) for p in docs_path.rglob(f'*{ext}'))

            if not doc_paths:
                logger.error("❌ Keine Dokumente gefunden")
                sys.exit(1)
            
            results = run_full_benchmark(doc_paths, args.output)
            
            if 'error' not in results:
                print("\n🏁 BENCHMARK ABGESCHLOSSEN")
                print("=" * 30)
                summary = results.get('test_summary', {})
                print(f"✅ Tests abgeschlossen: {summary.get('completed_tests', 0)}")
                print(f"❌ Tests fehlgeschlagen: {summary.get('failed_tests', 0)}")
                print(f"⏱️ Gesamtdauer: {summary.get('total_duration_minutes', 0):.1f} Minuten")
            return
        
        # Keine gültige Option
        parser.print_help()
        
    except KeyboardInterrupt:
        logger.info("🛑 Tests durch Benutzer abgebrochen")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler: {e}")
        logger.debug("Traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()