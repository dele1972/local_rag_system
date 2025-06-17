#!/usr/bin/env python3
"""
RAG System Test Framework
=========================

Systematische Tests für verschiedene Dokumentgrößen, Chunk-Größen und Modelle.
Speziell optimiert für deutsche Dokumente und große Dateien.

Autor: RAG System Optimizer
Datum: 2025-06-16
"""

import os
import sys
import json
import time
import statistics
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

# RAG System Imports
from app.config import config
from app.vectorstore import load_documents, get_vectorstore, build_vectorstore
from app.rag import build_qa_chain, suggest_optimal_chain_type, debug_retrieval_quality
from app.connection_utils import check_ollama_connection_with_retry

# Test-spezifische Konfiguration
logger = logging.getLogger(__name__)

@dataclass
class TestConfiguration:
    """Konfiguration für einen einzelnen Test"""
    name: str
    model_name: str
    chunk_size: int
    chunk_overlap: int
    chain_type: str
    retrieval_k: int
    search_k: int
    document_path: str
    test_questions: List[str]
    expected_keywords: List[str] = None
    timeout_seconds: int = 300

@dataclass
class TestResult:
    """Ergebnis eines einzelnen Tests"""
    config_name: str
    model_name: str
    chunk_size: int
    chunk_overlap: int
    chain_type: str
    retrieval_k: int
    search_k: int
    document_path: str
    document_size_mb: float
    
    # Performance Metriken
    total_runtime: float
    embedding_time: float
    retrieval_time: float
    generation_time: float
    
    # Qualität Metriken
    questions_answered: int
    questions_total: int
    avg_answer_length: float
    keyword_matches: int
    keyword_total: int
    
    # Technische Metriken
    chunks_created: int
    chunks_retrieved: int
    tokens_used: int
    memory_usage_mb: float
    
    # Fehler und Warnings
    errors: List[str]
    warnings: List[str]
    success: bool
    
    # Antworten für manuelle Bewertung
    qa_results: List[Dict[str, Any]]

class RAGTestFramework:
    """Hauptklasse für RAG-System Tests"""
    
    def __init__(self, results_dir: str = "test_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.logger = config.get_logger('RAG_TEST')
        
        # Standard Test-Konfigurationen
        self.default_models = [
            "llama3.2",              # Hauptmodell
            "phi4-mini-reasoning:3.8b",  # Reasoning
            "mistral:latest",        # Backup
            "deepseek-r1:8b"        # Komplexe Aufgaben
        ]
        
        # Deutsche Test-Fragen für verschiedene Dokumenttypen
        self.german_test_questions = {
            "general": [
                "Was ist das Hauptthema dieses Dokuments?",
                "Welche wichtigsten Punkte werden behandelt?",
                "Gibt es spezifische Zahlen oder Statistiken?",
                "Welche Schlussfolgerungen werden gezogen?",
                "Was sind die wichtigsten Empfehlungen?"
            ],
            "technical": [
                "Welche technischen Spezifikationen werden genannt?",
                "Gibt es Implementierungsdetails?",
                "Welche Anforderungen werden definiert?",
                "Welche Probleme und Lösungen werden beschrieben?",
                "Welche Standards oder Normen werden referenziert?"
            ],
            "legal": [
                "Welche rechtlichen Bestimmungen werden erwähnt?",
                "Gibt es Compliance-Anforderungen?",
                "Welche Verantwortlichkeiten werden definiert?",
                "Welche Fristen oder Termine sind wichtig?",
                "Welche Sanktionen oder Konsequenzen werden beschrieben?"
            ]
        }
        
    def create_test_configurations(self, document_paths: List[str]) -> List[TestConfiguration]:
        """Erstellt Test-Konfigurationen für verschiedene Szenarien"""
        configurations = []
        
        for doc_path in document_paths:
            doc_size_mb = self._get_file_size_mb(doc_path)
            
            # Basis-Konfigurationen für jedes Dokument
            base_configs = [
                # Kleine Chunks - gut für präzise Retrieval
                {
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "suffix": "small_chunks"
                },
                # Standard Chunks - ausgewogener Ansatz
                {
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "suffix": "standard_chunks"
                },
                # Große Chunks - mehr Kontext
                {
                    "chunk_size": 2000,
                    "chunk_overlap": 400,
                    "suffix": "large_chunks"
                },
                # Sehr große Chunks - für große Dokumente
                {
                    "chunk_size": 4000,
                    "chunk_overlap": 800,
                    "suffix": "xlarge_chunks"
                }
            ]
            
            # Für jede Basis-Konfiguration und jedes Modell
            for base_config in base_configs:
                for model in self.default_models:
                    # Chain-Type basierend auf Dokumentgröße
                    if doc_size_mb > 5.0:
                        chain_types = ["map_reduce", "refine"]
                    elif doc_size_mb > 2.0:
                        chain_types = ["stuff", "map_reduce"]
                    else:
                        chain_types = ["stuff"]
                    
                    for chain_type in chain_types:
                        # Retrieval-Parameter basierend auf Chunk-Größe
                        if base_config["chunk_size"] <= 500:
                            retrieval_k_values = [5, 10, 15]
                        elif base_config["chunk_size"] <= 1000:
                            retrieval_k_values = [3, 7, 12]
                        else:
                            retrieval_k_values = [2, 5, 8]
                        
                        for retrieval_k in retrieval_k_values:
                            search_k = min(retrieval_k * 3, 50)  # Search mehr als retrieve
                            
                            config_name = f"{Path(doc_path).stem}_{model.replace(':', '_')}_{base_config['suffix']}_{chain_type}_k{retrieval_k}"
                            
                            # Test-Fragen basierend auf Dokumenttyp auswählen
                            test_questions = self._select_test_questions(doc_path)
                            
                            configurations.append(TestConfiguration(
                                name=config_name,
                                model_name=model,
                                chunk_size=base_config["chunk_size"],
                                chunk_overlap=base_config["chunk_overlap"],
                                chain_type=chain_type,
                                retrieval_k=retrieval_k,
                                search_k=search_k,
                                document_path=doc_path,
                                test_questions=test_questions,
                                expected_keywords=self._extract_expected_keywords(doc_path),
                                timeout_seconds=600 if doc_size_mb > 5.0 else 300
                            ))
        
        return configurations
    
    def run_single_test(self, test_config: TestConfiguration) -> TestResult:
        """Führt einen einzelnen Test durch"""
        self.logger.info(f"Starte Test: {test_config.name}")
        start_time = time.time()
        
        # Initialisiere Ergebnis-Objekt
        result = TestResult(
            config_name=test_config.name,
            model_name=test_config.model_name,
            chunk_size=test_config.chunk_size,
            chunk_overlap=test_config.chunk_overlap,
            chain_type=test_config.chain_type,
            retrieval_k=test_config.retrieval_k,
            search_k=test_config.search_k,
            document_path=test_config.document_path,
            document_size_mb=self._get_file_size_mb(test_config.document_path),
            total_runtime=0.0,
            embedding_time=0.0,
            retrieval_time=0.0,
            generation_time=0.0,
            questions_answered=0,
            questions_total=len(test_config.test_questions),
            avg_answer_length=0.0,
            keyword_matches=0,
            keyword_total=len(test_config.expected_keywords or []),
            chunks_created=0,
            chunks_retrieved=0,
            tokens_used=0,
            memory_usage_mb=0.0,
            errors=[],
            warnings=[],
            success=False,
            qa_results=[]
        )
        
        try:
            # 1. Ollama-Verbindung prüfen
            if not check_ollama_connection_with_retry(test_config.model_name):
                result.errors.append(f"Ollama-Verbindung zu {test_config.model_name} fehlgeschlagen")
                return result
            
            # 2. Dokumente laden und chunken
            self.logger.info(f"Lade Dokumente: {test_config.document_path}")
            embedding_start = time.time()
            
            documents = load_documents(
                test_config.document_path,
                chunk_size=test_config.chunk_size,
                chunk_overlap=test_config.chunk_overlap
            )
            
            if not documents:
                result.errors.append("Keine Dokumente geladen")
                return result
            
            result.chunks_created = len(documents)
            
            # 3. Vectorstore erstellen
            vectorstore = get_vectorstore(
                documents,
                model_name="nomic-embed-text",  # Standard Embedding-Modell
                persist_directory=None,  # Kein Persistieren für Tests
                chunk_size=test_config.chunk_size,
                chunk_overlap=test_config.chunk_overlap
            )
            
            result.embedding_time = time.time() - embedding_start
            
            # 4. QA-Chain erstellen
            qa_chain = build_qa_chain(
                vectorstore,
                test_config.model_name,
                chain_type=test_config.chain_type,
                enable_debug=True
            )
            
            # 5. Test-Fragen durchführen
            qa_results = []
            answer_lengths = []
            successful_answers = 0
            
            for i, question in enumerate(test_config.test_questions):
                self.logger.info(f"Frage {i+1}/{len(test_config.test_questions)}: {question[:50]}...")
                
                try:
                    retrieval_start = time.time()
                    
                    # Antwort generieren
                    response = qa_chain.invoke({
                        "query": question,
                        "k": test_config.retrieval_k
                    })
                    
                    retrieval_time = time.time() - retrieval_start
                    result.retrieval_time += retrieval_time
                    
                    answer = response.get("result", "").strip()
                    source_docs = response.get("source_documents", [])
                    
                    if answer and len(answer) > 10:  # Mindestlänge für gültige Antwort
                        successful_answers += 1
                        answer_lengths.append(len(answer))
                        
                        # Keyword-Matching
                        keyword_matches = self._count_keyword_matches(
                            answer, test_config.expected_keywords or []
                        )
                        result.keyword_matches += keyword_matches
                    
                    qa_result = {
                        "question": question,
                        "answer": answer,
                        "answer_length": len(answer),
                        "retrieval_time": retrieval_time,
                        "source_documents_count": len(source_docs),
                        "keyword_matches": keyword_matches,
                        "source_snippets": [doc.page_content[:200] + "..." 
                                          for doc in source_docs[:3]]
                    }
                    qa_results.append(qa_result)
                    
                except Exception as e:
                    error_msg = f"Fehler bei Frage {i+1}: {str(e)}"
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
                    
                    qa_results.append({
                        "question": question,
                        "answer": "",
                        "error": str(e),
                        "retrieval_time": 0.0,
                        "source_documents_count": 0,
                        "keyword_matches": 0
                    })
            
            # 6. Ergebnisse zusammenfassen
            result.questions_answered = successful_answers
            result.avg_answer_length = statistics.mean(answer_lengths) if answer_lengths else 0.0
            result.qa_results = qa_results
            result.chunks_retrieved = sum(qa.get("source_documents_count", 0) for qa in qa_results)
            
            # 7. Performance-Metriken
            result.total_runtime = time.time() - start_time
            result.generation_time = result.total_runtime - result.embedding_time - result.retrieval_time
            
            # 8. Erfolg bewerten
            success_rate = successful_answers / len(test_config.test_questions)
            result.success = success_rate >= 0.7  # 70% der Fragen erfolgreich beantwortet
            
            if not result.success:
                result.warnings.append(f"Niedrige Erfolgsrate: {success_rate:.1%}")
            
            self.logger.info(f"Test abgeschlossen: {test_config.name} - Erfolg: {result.success}")
            
        except Exception as e:
            error_msg = f"Schwerwiegender Fehler in Test {test_config.name}: {str(e)}"
            result.errors.append(error_msg)
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            result.total_runtime = time.time() - start_time
        
        return result
    
    def run_test_suite(self, document_paths: List[str], 
                       output_file: str = None) -> Dict[str, Any]:
        """Führt eine komplette Test-Suite durch"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"rag_test_results_{timestamp}.json"
        
        self.logger.info(f"Starte Test-Suite mit {len(document_paths)} Dokumenten")
        
        # Test-Konfigurationen erstellen
        configurations = self.create_test_configurations(document_paths)
        self.logger.info(f"Erstellt {len(configurations)} Test-Konfigurationen")
        
        # Tests durchführen
        results = []
        start_time = time.time()
        
        for i, config in enumerate(configurations):
            self.logger.info(f"Test {i+1}/{len(configurations)}: {config.name}")
            
            try:
                result = self.run_single_test(config)
                results.append(result)
                
                # Zwischenergebnisse speichern
                if i % 10 == 0:  # Alle 10 Tests
                    self._save_intermediate_results(results, output_file)
                    
            except Exception as e:
                self.logger.error(f"Fehler bei Test {config.name}: {str(e)}")
                continue
        
        # Finale Analyse
        total_runtime = time.time() - start_time
        analysis = self._analyze_results(results)
        
        # Komplette Ergebnisse
        final_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(configurations),
                "successful_tests": len([r for r in results if r.success]),
                "total_runtime": total_runtime,
                "document_paths": document_paths
            },
            "test_results": [asdict(result) for result in results],
            "analysis": analysis,
            "recommendations": self._generate_recommendations(analysis, results)
        }
        
        # Ergebnisse speichern
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Test-Suite abgeschlossen. Ergebnisse gespeichert: {output_file}")
        return final_results
    
    def run_quick_comparison(self, document_path: str, 
                           models: List[str] = None) -> Dict[str, Any]:
        """Schneller Vergleich verschiedener Modelle für ein Dokument"""
        if models is None:
            models = self.default_models
        
        self.logger.info(f"Schnellvergleich für: {document_path}")
        
        # Standard-Konfiguration
        standard_config = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "chain_type": "stuff",
            "retrieval_k": 5,
            "search_k": 15
        }
        
        results = []
        test_questions = self.german_test_questions["general"][:3]  # Nur 3 Fragen für Schnelltest
        
        for model in models:
            config = TestConfiguration(
                name=f"quick_{model.replace(':', '_')}",
                model_name=model,
                document_path=document_path,
                test_questions=test_questions,
                timeout_seconds=120,  # Kurzes Timeout
                **standard_config
            )
            
            try:
                result = self.run_single_test(config)
                results.append(result)
                self.logger.info(f"Model {model}: {'✓' if result.success else '✗'} "
                               f"({result.questions_answered}/{result.questions_total} beantwortet)")
            except Exception as e:
                self.logger.error(f"Fehler bei Model {model}: {str(e)}")
        
        # Schnelle Analyse
        analysis = {
            "best_model": max(results, key=lambda r: r.questions_answered).model_name if results else None,
            "model_comparison": [
                {
                    "model": r.model_name,
                    "success_rate": r.questions_answered / r.questions_total,
                    "avg_response_time": r.retrieval_time / len(test_questions),
                    "avg_answer_length": r.avg_answer_length
                } for r in results
            ]
        }
        
        return {
            "results": [asdict(r) for r in results],
            "analysis": analysis
        }
    
    # Helper Methods
    def _get_file_size_mb(self, file_path: str) -> float:
        """Ermittelt Dateigröße in MB"""
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except:
            return 0.0
    
    def _select_test_questions(self, doc_path: str) -> List[str]:
        """Wählt passende Test-Fragen basierend auf Dokumenttyp"""
        doc_name = Path(doc_path).name.lower()
        
        if any(keyword in doc_name for keyword in ["legal", "recht", "gesetz", "vertrag"]):
            return self.german_test_questions["legal"]
        elif any(keyword in doc_name for keyword in ["tech", "system", "api", "code"]):
            return self.german_test_questions["technical"]
        else:
            return self.german_test_questions["general"]
    
    def _extract_expected_keywords(self, doc_path: str) -> List[str]:
        """Extrahiert erwartete Keywords basierend auf Dokumenttyp"""
        # Vereinfachte Implementierung - könnte erweitert werden
        doc_name = Path(doc_path).stem.lower()
        return doc_name.split("_")[:5]  # Erste 5 Wörter aus Dateiname
    
    def _count_keyword_matches(self, text: str, keywords: List[str]) -> int:
        """Zählt Keyword-Matches in Text"""
        if not keywords:
            return 0
        text_lower = text.lower()
        return sum(1 for keyword in keywords if keyword.lower() in text_lower)
    
    def _save_intermediate_results(self, results: List[TestResult], output_file: str):
        """Speichert Zwischenergebnisse"""
        try:
            intermediate_data = {
                "timestamp": datetime.now().isoformat(),
                "partial_results": [asdict(result) for result in results]
            }
            
            intermediate_file = str(output_file).replace(".json", "_intermediate.json")
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"Fehler beim Speichern der Zwischenergebnisse: {str(e)}")
    
    def _analyze_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """Analysiert Test-Ergebnisse"""
        if not results:
            return {"error": "Keine Ergebnisse vorhanden"}
        
        successful_results = [r for r in results if r.success]
        
        analysis = {
            "overall_stats": {
                "total_tests": len(results),
                "successful_tests": len(successful_results),
                "success_rate": len(successful_results) / len(results),
                "avg_runtime": statistics.mean([r.total_runtime for r in results]),
                "avg_questions_answered": statistics.mean([r.questions_answered for r in results])
            },
            "model_performance": {},
            "chunk_size_analysis": {},
            "chain_type_analysis": {},
            "document_size_impact": {}
        }
        
        # Modell-Performance
        for model in self.default_models:
            model_results = [r for r in results if r.model_name == model]
            if model_results:
                analysis["model_performance"][model] = {
                    "success_rate": len([r for r in model_results if r.success]) / len(model_results),
                    "avg_runtime": statistics.mean([r.total_runtime for r in model_results]),
                    "avg_answer_quality": statistics.mean([r.questions_answered / r.questions_total for r in model_results])
                }
        
        # Chunk-Size Analyse
        chunk_sizes = list(set(r.chunk_size for r in results))
        for chunk_size in chunk_sizes:
            chunk_results = [r for r in results if r.chunk_size == chunk_size]
            analysis["chunk_size_analysis"][chunk_size] = {
                "success_rate": len([r for r in chunk_results if r.success]) / len(chunk_results),
                "avg_retrieval_time": statistics.mean([r.retrieval_time for r in chunk_results])
            }
        
        # Chain-Type Analyse
        chain_types = list(set(r.chain_type for r in results))
        for chain_type in chain_types:
            chain_results = [r for r in results if r.chain_type == chain_type]
            analysis["chain_type_analysis"][chain_type] = {
                "success_rate": len([r for r in chain_results if r.success]) / len(chain_results),
                "avg_generation_time": statistics.mean([r.generation_time for r in chain_results])
            }
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any], 
                                results: List[TestResult]) -> List[str]:
        """Generiert Empfehlungen basierend auf Test-Ergebnissen"""
        recommendations = []
        
        # Modell-Empfehlungen
        if analysis.get("model_performance"):
            best_model = max(analysis["model_performance"].items(), 
                           key=lambda x: x[1]["success_rate"])
            recommendations.append(
                f"Bestes Modell: {best_model[0]} (Erfolgsrate: {best_model[1]['success_rate']:.1%})"
            )
        
        # Chunk-Size Empfehlungen
        if analysis.get("chunk_size_analysis"):
            best_chunk_size = max(analysis["chunk_size_analysis"].items(),
                                key=lambda x: x[1]["success_rate"])
            recommendations.append(
                f"Optimale Chunk-Größe: {best_chunk_size[0]} Zeichen"
            )
        
        # Chain-Type Empfehlungen
        if analysis.get("chain_type_analysis"):
            best_chain_type = max(analysis["chain_type_analysis"].items(),
                                key=lambda x: x[1]["success_rate"])
            recommendations.append(
                f"Bester Chain-Type: {best_chain_type[0]}"
            )
        
        # Performance-Empfehlungen
        slow_results = [r for r in results if r.total_runtime > 60]
        if slow_results:
            recommendations.append(
                f"Performance-Problem: {len(slow_results)} Tests über 60s - "
                "Chunk-Größe oder Retrieval-K reduzieren"
            )
        
        # Qualitäts-Empfehlungen
        low_quality_results = [r for r in results if r.questions_answered / r.questions_total < 0.5]
        if low_quality_results:
            recommendations.append(
                f"Qualitätsproblem: {len(low_quality_results)} Tests unter 50% Erfolgsrate - "
                "Embedding-Modell oder Retrieval-Strategie überprüfen"
            )
        
        return recommendations


def main():
    """Hauptfunktion für CLI-Nutzung des Test-Frameworks"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System Test Framework")
    parser.add_argument("documents", nargs="+", help="Pfade zu Test-Dokumenten")
    parser.add_argument("--output", "-o", help="Output-Datei für Ergebnisse")
    parser.add_argument("--quick", "-q", action="store_true", 
                       help="Schneller Vergleich nur mit Standard-Konfiguration")
    parser.add_argument("--models", "-m", nargs="+", 
                       help="Zu testende Modelle (Standard: alle verfügbaren)")
    parser.add_argument("--results-dir", default="test_results",
                       help="Verzeichnis für Ergebnisse")
    
    args = parser.parse_args()
    
    # Test-Framework initialisieren
    framework = RAGTestFramework(results_dir=args.results_dir)
    
    try:
        if args.quick:
            # Schnelltest
            for doc_path in args.documents:
                if not os.path.exists(doc_path):
                    print(f"❌ Dokument nicht gefunden: {doc_path}")
                    continue
                
                print(f"\n🔍 Schnelltest für: {doc_path}")
                results = framework.run_quick_comparison(doc_path, args.models)
                
                print("\n📊 Ergebnisse:")
                for model_info in results["analysis"]["model_comparison"]:
                    print(f"  {model_info['model']}: "
                          f"{model_info['success_rate']:.0%} Erfolg, "
                          f"{model_info['avg_response_time']:.1f}s Antwortzeit")
                
                if results["analysis"]["best_model"]:
                    print(f"\n🏆 Bestes Modell: {results['analysis']['best_model']}")
        else:
            # Komplette Test-Suite
            print(f"\n🚀 Starte komplette Test-Suite mit {len(args.documents)} Dokumenten")
            
            # Dokumentexistenz prüfen
            valid_docs = []
            for doc_path in args.documents:
                if os.path.exists(doc_path):
                    valid_docs.append(doc_path)
                    print(f"✅ {doc_path} ({framework._get_file_size_mb(doc_path):.1f} MB)")
                else:
                    print(f"❌ Nicht gefunden: {doc_path}")
            
            if not valid_docs:
                print("❌ Keine gültigen Dokumente gefunden!")
                sys.exit(1)
            
            # Test-Suite ausführen
            results = framework.run_test_suite(valid_docs, args.output)
            
            print(f"\n📈 Test-Suite abgeschlossen!")
            print(f"   Erfolgreiche Tests: {results['metadata']['successful_tests']}/{results['metadata']['total_tests']}")
            print(f"   Gesamtlaufzeit: {results['metadata']['total_runtime']:.1f}s")
            print(f"   Ergebnisse gespeichert: {args.output or 'auto-generiert'}")
            
            # Top-Empfehlungen anzeigen
            if results.get("recommendations"):
                print(f"\n💡 Top-Empfehlungen:")
                for rec in results["recommendations"][:3]:
                    print(f"   • {rec}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Test-Suite abgebrochen")
    except Exception as e:
        print(f"❌ Fehler: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()