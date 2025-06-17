#!/usr/bin/env python3
# test_utils.py
"""
RAG Test Framework - Utilities und Helper-Funktionen
====================================================

Zusätzliche Hilfsfunktionen für erweiterte Tests und Analysen.
Speziell für deutsche Dokumente und Performance-Optimierung.

Autor: RAG System Optimizer
Datum: 2025-06-16
"""

import os
import json
import time
import psutil
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetrics:
    """Metriken für Dokument-Analyse"""
    file_path: str
    file_size_mb: float
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    complexity_score: float
    language_detected: str
    file_type: str
    encoding: str

class DocumentAnalyzer:
    """Analysiert Dokumente vor dem RAG-Test"""
    
    def __init__(self):
        self.supported_extensions = {
            '.pdf': 'PDF',
            '.docx': 'Word',
            '.doc': 'Word Legacy',
            '.txt': 'Text',
            '.md': 'Markdown',
            '.xlsx': 'Excel',
            '.pptx': 'PowerPoint'
        }
    
    def analyze_document(self, file_path: str) -> DocumentMetrics:
        """Führt umfassende Dokument-Analyse durch"""
        from app.vectorstore import load_documents_from_file
        
        try:
            # Basis-Informationen
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            file_ext = Path(file_path).suffix.lower()
            file_type = self.supported_extensions.get(file_ext, 'Unknown')
            
            # Text extrahieren
            documents = load_documents_from_file(file_path, chunk_size=10000, chunk_overlap=0)
            if not documents:
                raise ValueError("Keine Dokumente geladen")
            
            full_text = "\n".join([doc.page_content for doc in documents])
            
            # Text-Analyse
            word_count = len(full_text.split())
            sentences = self._split_sentences(full_text)
            sentence_count = len(sentences)
            paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
            paragraph_count = len(paragraphs)
            
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            complexity_score = self._calculate_complexity(full_text)
            language_detected = self._detect_language(full_text)
            encoding = self._detect_encoding(file_path)
            
            return DocumentMetrics(
                file_path=file_path,
                file_size_mb=file_size_mb,
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                avg_sentence_length=avg_sentence_length,
                complexity_score=complexity_score,
                language_detected=language_detected,
                file_type=file_type,
                encoding=encoding
            )
            
        except Exception as e:
            logger.error(f"Fehler bei Dokument-Analyse {file_path}: {str(e)}")
            return DocumentMetrics(
                file_path=file_path,
                file_size_mb=0.0,
                word_count=0,
                sentence_count=0,
                paragraph_count=0,
                avg_sentence_length=0.0,
                complexity_score=0.0,
                language_detected="unknown",
                file_type="unknown",
                encoding="unknown"
            )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Einfache Satz-Trennung für deutsche Texte"""
        import re
        # Deutsche Satzenden: . ! ? mit nachfolgendem Großbuchstaben oder Zeilenende
        sentences = re.split(r'[.!?]+(?=\s+[A-ZÜÖÄ]|\s*$)', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_complexity(self, text: str) -> float:
        """Berechnet Text-Komplexität (vereinfacht)"""
        words = text.split()
        if not words:
            return 0.0
        
        # Durchschnittliche Wortlänge
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Lange Wörter zählen (typisch für Deutsche)
        long_words = sum(1 for word in words if len(word) > 12)
        long_word_ratio = long_words / len(words)
        
        # Komplexitäts-Score (0-100)
        complexity = min(100, (avg_word_length * 10) + (long_word_ratio * 50))
        return round(complexity, 2)
    
    def _detect_language(self, text: str) -> str:
        """Einfache Sprach-Erkennung"""
        german_indicators = ['der', 'die', 'das', 'und', 'ist', 'mit', 'für', 'auf', 'eine', 'einen']
        english_indicators = ['the', 'and', 'is', 'with', 'for', 'on', 'a', 'an', 'this', 'that']
        
        text_lower = text.lower()
        german_count = sum(1 for word in german_indicators if word in text_lower)
        english_count = sum(1 for word in english_indicators if word in text_lower)
        
        if german_count > english_count:
            return "german"
        elif english_count > german_count:
            return "english"
        else:
            return "mixed"
    
    def _detect_encoding(self, file_path: str) -> str:
        """Erkennt Datei-Encoding"""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Erste 10KB
                result = chardet.detect(raw_data)
                return result.get('encoding', 'unknown')
        except:
            return "utf-8"  # Standard-Annahme

class PerformanceMonitor:
    """Überwacht System-Performance während Tests"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.start_time = None
    
    def start_monitoring(self):
        """Startet Performance-Monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        self.metrics = []
        logger.info("Performance-Monitoring gestartet")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stoppt Monitoring und gibt Zusammenfassung zurück"""
        self.monitoring = False
        
        if not self.metrics:
            return {"error": "Keine Metriken gesammelt"}
        
        # Statistiken berechnen
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_mb'] for m in self.metrics]
        
        summary = {
            "duration_seconds": time.time() - self.start_time if self.start_time else 0,
            "cpu_stats": {
                "avg": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values)
            },
            "memory_stats": {
                "avg_mb": np.mean(memory_values),
                "max_mb": np.max(memory_values),
                "min_mb": np.min(memory_values)
            },
            "sample_count": len(self.metrics)
        }
        
        logger.info(f"Performance-Monitoring beendet: CPU avg={summary['cpu_stats']['avg']:.1f}%, "
                   f"Memory avg={summary['memory_stats']['avg_mb']:.1f}MB")
        
        return summary
    
    def collect_sample(self):
        """Sammelt eine Performance-Probe"""
        if not self.monitoring:
            return
        
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            
            sample = {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_mb": memory_info.rss / (1024 * 1024),
                "system_cpu": psutil.cpu_percent(),
                "system_memory": psutil.virtual_memory().percent
            }
            
            self.metrics.append(sample)
            
        except Exception as e:
            logger.warning(f"Fehler beim Performance-Sampling: {str(e)}")

class TestResultVisualizer:
    """Erstellt Visualisierungen für Test-Ergebnisse"""
    
    def __init__(self, output_dir: str = "test_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Matplotlib für deutsche Texte konfigurieren
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    
    def create_performance_dashboard(self, results_file: str) -> str:
        """Erstellt umfassendes Performance-Dashboard"""
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # DataFrame erstellen
        df = pd.DataFrame(data.get('test_results', []))
        if df.empty:
            logger.warning("Keine Test-Ergebnisse für Visualisierung")
            return None
        
        # Dashboard mit mehreren Subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RAG System Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Erfolgsrate pro Modell
        model_success = df.groupby('model_name')['success'].mean()
        axes[0, 0].bar(model_success.index, model_success.values)
        axes[0, 0].set_title('Erfolgsrate pro Modell')
        axes[0, 0].set_ylabel('Erfolgsrate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Runtime vs. Chunk Size
        axes[0, 1].scatter(df['chunk_size'], df['total_runtime'], 
                          c=df['success'].map({True: 'green', False: 'red'}),
                          alpha=0.6)
        axes[0, 1].set_title('Laufzeit vs. Chunk-Größe')
        axes[0, 1].set_xlabel('Chunk-Größe')
        axes[0, 1].set_ylabel('Laufzeit (s)')
        
        # 3. Antwortqualität (Fragen beantwortet)
        axes[0, 2].hist(df['questions_answered'], bins=10, alpha=0.7)
        axes[0, 2].set_title('Verteilung beantworteter Fragen')
        axes[0, 2].set_xlabel('Anzahl beantworteter Fragen')
        axes[0, 2].set_ylabel('Häufigkeit')
        
        # 4. Chain-Type Performance
        chain_perf = df.groupby('chain_type').agg({
            'total_runtime': 'mean',
            'success': 'mean'
        })
        
        x_pos = np.arange(len(chain_perf.index))
        axes[1, 0].bar(x_pos - 0.2, chain_perf['total_runtime'], 0.4, 
                      label='Avg Runtime', alpha=0.7)
        axes[1, 0].bar(x_pos + 0.2, chain_perf['success'] * 100, 0.4, 
                      label='Success Rate (%)', alpha=0.7)
        axes[1, 0].set_title('Chain-Type Performance')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(chain_perf.index)
        axes[1, 0].legend()
        
        # 5. Dokumentgröße vs. Performance
        axes[1, 1].scatter(df['document_size_mb'], df['embedding_time'], 
                          alpha=0.6, label='Embedding Time')
        axes[1, 1].scatter(df['document_size_mb'], df['retrieval_time'], 
                          alpha=0.6, label='Retrieval Time')
        axes[1, 1].set_title('Dokumentgröße vs. Verarbeitungszeit')
        axes[1, 1].set_xlabel('Dokumentgröße (MB)')
        axes[1, 1].set_ylabel('Zeit (s)')
        axes[1, 1].legend()
        
        # 6. Retrieval-K Optimierung
        k_performance = df.groupby('retrieval_k').agg({
            'questions_answered': 'mean',
            'retrieval_time': 'mean'
        })
        
        axes[1, 2].plot(k_performance.index, k_performance['questions_answered'], 
                       'o-', label='Avg Questions Answered')
        ax2 = axes[1, 2].twinx()
        ax2.plot(k_performance.index, k_performance['retrieval_time'], 
                'r^-', label='Avg Retrieval Time')
        axes[1, 2].set_title('Retrieval-K Optimierung')
        axes[1, 2].set_xlabel('Retrieval K')
        axes[1, 2].set_ylabel('Fragen beantwortet', color='b')
        ax2.set_ylabel('Retrieval Zeit (s)', color='r')
        
        plt.tight_layout()
        
        # Dashboard speichern
        output_file = self.output_dir / f"performance_dashboard_{int(time.time())}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance Dashboard erstellt: {output_file}")
        return str(output_file)
    
    def create_model_comparison_chart(self, results_file: str) -> str:
        """Erstellt detaillierten Modell-Vergleich"""
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data.get('test_results', []))
        if df.empty:
            return None
        
        # Modell-Metriken aggregieren
        model_metrics = df.groupby('model_name').agg({
            'success': 'mean',
            'total_runtime': 'mean',
            'questions_answered': 'mean',
            'avg_answer_length': 'mean',
            'embedding_time': 'mean',
            'retrieval_time': 'mean'
        }).round(2)
        
        # Heatmap erstellen
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalisierte Werte für bessere Visualisierung
        normalized_metrics = model_metrics.copy()
        for col in normalized_metrics.columns:
            normalized_metrics[col] = (normalized_metrics[col] - normalized_metrics[col].min()) / \
                                    (normalized_metrics[col].max() - normalized_metrics[col].min())
        
        sns.heatmap(normalized_metrics.T, annot=model_metrics.T, fmt='.2f', 
                   cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Normalisierte Performance'})
        
        ax.set_title('Modell-Performance Vergleich', fontsize=14, fontweight='bold')
        ax.set_xlabel('Modelle')
        ax.set_ylabel('Metriken')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"model_comparison_{int(time.time())}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Modell-Vergleich erstellt: {output_file}")
        return str(output_file)

class BatchTestRunner:
    """Führt Tests parallel aus für bessere Performance"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.performance_monitor = PerformanceMonitor()
    
    def run_parallel_tests(self, test_framework, configurations: List) -> List:
        """Führt Tests parallel aus"""
        logger.info(f"Starte parallele Tests mit {self.max_workers} Workern")
        
        self.performance_monitor.start_monitoring()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Tests einreichen
            future_to_config = {
                executor.submit(test_framework.run_single_test, config): config 
                for config in configurations
            }
            
            # Ergebnisse sammeln
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result(timeout=600)  # 10 Min Timeout
                    results.append(result)
                    logger.info(f"Test abgeschlossen: {config.name} - "
                               f"{'✓' if result.success else '✗'}")
                except Exception as e:
                    logger.error(f"Test fehlgeschlagen: {config.name} - {str(e)}")
                
                # Performance-Sample sammeln
                self.performance_monitor.collect_sample()
        
        perf_summary = self.performance_monitor.stop_monitoring()
        logger.info(f"Parallele Tests abgeschlossen. Performance: {perf_summary}")
        
        return results

class ReportGenerator:
    """Generiert detaillierte Test-Berichte"""
    
    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_html_report(self, results_file: str) -> str:
        """Generiert HTML-Bericht"""
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        html_content = self._create_html_template(data)
        
        # HTML-Datei speichern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"rag_test_report_{timestamp}.html"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML-Bericht erstellt: {output_file}")
        return str(output_file)
    
    def _create_html_template(self, data: Dict[str, Any]) -> str:
        """Erstellt HTML-Template für Bericht"""
        metadata = data.get('metadata', {})
        analysis = data.get('analysis', {})
        recommendations = data.get('recommendations', [])
        
        html = f"""
        <!DOCTYPE html>
        <html lang="de">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RAG System Test Bericht</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .metric-box {{ background: #e9f5ff; padding: 15px; border-radius: 5px; text-align: center; }}
                .success {{ background: #d4edda; }}
                .warning {{ background: #fff3cd; }}
                .error {{ background: #f8d7da; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #f4f4f4; }}
                .recommendations {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 RAG System Test Bericht</h1>
                <p><strong>Erstellt:</strong> {metadata.get('timestamp', 'N/A')}</p>
                <p><strong>Dokumente:</strong> {len(metadata.get('document_paths', []))}</p>
            </div>
            
            <h2>📊 Überblick</h2>
            <div class="metrics">
                <div class="metric-box {'success' if metadata.get('successful_tests', 0) / metadata.get('total_tests', 1) > 0.7 else 'warning'}">
                    <h3>{metadata.get('successful_tests', 0)}/{metadata.get('total_tests', 0)}</h3>
                    <p>Erfolgreiche Tests</p>
                </div>
                <div class="metric-box">
                    <h3>{metadata.get('total_runtime', 0):.1f}s</h3>
                    <p>Gesamtlaufzeit</p>
                </div>
                <div class="metric-box">
                    <h3>{analysis.get('overall_stats', {}).get('success_rate', 0):.1%}</h3>
                    <p>Erfolgsrate</p>
                </div>
                <div class="metric-box">
                    <h3>{analysis.get('overall_stats', {}).get('avg_runtime', 0):.1f}s</h3>
                    <p>Ø Testlaufzeit</p>
                </div>
            </div>
            
            <h2>🏆 Top-Empfehlungen</h2>
            <div class="recommendations">
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in recommendations[:5])}
                </ul>
            </div>
            
            <h2>📈 Modell-Performance</h2>
            <table>
                <tr>
                    <th>Modell</th>
                    <th>Erfolgsrate</th>
                    <th>Ø Laufzeit</th>
                    <th>Ø Antwortqualität</th>
                </tr>
        """
        
        # Modell-Performance Tabelle
        model_perf = analysis.get('model_performance', {})
        for model, metrics in model_perf.items():
            html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{metrics.get('success_rate', 0):.1%}</td>
                    <td>{metrics.get('avg_runtime', 0):.1f}s</td>
                    <td>{metrics.get('avg_answer_quality', 0):.1%}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>⚙️ Konfiguration-Analyse</h2>
            <h3>Chunk-Größen</h3>
            <table>
                <tr>
                    <th>Chunk-Größe</th>
                    <th>Erfolgsrate</th>
                    <th>Ø Retrieval-Zeit</th>
                </tr>
        """
        
        # Chunk-Size Analyse
        chunk_analysis = analysis.get('chunk_size_analysis', {})
        for chunk_size, metrics in chunk_analysis.items():
            html += f"""
                <tr>
                    <td>{chunk_size}</td>
                    <td>{metrics.get('success_rate', 0):.1%}</td>
                    <td>{metrics.get('avg_retrieval_time', 0):.2f}s</td>
                </tr>
            """
        
        html += """
            </table>
            
            <footer style="margin-top: 50px; text-align: center; color: #666;">
                <p>Generiert vom RAG Test Framework - Optimiert für deutsche Dokumente</p>
            </footer>
        </body>
        </html>
        """
        
        return html

# Praktische Utility-Funktionen
def quick_document_analysis(file_path: str) -> Dict[str, Any]:
    """Schnelle Dokument-Analyse für Vor-Test-Bewertung"""
    analyzer = DocumentAnalyzer()
    metrics = analyzer.analyze_document(file_path)
    
    # Empfehlungen basierend auf Analyse
    recommendations = []
    
    if metrics.file_size_mb > 10:
        recommendations.append("Sehr große Datei - map_reduce Chain empfohlen")
    if metrics.complexity_score > 70:
        recommendations.append("Komplexer Text - kleinere Chunks verwenden")
    if metrics.language_detected != "german":
        recommendations.append("Nicht-deutsche Sprache erkannt - Embedding-Modell prüfen")
    
    return {
        "metrics": metrics,
        "recommendations": recommendations
    }

def estimate_test_duration(document_paths: List[str], 
                          models: List[str]) -> Dict[str, Any]:
    """Schätzt Testdauer basierend auf Dokumenten und Modellen"""
    total_size_mb = sum(os.path.getsize(path) / (1024 * 1024) 
                       for path in document_paths if os.path.exists(path))
    
    # Basis-Schätzung: ~30s pro MB pro Modell für Standard-Tests
    base_time_per_mb = 30
    num_configs_per_doc = 12  # Verschiedene Chunk-Größen, Chain-Types, etc.
    
    estimated_seconds = total_size_mb * base_time_per_mb * len(models) * num_configs_per_doc
    
    return {
        "total_documents": len(document_paths),
        "total_size_mb": total_size_mb,
        "models_count": len(models),
        "estimated_tests": len(document_paths) * len(models) * num_configs_per_doc,
        "estimated_duration_minutes": estimated_seconds / 60,
        "estimated_duration_hours": estimated_seconds / 3600
    }

if __name__ == "__main__":
    # Beispiel-Nutzung der Utilities
    print("🔧 RAG Test Utilities geladen")
    
    # Dokument-Analyse Beispiel
    test_file = "test_document.pdf"
    if os.path.exists(test_file):
        analysis = quick_document_analysis(test_file)
        print(f"📄 Dokument-Analyse für {test_file}:")
        print(f"   Größe: {analysis['metrics'].file_size_mb:.1f} MB")
        print(f"   Wörter: {analysis['metrics'].word_count}")
        print(f"   Komplexität: {analysis['metrics'].complexity_score}")
        print(f"   Sprache: {analysis['metrics'].language_detected}")
        print("💡 Empfehlungen:")
        for rec in analysis['recommendations']:
            print(f"   • {rec}")