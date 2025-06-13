# app/rag.py - Erweitert um detailliertes Debugging und Performance-Analyse
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from app.config import config
from app.connection_utils import check_ollama_connection_with_retry
from app.token_tracker import token_tracker, TokenUsage
import requests
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
import statistics

# Logger aus Config verwenden
logger = config.get_logger("RAG")

def check_ollama_connection():
    """Einfache Ollama-Verbindungsprüfung (Kompatibilität)"""
    return check_ollama_connection_with_retry(max_retries=3)

def get_prompt_template():
    """Define a prompt template for RAG"""
    template = """Du bist ein hilfreicher Assistent, der Fragen basierend auf den bereitgestellten Dokumenten beantwortet.
    
Kontext:
{context}

Frage: {question}

Beantworte die Frage nur basierend auf dem gegebenen Kontext. Wenn die Antwort nicht im Kontext gefunden werden kann, 
sage "Ich kann diese Frage nicht auf Basis der verfügbaren Dokumente beantworten." und gib keine Spekulationen ab.

Deine Antwort:"""
    
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

class DocumentRetrievalDebugger:
    """Detailliertes Debugging für Document Retrieval"""
    
    def __init__(self):
        self.debug_data = []
        self.similarity_threshold = 0.7  # Konfigurierbar
        
    def analyze_retrieval(self, question: str, documents: List, vectorstore, k_values: List[int] = None) -> Dict:
        """
        Analysiert Document Retrieval mit verschiedenen k-Werten
        
        Args:
            question: Die gestellte Frage
            documents: Abgerufene Dokumente
            vectorstore: Der Vektorspeicher
            k_values: Liste verschiedener k-Werte zum Testen
            
        Returns:
            Dict mit detaillierter Analyse
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 15, 20]
            
        analysis = {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "current_retrieval": self._analyze_documents(documents),
            "k_value_comparison": {},
            "similarity_analysis": {},
            "recommendations": []
        }
        
        # Verschiedene k-Werte testen
        for k in k_values:
            try:
                test_docs = vectorstore.similarity_search_with_score(question, k=k)
                analysis["k_value_comparison"][k] = {
                    "document_count": len(test_docs),
                    "avg_similarity": statistics.mean([score for _, score in test_docs]) if test_docs else 0,
                    "min_similarity": min([score for _, score in test_docs]) if test_docs else 0,
                    "max_similarity": max([score for _, score in test_docs]) if test_docs else 0,
                    "documents": [{"content_length": len(doc.page_content), "similarity": score} 
                                for doc, score in test_docs]
                }
            except Exception as e:
                logger.warning(f"Fehler bei k={k} Test: {str(e)}")
                analysis["k_value_comparison"][k] = {"error": str(e)}
        
        # Similarity-Score-Analyse
        if hasattr(vectorstore, 'similarity_search_with_score'):
            try:
                scored_docs = vectorstore.similarity_search_with_score(question, k=20)
                analysis["similarity_analysis"] = self._analyze_similarity_scores(scored_docs)
            except Exception as e:
                logger.warning(f"Similarity-Score-Analyse fehlgeschlagen: {str(e)}")
        
        # Empfehlungen generieren
        analysis["recommendations"] = self._generate_retrieval_recommendations(analysis)
        
        self.debug_data.append(analysis)
        return analysis
    
    def _analyze_documents(self, documents: List) -> Dict:
        """Analysiert eine Liste von Dokumenten"""
        if not documents:
            return {"count": 0, "total_length": 0, "avg_length": 0}
            
        lengths = [len(doc.page_content) for doc in documents]
        return {
            "count": len(documents),
            "total_length": sum(lengths),
            "avg_length": statistics.mean(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "std_length": statistics.stdev(lengths) if len(lengths) > 1 else 0
        }
    
    def _analyze_similarity_scores(self, scored_docs: List[Tuple]) -> Dict:
        """Analysiert Similarity-Scores"""
        if not scored_docs:
            return {}
            
        scores = [score for _, score in scored_docs]
        high_quality = [s for s in scores if s >= self.similarity_threshold]
        
        return {
            "total_documents": len(scored_docs),
            "avg_similarity": statistics.mean(scores),
            "min_similarity": min(scores),
            "max_similarity": max(scores),
            "std_similarity": statistics.stdev(scores) if len(scores) > 1 else 0,
            "high_quality_docs": len(high_quality),
            "quality_ratio": len(high_quality) / len(scored_docs) if scored_docs else 0,
            "similarity_distribution": {
                "excellent": len([s for s in scores if s >= 0.9]),
                "good": len([s for s in scores if 0.7 <= s < 0.9]),
                "fair": len([s for s in scores if 0.5 <= s < 0.7]),
                "poor": len([s for s in scores if s < 0.5])
            }
        }
    
    def _generate_retrieval_recommendations(self, analysis: Dict) -> List[str]:
        """Generiert Empfehlungen basierend auf der Analyse"""
        recommendations = []
        
        # K-Wert-Empfehlungen
        k_comparison = analysis.get("k_value_comparison", {})
        best_k = None
        best_avg_sim = -1
        
        for k, data in k_comparison.items():
            if isinstance(data, dict) and "avg_similarity" in data:
                if data["avg_similarity"] > best_avg_sim:
                    best_avg_sim = data["avg_similarity"]
                    best_k = k
        
        if best_k:
            recommendations.append(f"Optimaler k-Wert: {best_k} (Durchschn. Similarity: {best_avg_sim:.3f})")
        
        # Similarity-Empfehlungen
        sim_analysis = analysis.get("similarity_analysis", {})
        if sim_analysis:
            quality_ratio = sim_analysis.get("quality_ratio", 0)
            if quality_ratio < 0.3:
                recommendations.append("⚠️ Niedrige Retrieval-Qualität: Weniger als 30% der Dokumente haben hohe Similarity")
                recommendations.append("💡 Überprüfen Sie: Chunking-Strategie, Embedding-Modell, Frage-Formulierung")
            
            avg_sim = sim_analysis.get("avg_similarity", 0)
            if avg_sim < 0.5:
                recommendations.append("⚠️ Sehr niedrige durchschnittliche Similarity: Möglicherweise unpassende Dokumente")
        
        # Dokument-Längen-Empfehlungen
        current_retrieval = analysis.get("current_retrieval", {})
        if current_retrieval.get("avg_length", 0) > 4000:
            recommendations.append("💡 Lange Dokumente detected: Erwägen Sie feineres Chunking")
        
        return recommendations

class ChunkAnalyzer:
    """Analysiert Chunk-Qualität und -Verteilung"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_chunks(self, vectorstore, sample_size: int = 100) -> Dict:
        """
        Analysiert die Qualität der Chunks im Vektorspeicher
        
        Args:
            vectorstore: Der zu analysierende Vektorspeicher
            sample_size: Anzahl der zu analysierenden Sample-Chunks
            
        Returns:
            Dict mit Chunk-Analyse
        """
        cache_key = f"{id(vectorstore)}_{sample_size}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        try:
            # Sample Chunks abrufen
            if hasattr(vectorstore, '_collection'):
                all_docs = vectorstore._collection.get()
                documents = all_docs.get('documents', [])
                metadatas = all_docs.get('metadatas', [])
            else:
                # Fallback: Dummy-Query für Sample
                dummy_docs = vectorstore.similarity_search("sample", k=min(sample_size, 50))
                documents = [doc.page_content for doc in dummy_docs]
                metadatas = [doc.metadata for doc in dummy_docs]
            
            if not documents:
                return {"error": "Keine Dokumente im Vektorspeicher gefunden"}
            
            # Chunk-Statistiken berechnen
            analysis = self._compute_chunk_statistics(documents, metadatas)
            analysis["sample_size"] = len(documents)
            analysis["timestamp"] = datetime.now().isoformat()
            
            # Cache speichern
            self.analysis_cache[cache_key] = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"Chunk-Analyse fehlgeschlagen: {str(e)}")
            return {"error": str(e)}
    
    def _compute_chunk_statistics(self, documents: List[str], metadatas: List[Dict]) -> Dict:
        """Berechnet detaillierte Chunk-Statistiken"""
        lengths = [len(doc) for doc in documents]
        word_counts = [len(doc.split()) for doc in documents]
        
        # Source-Dateien analysieren
        sources = {}
        for metadata in metadatas:
            source = metadata.get('source', 'unknown')
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        # Sprach-Charakteristika (einfache Heuristik für Deutsch)
        german_indicators = ['der', 'die', 'das', 'und', 'ist', 'mit', 'zu', 'auf', 'für']
        avg_german_score = 0
        if documents:
            german_scores = []
            for doc in documents[:min(50, len(documents))]:  # Sample für Performance
                words = doc.lower().split()
                german_count = sum(1 for word in words if word in german_indicators)
                german_scores.append(german_count / len(words) if words else 0)
            avg_german_score = statistics.mean(german_scores) if german_scores else 0
        
        return {
            "chunk_count": len(documents),
            "length_stats": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "mean": statistics.mean(lengths) if lengths else 0,
                "median": statistics.median(lengths) if lengths else 0,
                "std": statistics.stdev(lengths) if len(lengths) > 1 else 0
            },
            "word_count_stats": {
                "min": min(word_counts) if word_counts else 0,
                "max": max(word_counts) if word_counts else 0,
                "mean": statistics.mean(word_counts) if word_counts else 0,
                "median": statistics.median(word_counts) if word_counts else 0
            },
            "source_distribution": sources,
            "source_count": len(sources),
            "german_language_score": avg_german_score,
            "recommendations": self._generate_chunk_recommendations(lengths, avg_german_score)
        }
    
    def _generate_chunk_recommendations(self, lengths: List[int], german_score: float) -> List[str]:
        """Generiert Empfehlungen für Chunk-Optimierung"""
        recommendations = []
        
        if not lengths:
            return ["⚠️ Keine Chunks zur Analyse verfügbar"]
        
        avg_length = statistics.mean(lengths)
        std_length = statistics.stdev(lengths) if len(lengths) > 1 else 0
        
        # Längen-Empfehlungen
        if avg_length > 2000:
            recommendations.append("💡 Chunks sind sehr lang - erwägen Sie kleinere Chunk-Größen")
        elif avg_length < 200:
            recommendations.append("💡 Chunks sind sehr kurz - möglicherweise zu wenig Kontext")
        
        # Variabilität
        if std_length > avg_length * 0.5:
            recommendations.append("⚠️ Hohe Variabilität in Chunk-Größen - inkonsistentes Chunking")
        
        # Sprach-spezifische Empfehlungen
        if german_score < 0.05:
            recommendations.append("⚠️ Niedrige deutsche Sprach-Indikatoren - prüfen Sie deutsche Embedding-Modelle")
        
        return recommendations

class PerformanceProfiler:
    """Profiling für RAG-Performance"""
    
    def __init__(self):
        self.performance_log = []
        self.current_session = None
    
    def start_profiling(self, operation: str) -> str:
        """Startet Profiling für eine Operation"""
        session_id = f"{operation}_{int(time.time() * 1000)}"
        self.current_session = {
            "session_id": session_id,
            "operation": operation,
            "start_time": time.time(),
            "stages": {}
        }
        return session_id
    
    def log_stage(self, stage_name: str, **kwargs):
        """Loggt eine Stage im aktuellen Profiling"""
        if not self.current_session:
            return
            
        current_time = time.time()
        self.current_session["stages"][stage_name] = {
            "timestamp": current_time,
            "elapsed_from_start": current_time - self.current_session["start_time"],
            **kwargs
        }
    
    def end_profiling(self) -> Dict:
        """Beendet das aktuelle Profiling und gibt Statistiken zurück"""
        if not self.current_session:
            return {}
        
        end_time = time.time()
        total_time = end_time - self.current_session["start_time"]
        
        # Performance-Analyse
        performance_data = {
            **self.current_session,
            "end_time": end_time,
            "total_duration": total_time,
            "performance_analysis": self._analyze_performance(self.current_session["stages"], total_time)
        }
        
        self.performance_log.append(performance_data)
        self.current_session = None
        
        return performance_data
    
    def _analyze_performance(self, stages: Dict, total_time: float) -> Dict:
        """Analysiert Performance-Daten"""
        if not stages:
            return {}
        
        stage_times = []
        bottlenecks = []
        
        prev_time = 0
        for stage_name, stage_data in stages.items():
            elapsed = stage_data["elapsed_from_start"]
            stage_duration = elapsed - prev_time
            stage_times.append((stage_name, stage_duration))
            
            # Bottleneck-Detection (>30% der Gesamtzeit)
            if stage_duration > total_time * 0.3:
                bottlenecks.append(f"{stage_name}: {stage_duration:.2f}s ({stage_duration/total_time*100:.1f}%)")
            
            prev_time = elapsed
        
        return {
            "stage_durations": stage_times,
            "bottlenecks": bottlenecks,
            "recommendations": self._generate_performance_recommendations(stage_times, total_time)
        }
    
    def _generate_performance_recommendations(self, stage_times: List[Tuple], total_time: float) -> List[str]:
        """Generiert Performance-Empfehlungen"""
        recommendations = []
        
        if total_time > 30:
            recommendations.append("⚠️ Sehr lange Antwortzeit (>30s) - System-Optimierung erforderlich")
        elif total_time > 10:
            recommendations.append("💡 Lange Antwortzeit (>10s) - Chunking oder Modell-Optimierung erwägen")
        
        # Stage-spezifische Empfehlungen
        for stage_name, duration in stage_times:
            if duration > total_time * 0.4:
                if "retrieval" in stage_name.lower():
                    recommendations.append(f"🔍 {stage_name} ist Bottleneck - Vektorspeicher-Optimierung nötig")
                elif "token" in stage_name.lower() or "context" in stage_name.lower():
                    recommendations.append(f"⚡ {stage_name} ist Bottleneck - Token-Management optimieren")
                elif "llm" in stage_name.lower() or "generation" in stage_name.lower():
                    recommendations.append(f"🤖 {stage_name} ist Bottleneck - kleineres/schnelleres Modell erwägen")
        
        return recommendations

class SmartContextManager:
    """Intelligentes Context-Management mit erweiterten Debugging-Funktionen"""
    
    def __init__(self, token_counter, model_name):
        self.token_counter = token_counter
        self.model_name = model_name
        self.debug_enabled = True
        
        # Token-Limits aus Config laden
        self.context_limits = config.calculate_context_limits(model_name)
        self.context_limit = self.context_limits['context_limit']
        
        logger.info(f"Context Manager für {model_name}:")
        logger.info(f"  - Gesamt-Limit: {self.context_limits['total_limit']:,} Token")
        logger.info(f"  - Context-Limit: {self.context_limit:,} Token")
        logger.info(f"  - Sicherheitspuffer: {self.context_limits['safety_margin']:.1%}")
    
    def prepare_context(self, documents, question, debug_info=None):
        """
        Bereitet den Kontext vor mit erweiterten Debugging-Informationen
        
        Args:
            documents: Liste der relevanten Dokumente
            question: Die gestellte Frage
            debug_info: Optionale Debug-Informationen
            
        Returns:
            tuple: (optimized_context, detailed_truncation_info)
        """
        start_time = time.time()
        
        question_tokens = self.token_counter.count_tokens(question, self.model_name)
        available_context_tokens = self.context_limit - question_tokens
        
        if available_context_tokens <= 0:
            logger.warning(f"Frage zu lang ({question_tokens} Token). Context-Limit überschritten.")
            return "", {
                "truncated": True, 
                "reason": "question_too_long",
                "question_tokens": question_tokens,
                "available_tokens": 0,
                "processing_time": time.time() - start_time
            }
        
        # Detaillierte Dokument-Analyse
        doc_analysis = []
        context_parts = []
        total_context_tokens = 0
        used_docs = 0
        
        for i, doc in enumerate(documents):
            doc_content = doc.page_content
            doc_tokens = self.token_counter.count_tokens(doc_content, self.model_name)
            
            doc_info = {
                "index": i,
                "content_length": len(doc_content),
                "token_count": doc_tokens,
                "source": doc.metadata.get('source', 'unknown'),
                "chunk_id": doc.metadata.get('chunk_id', f'chunk_{i}'),
                "included": False,
                "truncated": False
            }
            
            if total_context_tokens + doc_tokens <= available_context_tokens:
                context_parts.append(doc_content)
                total_context_tokens += doc_tokens
                used_docs += 1
                doc_info["included"] = True
            else:
                # Versuche, einen Teil des Dokuments zu verwenden
                remaining_tokens = available_context_tokens - total_context_tokens
                if remaining_tokens > 100:  # Mindestens 100 Token für sinnvollen Content
                    truncated_content = self._truncate_text_to_tokens(
                        doc_content, remaining_tokens, self.model_name
                    )
                    if truncated_content:
                        context_parts.append(truncated_content + "... [abgeschnitten]")
                        total_context_tokens += remaining_tokens
                        used_docs += 1
                        doc_info["included"] = True
                        doc_info["truncated"] = True
                        doc_info["truncated_tokens"] = remaining_tokens
                break
            
            doc_analysis.append(doc_info)
        
        final_context = "\n\n".join(context_parts)
        processing_time = time.time() - start_time
        
        # Erweiterte Truncation-Informationen
        truncation_info = {
            "truncated": used_docs < len(documents),
            "used_docs": used_docs,
            "total_docs": len(documents),
            "context_tokens": total_context_tokens,
            "available_tokens": available_context_tokens,
            "question_tokens": question_tokens,
            "processing_time": processing_time,
            "doc_analysis": doc_analysis,
            "efficiency_ratio": total_context_tokens / available_context_tokens if available_context_tokens > 0 else 0,
            "unused_docs": len(documents) - used_docs,
            "token_utilization": {
                "used": total_context_tokens,
                "available": available_context_tokens,
                "wasted": available_context_tokens - total_context_tokens,
                "efficiency": (total_context_tokens / available_context_tokens * 100) if available_context_tokens > 0 else 0
            }
        }
        
        # Erweiterte Warnungen und Empfehlungen
        if truncation_info["truncated"]:
            unused_docs = truncation_info["unused_docs"]
            logger.warning(f"Context truncated: {used_docs}/{len(documents)} Dokumente verwendet")
            logger.warning(f"Token-Effizienz: {truncation_info['efficiency_ratio']:.1%}")
            logger.warning(f"{unused_docs} Dokumente nicht berücksichtigt")
            
            if unused_docs > len(documents) * 0.5:
                logger.error("⚠️ KRITISCH: Mehr als 50% der relevanten Dokumente wurden ignoriert!")
                truncation_info["critical_truncation"] = True
        
        return final_context, truncation_info
    
    def _truncate_text_to_tokens(self, text, max_tokens, model_name):
        """Erweiterte Text-Truncation mit besserer Strategie"""
        if self.token_counter.count_tokens(text, model_name) <= max_tokens:
            return text
        
        # Strategisches Truncation: Versuche bei Satzenden zu kürzen
        sentences = text.split('.')
        truncated_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence + '.', model_name)
            if current_tokens + sentence_tokens <= max_tokens * 0.95:  # 5% Puffer
                truncated_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        if truncated_sentences:
            return '. '.join(truncated_sentences) + '.'
        
        # Fallback: Zeichen-basierte Truncation
        char_ratio = 0.75
        while char_ratio > 0.1:
            truncated = text[:int(len(text) * char_ratio)]
            if self.token_counter.count_tokens(truncated, model_name) <= max_tokens:
                return truncated
            char_ratio -= 0.1
        
        return ""

class TokenTrackingQA:
    """Erweiterte QA-Chain mit umfassendem Debugging"""
    
    def __init__(self, qa_chain, model_name):
        self.qa_chain = qa_chain
        self.model_name = model_name
        self.context_manager = SmartContextManager(token_tracker.counter, model_name)
        self.retrieval_debugger = DocumentRetrievalDebugger()
        self.performance_profiler = PerformanceProfiler()
        self.chunk_analyzer = ChunkAnalyzer()
    
    def invoke(self, query_dict, enable_debug=True):
        """
        Erweiterte QA-Ausführung mit umfassendem Debugging
        
        Args:
            query_dict: Dictionary mit 'query' Key
            enable_debug: Aktiviert detailliertes Debugging
            
        Returns:
            Dictionary mit Ergebnis und umfassenden Debug-Informationen
        """
        question = query_dict.get('query', '')
        
        # Performance-Profiling starten
        session_id = self.performance_profiler.start_profiling("qa_invoke")
        
        # Token-Tracking starten
        token_tracker.start_request()
        
        try:
            # Stage 1: Document Retrieval
            self.performance_profiler.log_stage("retrieval_start")
            retriever = self.qa_chain.retriever
            docs = retriever.get_relevant_documents(question)
            self.performance_profiler.log_stage("retrieval_complete", doc_count=len(docs))
            
            # Debug-Analyse des Retrievals
            debug_info = {}
            if enable_debug:
                self.performance_profiler.log_stage("debug_analysis_start")
                debug_info["retrieval_analysis"] = self.retrieval_debugger.analyze_retrieval(
                    question, docs, retriever.vectorstore
                )
                
                # Chunk-Analyse (gecacht für Performance)
                debug_info["chunk_analysis"] = self.chunk_analyzer.analyze_chunks(
                    retriever.vectorstore, sample_size=50
                )
                self.performance_profiler.log_stage("debug_analysis_complete")
            
            # Stage 2: Context Preparation
            self.performance_profiler.log_stage("context_preparation_start")
            optimized_context, truncation_info = self.context_manager.prepare_context(
                docs, question, debug_info
            )
            self.performance_profiler.log_stage("context_preparation_complete", 
                                              context_tokens=truncation_info.get("context_tokens", 0))
            
            # Stage 3: LLM Generation
            self.performance_profiler.log_stage("llm_generation_start")
            
            # Warnung bei kritischer Context-Truncation
            if truncation_info.get("critical_truncation", False):
                logger.error("🚨 KRITISCHE TRUNCATION: Antwort-Qualität stark beeinträchtigt!")
            
            # Original QA-Chain mit optimiertem Context ausführen
            result = self.qa_chain.invoke(query_dict)
            self.performance_profiler.log_stage("llm_generation_complete")
            
            # Stage 4: Token-Tracking und Finalisierung
            self.performance_profiler.log_stage("finalization_start")
            
            token_usage = token_tracker.track_request(
                question=question,
                answer=result['result'],
                context=optimized_context,
                model_name=self.model_name
            )
            
            # Performance-Profiling beenden
            performance_data = self.performance_profiler.end_profiling()
            
            # Umfassende Ergebnis-Zusammenstellung
            result.update({
                'token_usage': token_usage,
                'token_info': token_tracker.format_token_info(token_usage),
                'context_info': self._format_context_info(truncation_info),
                'performance_profile': performance_data,
                'debug_info': debug_info if enable_debug else {},
                'model_info': {
                    'name': self.model_name,
                    'limits': self.context_manager.context_limits
                }
            })
            
            # Erweiterte Warnungen in Antwort einbauen
            warnings = self._generate_response_warnings(truncation_info, debug_info, performance_data)
            if warnings:
                result['result'] += "\n\n" + "\n".join(warnings)
            
            # Detailliertes Logging
            self._log_comprehensive_debug_info(question, result, truncation_info, debug_info, performance_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei erweiterten QA-Invoke: {str(e)}")
            
            # Performance-Profiling auch bei Fehlern beenden
            performance_data = self.performance_profiler.end_profiling()
            
            # Fallback zu ursprünglichem Verhalten
            result = self.qa_chain.invoke(query_dict)
            
            # Minimales Token-Tracking
            context = ""
            if 'source_documents' in result:
                context = "\n".join([doc.page_content for doc in result['source_documents']])
            
            token_usage = token_tracker.track_request(
                question=question,
                answer=result['result'],
                context=context,
                model_name=self.model_name
            )
            
            result.update({
                'token_usage': token_usage,
                'token_info': token_tracker.format_token_info(token_usage),
                'error': f"Extended debugging failed: {str(e)}",
                'performance_profile': performance_data
            })
            
            return result
    
    def _generate_response_warnings(self, truncation_info: Dict, debug_info: Dict, performance_data: Dict) -> List[str]:
        """Generiert Warnungen für die Benutzer-Antwort"""
        warnings = []
        
        # Context-Truncation-Warnungen
        if truncation_info.get("truncated", False):
            used_docs = truncation_info.get("used_docs", 0)
            total_docs = truncation_info.get("total_docs", 0)
            
            if truncation_info.get("critical_truncation", False):
                warnings.append(f"🚨 **KRITISCHE WARNUNG**: Nur {used_docs} von {total_docs} relevanten Dokumenten konnten aufgrund von Token-Limits berücksichtigt werden. Die Antwort ist möglicherweise unvollständig.")
            else:
                warnings.append(f"⚠️ **Token-Limit-Warnung**: {used_docs} von {total_docs} relevanten Dokumenten verwendet.")
        
        # Performance-Warnungen
        total_duration = performance_data.get("total_duration", 0)
        if total_duration > 30:
            warnings.append(f"⏱️ **Performance-Warnung**: Sehr lange Antwortzeit ({total_duration:.1f}s). System-Optimierung empfohlen.")
        
        # Retrieval-Qualitäts-Warnungen
        if debug_info.get("retrieval_analysis"):
            sim_analysis = debug_info["retrieval_analysis"].get("similarity_analysis", {})
            quality_ratio = sim_analysis.get("quality_ratio", 1)
            if quality_ratio < 0.3:
                warnings.append("🔍 **Retrieval-Warnung**: Niedrige Dokumenten-Qualität. Möglicherweise sind die gefundenen Dokumente nicht optimal relevant.")
        
        return warnings
    
    def _log_comprehensive_debug_info(self, question: str, result: Dict, truncation_info: Dict, 
                                    debug_info: Dict, performance_data: Dict):
        """Loggt umfassende Debug-Informationen"""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE RAG DEBUG REPORT")
        logger.info("=" * 80)
        
        # Basis-Informationen
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        logger.info(f"Answer Length: {len(result.get('result', ''))} chars")
        
        # Performance-Übersicht
        total_time = performance_data.get("total_duration", 0)
        logger.info(f"Total Processing Time: {total_time:.2f}s")
        
        # Token-Informationen
        token_usage = result.get("token_usage", {})
        logger.info(f"Token Usage - Input: {token_usage.input_tokens}, "
                    f"Output: {token_usage.output_tokens}, "
                    f"Total: {token_usage.total_tokens}")
        
        # Context-Informationen
        logger.info(f"Context Stats - Used Docs: {truncation_info.get('used_docs', 0)}/{truncation_info.get('total_docs', 0)}")
        logger.info(f"Context Tokens: {truncation_info.get('context_tokens', 0)}/{truncation_info.get('available_tokens', 0)}")
        
        if truncation_info.get("truncated", False):
            logger.warning(f"Context Truncation: {truncation_info.get('unused_docs', 0)} documents ignored")
            efficiency = truncation_info.get('token_utilization', {}).get('efficiency', 0)
            logger.warning(f"Token Efficiency: {efficiency:.1f}%")
        
        # Performance-Bottlenecks
        bottlenecks = performance_data.get("performance_analysis", {}).get("bottlenecks", [])
        if bottlenecks:
            logger.warning("Performance Bottlenecks:")
            for bottleneck in bottlenecks:
                logger.warning(f"  - {bottleneck}")
        
        # Retrieval-Qualität
        if debug_info.get("retrieval_analysis"):
            retrieval = debug_info["retrieval_analysis"]
            sim_analysis = retrieval.get("similarity_analysis", {})
            if sim_analysis:
                avg_sim = sim_analysis.get("avg_similarity", 0)
                quality_ratio = sim_analysis.get("quality_ratio", 0)
                logger.info(f"Retrieval Quality - Avg Similarity: {avg_sim:.3f}, Quality Ratio: {quality_ratio:.1%}")
        
        # Empfehlungen
        all_recommendations = []
        if debug_info.get("retrieval_analysis", {}).get("recommendations"):
            all_recommendations.extend(debug_info["retrieval_analysis"]["recommendations"])
        if debug_info.get("chunk_analysis", {}).get("recommendations"):
            all_recommendations.extend(debug_info["chunk_analysis"]["recommendations"])
        if performance_data.get("performance_analysis", {}).get("recommendations"):
            all_recommendations.extend(performance_data["performance_analysis"]["recommendations"])
        
        if all_recommendations:
            logger.info("RECOMMENDATIONS:")
            for rec in all_recommendations[:5]:  # Top 5 Empfehlungen
                logger.info(f"  💡 {rec}")
        
        logger.info("=" * 80)
    
    def _format_context_info(self, truncation_info: Dict) -> str:
        """Formatiert erweiterte Context-Informationen für die Anzeige"""
        used_docs = truncation_info.get("used_docs", 0)
        total_docs = truncation_info.get("total_docs", 0)
        context_tokens = truncation_info.get("context_tokens", 0)
        available_tokens = truncation_info.get("available_tokens", 0)
        processing_time = truncation_info.get("processing_time", 0)
        
        if not truncation_info.get("truncated", False):
            return f"""📄 **Context-Info**:
• Dokumente: {used_docs} vollständig verwendet
• Token: {context_tokens:,} von {available_tokens:,} verfügbar
• Verarbeitungszeit: {processing_time:.3f}s
• Effizienz: {truncation_info.get('token_utilization', {}).get('efficiency', 0):.1f}%"""
        else:
            unused_docs = truncation_info.get("unused_docs", 0)
            wasted_tokens = truncation_info.get('token_utilization', {}).get('wasted', 0)
            
            return f"""📄 **Context-Info (angepasst)**:
• Verwendet: {used_docs}/{total_docs} Dokumente
• Ignoriert: {unused_docs} Dokumente ⚠️
• Token: {context_tokens:,}/{available_tokens:,} ({wasted_tokens:,} ungenutzt)
• Verarbeitungszeit: {processing_time:.3f}s
• Effizienz: {truncation_info.get('token_utilization', {}).get('efficiency', 0):.1f}%
• ⚠️ Context wurde aufgrund von Token-Limits angepasst"""

def build_qa_chain(vectorstore, model_name, chain_type="stuff", enable_debug=True):
    """Build a question-answering chain with comprehensive debugging capabilities
    
    Args:
        vectorstore: Der Vektorspeicher mit Dokumenten
        model_name: Name des zu verwendenden Ollama-Modells
        chain_type: Art der Chain ('stuff', 'map_reduce', 'refine', 'map_rerank')
        enable_debug: Aktiviert erweiterte Debugging-Funktionen
    """
    if not check_ollama_connection():
        raise ConnectionError(f"Cannot connect to Ollama at {config.get_ollama_base_url()}")
    
    # Modell-Konfiguration loggen
    config.log_model_selection(model_name)
    
    # Chain-Typ validieren und ggf. anpassen basierend auf Vektorspeicher-Größe
    recommended_chain = config.get_recommended_chain_type(model_name)
    
    # Erweiterte Chain-Typ-Logik für große Dokumente
    try:
        # Versuche Vektorspeicher-Größe zu ermitteln
        if hasattr(vectorstore, '_collection'):
            doc_count = vectorstore._collection.count()
        else:
            # Fallback: Sample-Query
            sample_docs = vectorstore.similarity_search("test", k=1)
            doc_count = len(sample_docs) * 10  # Große Schätzung
        
        # Automatische Chain-Typ-Anpassung für große Dokumente
        if doc_count > 100 and chain_type == "stuff":
            logger.warning(f"Große Dokumentenbasis ({doc_count} Chunks) detected. "
                          f"Chain-Typ '{chain_type}' möglicherweise nicht optimal.")
            if recommended_chain != "stuff":
                logger.info(f"Empfehlung: Verwenden Sie '{recommended_chain}' für bessere Performance")
        
    except Exception as e:
        logger.warning(f"Konnte Vektorspeicher-Größe nicht ermitteln: {str(e)}")
        doc_count = "unknown"
    
    if chain_type == "stuff" and recommended_chain != "stuff":
        logger.warning(f"Chain-Typ '{chain_type}' möglicherweise nicht optimal für {model_name}. "
                      f"Empfohlen: '{recommended_chain}'")
    
    llm = OllamaLLM(
        model=model_name,
        base_url=config.get_ollama_base_url()
    )
    
    # Verwende das angepasste Prompt-Template
    prompt = get_prompt_template()
    
    # Optimale Retriever-Konfiguration aus Config mit Debug-Erweiterung
    search_k = config.get_optimal_retrieval_k(model_name)
    
    # Erweiterte Retriever-Konfiguration für große Dokumente
    if isinstance(doc_count, int) and doc_count > 500:
        # Für sehr große Dokumentenbasen: Erhöhe k-Wert
        enhanced_k = min(search_k * 2, 25)  # Maximal 25 Dokumente
        logger.info(f"Große Dokumentenbasis detected: Erhöhe k von {search_k} auf {enhanced_k}")
        search_k = enhanced_k
    
    logger.info(f"Retriever konfiguriert: k={search_k} Dokumente")
    
    # Erweiterte Retriever-Konfiguration mit Similarity-Threshold
    retriever_kwargs = {"k": search_k}
    
    # Für unterstützte Vektorspeicher: Similarity-Threshold hinzufügen
    if hasattr(vectorstore, 'similarity_search_with_score'):
        try:
            test_docs = vectorstore.similarity_search_with_score("test", k=1)
            if test_docs:
                logger.info("Similarity-Score verfügbar, aber 'score_threshold' wird von Chroma nicht unterstützt.")
        except TypeError as e:
            logger.warning(f"'score_threshold' wird nicht unterstützt und wurde daher nicht gesetzt: {e}")
                
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=vectorstore.as_retriever(search_kwargs=retriever_kwargs),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt} if chain_type == "stuff" else {}
        )
    
    # Wrappen mit erweiterten Token-Tracking und Debug-Funktionen
    logger.info(f"QA-Chain erfolgreich erstellt: {chain_type} Chain mit {model_name}")
    if enable_debug:
        logger.info("🔍 Erweiterte Debug-Funktionen aktiviert")
    
    enhanced_qa = TokenTrackingQA(qa, model_name)
    
    # Initial-Analyse des Vektorspeichers (nur wenn Debug aktiviert)
    if enable_debug:
        try:
            initial_analysis = enhanced_qa.chunk_analyzer.analyze_chunks(vectorstore, sample_size=20)
            logger.info(f"Vektorspeicher-Analyse: {initial_analysis.get('chunk_count', 'N/A')} Chunks")
            
            # Wichtige Warnungen sofort loggen
            recommendations = initial_analysis.get('recommendations', [])
            for rec in recommendations[:3]:  # Top 3 kritische Empfehlungen
                logger.warning(f"INIT-WARNING: {rec}")
                
        except Exception as e:
            logger.warning(f"Initial-Analyse fehlgeschlagen: {str(e)}")
    
    return enhanced_qa

def get_chain_type_description(chain_type):
    """Gibt eine erweiterte Beschreibung für jeden Chain-Typ zurück"""
    descriptions = {
        "stuff": """Standard für kleine-mittlere Dokumente
        - Fügt alle relevanten Dokumente in einen einzigen Prompt ein
        - Schnell und effizient für <100 Chunks
        - Problem bei Token-Limits mit großen Dokumenten
        - Optimal für: Einzeldokumente, kurze Texte, schnelle Antworten""",
        
        "map_reduce": """Für große Dokumentenbasen optimiert
        - Verarbeitet jedes Dokument einzeln, dann Zusammenfassung
        - Skaliert gut mit Dokumentenanzahl
        - Höhere Latenz, aber keine Token-Limit-Probleme
        - Optimal für: >100 Chunks, große Dateien, umfassende Analysen""",
        
        "refine": """Iterativer Verbesserungsansatz
        - Verfeinert die Antwort schrittweise mit jedem Dokument
        - Gute Qualität bei sequenziellen Informationen
        - Moderate Performance bei mittleren Dokumentenmengen
        - Optimal für: Chronologische Texte, aufbauende Informationen""",
        
        "map_rerank": """Qualitätsorientierte Bewertung
        - Bewertet und ordnet Antworten nach Relevanz
        - Höchste Qualität, aber auch höchste Kosten
        - Geeignet für kritische Anwendungen
        - Optimal für: Präzise Antworten, Fakten-Checking, hohe Qualitätsanforderungen"""
    }
    return descriptions.get(chain_type, "Keine Beschreibung verfügbar")

def suggest_optimal_chain_type(model_name, document_count=None, file_size_mb=None):
    """
    Erweiterte Chain-Typ-Empfehlung basierend auf mehreren Faktoren
    
    Args:
        model_name: Name des Modells
        document_count: Anzahl der Chunks/Dokumente
        file_size_mb: Größe der ursprünglichen Datei in MB
        
    Returns:
        tuple: (recommended_chain, detailed_explanation, confidence_score)
    """
    model_info = config.get_model_info(model_name)
    base_recommendation = model_info["recommended_chain"]
    token_limit = model_info["token_limit"]
    
    # Konfidenz-Score und detaillierte Analyse
    factors = []
    confidence = 0.7  # Basis-Konfidenz
    
    explanation = f"Basis-Empfehlung für {model_info['display_name']} ({token_limit:,} Token): {base_recommendation}"
    
    # Dokument-Anzahl-Faktor
    if document_count:
        if document_count > 200:
            if base_recommendation == "stuff":
                recommended_chain = "map_reduce"
                factors.append(f"Sehr viele Chunks ({document_count}) -> map_reduce")
                confidence += 0.2
            else:
                factors.append(f"Chunk-Anzahl ({document_count}) bestätigt {base_recommendation}")
                confidence += 0.1
        elif document_count > 50 and token_limit < 4096:
            if base_recommendation == "stuff":
                recommended_chain = "map_reduce"
                factors.append(f"Mittlere Chunk-Anzahl ({document_count}) + kleines Token-Limit -> map_reduce")
                confidence += 0.15
            else:
                recommended_chain = base_recommendation
        else:
            recommended_chain = base_recommendation
            factors.append(f"Chunk-Anzahl ({document_count}) passt zu {base_recommendation}")
    else:
        recommended_chain = base_recommendation
    
    # Datei-Größen-Faktor
    if file_size_mb:
        if file_size_mb > 5:
            if recommended_chain == "stuff":
                recommended_chain = "map_reduce"
                factors.append(f"Große Datei ({file_size_mb}MB) -> map_reduce empfohlen")
                confidence += 0.2
            else:
                factors.append(f"Datei-Größe ({file_size_mb}MB) bestätigt {recommended_chain}")
        elif file_size_mb > 1 and token_limit < 4096:
            factors.append(f"Mittlere Datei ({file_size_mb}MB) + begrenztes Token-Limit")
            if recommended_chain == "stuff":
                recommended_chain = "map_reduce" 
                confidence += 0.1
    
    # Modell-spezifische Anpassungen
    if "phi" in model_name.lower() or "mistral" in model_name.lower():
        if document_count and document_count > 30:
            recommended_chain = "map_reduce"
            factors.append("Kleinere Modelle bevorzugen map_reduce bei vielen Dokumenten")
            confidence += 0.1
    
    # Detaillierte Erklärung zusammenstellen
    detailed_explanation = f"""
🎯 **Empfehlung**: {recommended_chain} (Konfidenz: {confidence:.1%})

📊 **Analyse-Faktoren**:
""" + "\n".join([f"   • {factor}" for factor in factors])
    
    if file_size_mb and file_size_mb > 6:
        detailed_explanation += f"""

⚠️ **Spezielle Warnung für große Datei ({file_size_mb}MB)**:
   • 'stuff' Chain wird wahrscheinlich Token-Limits überschreiten
   • 'map_reduce' ist praktisch die einzige skalierbare Option
   • Erwägen Sie zusätzlich feineres Chunking"""
    
    return recommended_chain, detailed_explanation, confidence

def debug_retrieval_quality(vectorstore, test_questions: List[str], k_values: List[int] = None) -> Dict:
    """
    Umfassende Analyse der Retrieval-Qualität mit Test-Fragen
    
    Args:
        vectorstore: Zu testender Vektorspeicher
        test_questions: Liste von Test-Fragen
        k_values: Verschiedene k-Werte zum Testen
        
    Returns:
        Dict mit detaillierter Retrieval-Analyse
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 15, 20]
    
    debugger = DocumentRetrievalDebugger()
    results = {
        "test_questions": len(test_questions),
        "k_values_tested": k_values,
        "individual_results": [],
        "aggregate_analysis": {},
        "recommendations": []
    }
    
    all_similarities = []
    all_k_performance = {k: [] for k in k_values}
    
    for i, question in enumerate(test_questions):
        logger.info(f"Testing question {i+1}/{len(test_questions)}: {question[:50]}...")
        
        # Standard-Retrieval für diese Frage
        docs = vectorstore.similarity_search(question, k=10)
        
        # Detaillierte Analyse
        analysis = debugger.analyze_retrieval(question, docs, vectorstore, k_values)
        results["individual_results"].append(analysis)
        
        # Aggregate-Daten sammeln
        sim_analysis = analysis.get("similarity_analysis", {})
        if "avg_similarity" in sim_analysis:
            all_similarities.append(sim_analysis["avg_similarity"])
        
        # K-Wert-Performance sammeln
        k_comparison = analysis.get("k_value_comparison", {})
        for k in k_values:
            if k in k_comparison and "avg_similarity" in k_comparison[k]:
                all_k_performance[k].append(k_comparison[k]["avg_similarity"])
    
    # Aggregate-Analyse
    if all_similarities:
        results["aggregate_analysis"] = {
            "avg_similarity_across_questions": statistics.mean(all_similarities),
            "similarity_std": statistics.stdev(all_similarities) if len(all_similarities) > 1 else 0,
            "min_similarity": min(all_similarities),
            "max_similarity": max(all_similarities)
        }
    
    # Beste k-Werte identifizieren
    k_performance_summary = {}
    for k, similarities in all_k_performance.items():
        if similarities:
            k_performance_summary[k] = {
                "avg_similarity": statistics.mean(similarities),
                "consistency": 1 - (statistics.stdev(similarities) if len(similarities) > 1 else 0)
            }
    
    if k_performance_summary:
        best_k = max(k_performance_summary.keys(), 
                    key=lambda k: k_performance_summary[k]["avg_similarity"])
        results["optimal_k"] = best_k
        results["k_performance_summary"] = k_performance_summary
    
    # System-weite Empfehlungen
    recommendations = []
    
    if all_similarities:
        avg_quality = statistics.mean(all_similarities)
        if avg_quality < 0.5:
            recommendations.append("🚨 KRITISCH: Sehr niedrige durchschnittliche Similarity (<0.5)")
            recommendations.append("💡 Lösungsansätze: Andere Embedding-Modelle, verbessertes Chunking, Frage-Umformulierung")
        elif avg_quality < 0.7:
            recommendations.append("⚠️ Niedrige Retrieval-Qualität - Optimierung empfohlen")
    
    if "optimal_k" in results:
        optimal_k = results["optimal_k"]
        recommendations.append(f"🎯 Optimaler k-Wert für dieses System: {optimal_k}")
    
    results["recommendations"] = recommendations
    results["timestamp"] = datetime.now().isoformat()
    
    return results