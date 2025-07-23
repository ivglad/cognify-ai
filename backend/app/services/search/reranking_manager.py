"""
Reranking manager that selects and coordinates different reranking methods.
"""
import logging
from typing import List, Dict, Any, Optional, Union
import time
from enum import Enum

import trio

from app.services.search.reranker import LLMReranker
from app.services.search.cross_encoder_reranker import CrossEncoderReranker
from app.core.config import settings

logger = logging.getLogger(__name__)


class RerankingMethod(Enum):
    """Available reranking methods."""
    LLM = "llm"
    CROSS_ENCODER = "cross_encoder"
    HYBRID = "hybrid"
    AUTO = "auto"


class QueryType(Enum):
    """Query type classification for method selection."""
    GENERAL = "general"
    TECHNICAL = "technical"
    FACTUAL = "factual"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


class RerankingManager:
    """
    Manager for coordinating different reranking methods and model selection.
    """
    
    def __init__(self):
        """Initialize reranking manager."""
        self.llm_reranker = LLMReranker()
        self.cross_encoder_reranker = CrossEncoderReranker()
        
        self._initialized = False
        
        # Method selection rules based on query characteristics
        self.method_selection_rules = {
            QueryType.TECHNICAL: RerankingMethod.CROSS_ENCODER,
            QueryType.SEMANTIC: RerankingMethod.CROSS_ENCODER,
            QueryType.FACTUAL: RerankingMethod.LLM,
            QueryType.GENERAL: RerankingMethod.HYBRID,
            QueryType.KEYWORD: RerankingMethod.LLM
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_reranks': 0,
            'method_usage': {
                'llm': 0,
                'cross_encoder': 0,
                'hybrid': 0
            },
            'avg_times': {
                'llm': 0.0,
                'cross_encoder': 0.0,
                'hybrid': 0.0
            },
            'fallback_count': 0
        }
    
    async def initialize(self):
        """Initialize all reranking components."""
        if self._initialized:
            return
        
        try:
            # Initialize cross-encoder (LLM reranker initializes on demand)
            await self.cross_encoder_reranker.initialize()
            
            self._initialized = True
            logger.info("RerankingManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RerankingManager: {e}")
            self._initialized = False
    
    async def rerank_results(self,
                           query: str,
                           search_results: List[Dict[str, Any]],
                           method: Union[RerankingMethod, str] = RerankingMethod.AUTO,
                           rerank_type: str = "general",
                           top_k: Optional[int] = None,
                           fallback_enabled: bool = True) -> Dict[str, Any]:
        """
        Rerank search results using the specified or auto-selected method.
        
        Args:
            query: Search query
            search_results: List of search results to rerank
            method: Reranking method to use
            rerank_type: Type of reranking for LLM method
            top_k: Number of top results to return
            fallback_enabled: Whether to fallback to other methods on failure
            
        Returns:
            Reranked results with metadata
        """
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            if not search_results or not query.strip():
                return {
                    'results': search_results,
                    'method_used': 'none',
                    'rerank_time': 0.0,
                    'success': True
                }
            
            # Convert string method to enum
            if isinstance(method, str):
                try:
                    method = RerankingMethod(method.lower())
                except ValueError:
                    logger.warning(f"Unknown reranking method: {method}, using AUTO")
                    method = RerankingMethod.AUTO
            
            # Auto-select method if needed
            if method == RerankingMethod.AUTO:
                method = await self._select_optimal_method(query, search_results)
            
            # Perform reranking
            reranked_results, actual_method = await self._execute_reranking(
                query, search_results, method, rerank_type, top_k, fallback_enabled
            )
            
            # Update performance stats
            rerank_time = time.time() - start_time
            self._update_performance_stats(actual_method, rerank_time)
            
            return {
                'results': reranked_results,
                'method_used': actual_method.value,
                'rerank_time': rerank_time,
                'success': True,
                'query_type': await self._classify_query_type(query),
                'original_count': len(search_results),
                'reranked_count': len(reranked_results)
            }
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return {
                'results': search_results[:top_k] if top_k else search_results,
                'method_used': 'fallback',
                'rerank_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    async def _select_optimal_method(self,
                                   query: str,
                                   search_results: List[Dict[str, Any]]) -> RerankingMethod:
        """
        Automatically select the optimal reranking method based on query and results.
        """
        try:
            # Classify query type
            query_type = await self._classify_query_type(query)
            
            # Get method from rules
            selected_method = self.method_selection_rules.get(query_type, RerankingMethod.HYBRID)
            
            # Check if cross-encoder is available for cross-encoder methods
            if selected_method == RerankingMethod.CROSS_ENCODER:
                if not self.cross_encoder_reranker._initialized:
                    logger.warning("Cross-encoder not available, falling back to LLM")
                    selected_method = RerankingMethod.LLM
            
            # Consider result count for method selection
            result_count = len(search_results)
            if result_count > 20 and selected_method == RerankingMethod.CROSS_ENCODER:
                # Cross-encoder might be slow for many results
                selected_method = RerankingMethod.HYBRID
            elif result_count <= 5:
                # For few results, LLM might be more thorough
                selected_method = RerankingMethod.LLM
            
            logger.debug(f"Auto-selected method: {selected_method.value} for query type: {query_type.value}")
            return selected_method
            
        except Exception as e:
            logger.error(f"Method selection failed: {e}")
            return RerankingMethod.LLM  # Safe fallback
    
    async def _classify_query_type(self, query: str) -> QueryType:
        """
        Classify query type based on content and structure.
        """
        try:
            query_lower = query.lower()
            
            # Technical keywords
            technical_keywords = [
                'api', 'function', 'method', 'class', 'algorithm', 'implementation',
                'code', 'programming', 'software', 'system', 'architecture',
                'database', 'sql', 'python', 'javascript', 'java', 'c++',
                'framework', 'library', 'module', 'package', 'import'
            ]
            
            # Factual keywords
            factual_keywords = [
                'what is', 'who is', 'when did', 'where is', 'how many',
                'definition', 'meaning', 'explain', 'describe', 'history',
                'fact', 'statistics', 'data', 'research', 'study'
            ]
            
            # Semantic keywords (meaning-focused)
            semantic_keywords = [
                'similar to', 'like', 'related to', 'compared to', 'difference',
                'relationship', 'connection', 'analogy', 'metaphor', 'concept'
            ]
            
            # Count keyword matches
            technical_score = sum(1 for kw in technical_keywords if kw in query_lower)
            factual_score = sum(1 for kw in factual_keywords if kw in query_lower)
            semantic_score = sum(1 for kw in semantic_keywords if kw in query_lower)
            
            # Determine query type
            if technical_score > 0:
                return QueryType.TECHNICAL
            elif factual_score > 0:
                return QueryType.FACTUAL
            elif semantic_score > 0:
                return QueryType.SEMANTIC
            elif len(query.split()) <= 3:
                return QueryType.KEYWORD
            else:
                return QueryType.GENERAL
                
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            return QueryType.GENERAL
    
    async def _execute_reranking(self,
                               query: str,
                               search_results: List[Dict[str, Any]],
                               method: RerankingMethod,
                               rerank_type: str,
                               top_k: Optional[int],
                               fallback_enabled: bool) -> tuple[List[Dict[str, Any]], RerankingMethod]:
        """
        Execute reranking using the specified method with fallback support.
        """
        try:
            if method == RerankingMethod.LLM:
                results = await self.llm_reranker.rerank_results(
                    query, search_results, rerank_type, top_k
                )
                return results, RerankingMethod.LLM
                
            elif method == RerankingMethod.CROSS_ENCODER:
                if self.cross_encoder_reranker._initialized:
                    results = await self.cross_encoder_reranker.rerank_results(
                        query, search_results, top_k
                    )
                    return results, RerankingMethod.CROSS_ENCODER
                elif fallback_enabled:
                    logger.warning("Cross-encoder not available, falling back to LLM")
                    self.performance_stats['fallback_count'] += 1
                    results = await self.llm_reranker.rerank_results(
                        query, search_results, rerank_type, top_k
                    )
                    return results, RerankingMethod.LLM
                else:
                    raise Exception("Cross-encoder not available and fallback disabled")
                    
            elif method == RerankingMethod.HYBRID:
                return await self._hybrid_reranking(query, search_results, rerank_type, top_k)
                
            else:
                raise ValueError(f"Unknown reranking method: {method}")
                
        except Exception as e:
            if fallback_enabled:
                logger.warning(f"Reranking method {method.value} failed, falling back to LLM: {e}")
                self.performance_stats['fallback_count'] += 1
                results = await self.llm_reranker.rerank_results(
                    query, search_results, rerank_type, top_k
                )
                return results, RerankingMethod.LLM
            else:
                raise
    
    async def _hybrid_reranking(self,
                              query: str,
                              search_results: List[Dict[str, Any]],
                              rerank_type: str,
                              top_k: Optional[int]) -> tuple[List[Dict[str, Any]], RerankingMethod]:
        """
        Perform hybrid reranking using both LLM and cross-encoder methods.
        """
        try:
            # Run both methods concurrently if cross-encoder is available
            if self.cross_encoder_reranker._initialized:
                async with trio.open_nursery() as nursery:
                    llm_results = []
                    cross_encoder_results = []
                    
                    async def run_llm():
                        nonlocal llm_results
                        llm_results = await self.llm_reranker.rerank_results(
                            query, search_results, rerank_type, None  # Get all results
                        )
                    
                    async def run_cross_encoder():
                        nonlocal cross_encoder_results
                        cross_encoder_results = await self.cross_encoder_reranker.rerank_results(
                            query, search_results, None  # Get all results
                        )
                    
                    nursery.start_soon(run_llm)
                    nursery.start_soon(run_cross_encoder)
                
                # Combine results using weighted scoring
                combined_results = await self._combine_reranking_results(
                    llm_results, cross_encoder_results, query
                )
                
                # Apply top_k limit
                if top_k:
                    combined_results = combined_results[:top_k]
                
                return combined_results, RerankingMethod.HYBRID
            
            else:
                # Fallback to LLM only
                logger.warning("Cross-encoder not available for hybrid reranking, using LLM only")
                results = await self.llm_reranker.rerank_results(
                    query, search_results, rerank_type, top_k
                )
                return results, RerankingMethod.LLM
                
        except Exception as e:
            logger.error(f"Hybrid reranking failed: {e}")
            # Fallback to LLM
            results = await self.llm_reranker.rerank_results(
                query, search_results, rerank_type, top_k
            )
            return results, RerankingMethod.LLM
    
    async def _combine_reranking_results(self,
                                       llm_results: List[Dict[str, Any]],
                                       cross_encoder_results: List[Dict[str, Any]],
                                       query: str) -> List[Dict[str, Any]]:
        """
        Combine results from LLM and cross-encoder reranking.
        """
        try:
            # Create lookup for cross-encoder results
            cross_encoder_lookup = {
                result.get('chunk_id', result.get('id', '')): result
                for result in cross_encoder_results
            }
            
            # Create lookup for LLM results
            llm_lookup = {
                result.get('chunk_id', result.get('id', '')): result
                for result in llm_results
            }
            
            # Get all unique chunk IDs
            all_chunk_ids = set(llm_lookup.keys()) | set(cross_encoder_lookup.keys())
            
            combined_results = []
            
            for chunk_id in all_chunk_ids:
                llm_result = llm_lookup.get(chunk_id)
                cross_encoder_result = cross_encoder_lookup.get(chunk_id)
                
                # Use the result with more complete information as base
                base_result = llm_result or cross_encoder_result
                
                # Get scores
                llm_score = llm_result.get('llm_relevance_score', 0.0) if llm_result else 0.0
                cross_encoder_score = cross_encoder_result.get('cross_encoder_score', 0.0) if cross_encoder_result else 0.0
                original_score = base_result.get('original_score', base_result.get('score', 0.0))
                
                # Weighted combination: 40% LLM, 40% cross-encoder, 20% original
                hybrid_score = (
                    0.4 * llm_score +
                    0.4 * cross_encoder_score +
                    0.2 * original_score
                )
                
                # Create combined result
                combined_result = base_result.copy()
                combined_result.update({
                    'score': hybrid_score,
                    'hybrid_score': hybrid_score,
                    'llm_relevance_score': llm_score,
                    'cross_encoder_score': cross_encoder_score,
                    'original_score': original_score,
                    'rerank_method': 'hybrid',
                    'hybrid_weights': {
                        'llm': 0.4,
                        'cross_encoder': 0.4,
                        'original': 0.2
                    }
                })
                
                combined_results.append(combined_result)
            
            # Sort by hybrid score
            combined_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Result combination failed: {e}")
            # Return LLM results as fallback
            return llm_results
    
    def _update_performance_stats(self, method: RerankingMethod, rerank_time: float):
        """Update performance statistics."""
        try:
            self.performance_stats['total_reranks'] += 1
            
            method_key = method.value
            if method_key in self.performance_stats['method_usage']:
                self.performance_stats['method_usage'][method_key] += 1
                
                # Update average time
                current_avg = self.performance_stats['avg_times'][method_key]
                current_count = self.performance_stats['method_usage'][method_key]
                
                new_avg = ((current_avg * (current_count - 1)) + rerank_time) / current_count
                self.performance_stats['avg_times'][method_key] = new_avg
                
        except Exception as e:
            logger.error(f"Performance stats update failed: {e}")
    
    async def compare_reranking_methods(self,
                                      test_queries: List[str],
                                      test_results: List[List[Dict[str, Any]]],
                                      ground_truth: List[List[str]]) -> Dict[str, Any]:
        """
        Compare performance of different reranking methods.
        
        Args:
            test_queries: List of test queries
            test_results: List of search results for each query
            ground_truth: List of relevant result IDs for each query
            
        Returns:
            Comparison results
        """
        try:
            methods_to_test = [RerankingMethod.LLM, RerankingMethod.CROSS_ENCODER, RerankingMethod.HYBRID]
            comparison_results = {}
            
            for method in methods_to_test:
                logger.info(f"Testing reranking method: {method.value}")
                
                total_ndcg = 0.0
                total_time = 0.0
                successful_queries = 0
                
                for query, results, truth in zip(test_queries, test_results, ground_truth):
                    try:
                        start_time = time.time()
                        
                        rerank_result = await self.rerank_results(
                            query, results, method, fallback_enabled=False
                        )
                        
                        if rerank_result['success']:
                            # Calculate NDCG
                            ndcg = self._calculate_ndcg(rerank_result['results'], truth)
                            total_ndcg += ndcg
                            successful_queries += 1
                        
                        total_time += (time.time() - start_time)
                        
                    except Exception as query_error:
                        logger.warning(f"Query failed for method {method.value}: {query_error}")
                
                if successful_queries > 0:
                    comparison_results[method.value] = {
                        'avg_ndcg': total_ndcg / successful_queries,
                        'avg_time': total_time / len(test_queries),
                        'success_rate': successful_queries / len(test_queries),
                        'successful_queries': successful_queries,
                        'total_queries': len(test_queries)
                    }
                else:
                    comparison_results[method.value] = {
                        'error': 'No successful queries',
                        'success_rate': 0.0
                    }
            
            # Find best method
            best_method = None
            best_score = 0.0
            
            for method_name, results in comparison_results.items():
                if 'avg_ndcg' in results and results['avg_ndcg'] > best_score:
                    best_score = results['avg_ndcg']
                    best_method = method_name
            
            return {
                'comparison_results': comparison_results,
                'best_method': best_method,
                'best_score': best_score,
                'test_queries_count': len(test_queries)
            }
            
        except Exception as e:
            logger.error(f"Method comparison failed: {e}")
            return {'error': str(e)}
    
    def _calculate_ndcg(self, results: List[Dict[str, Any]], ground_truth: List[str]) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        try:
            if not results or not ground_truth:
                return 0.0
            
            # Calculate DCG
            dcg = 0.0
            for i, result in enumerate(results):
                result_id = result.get('chunk_id', result.get('id', ''))
                if result_id in ground_truth:
                    dcg += 1.0 / (1 + i)
            
            # Calculate IDCG
            idcg = sum(1.0 / (1 + i) for i in range(min(len(results), len(ground_truth))))
            
            return dcg / idcg if idcg > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive reranking manager statistics."""
        try:
            # Get stats from individual rerankers
            llm_stats = await self.llm_reranker.get_reranker_stats()
            
            cross_encoder_stats = {}
            if self.cross_encoder_reranker._initialized:
                cross_encoder_stats = await self.cross_encoder_reranker.get_model_info()
            
            return {
                'manager_initialized': self._initialized,
                'performance_stats': self.performance_stats,
                'method_selection_rules': {
                    qt.value: method.value for qt, method in self.method_selection_rules.items()
                },
                'available_methods': [method.value for method in RerankingMethod],
                'llm_reranker': llm_stats,
                'cross_encoder_reranker': cross_encoder_stats
            }
            
        except Exception as e:
            logger.error(f"Manager stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
reranking_manager = RerankingManager()