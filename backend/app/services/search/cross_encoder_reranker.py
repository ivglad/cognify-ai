"""
Cross-encoder based reranking system for improved accuracy.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from pathlib import Path

import trio
import numpy as np

from app.core.config import settings
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder based reranking system using specialized models.
    """
    
    def __init__(self,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 max_length: int = 512,
                 batch_size: int = 8,
                 device: str = "cpu",
                 enable_caching: bool = True):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            device: Device to run model on (cpu/cuda)
            enable_caching: Whether to cache results
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.enable_caching = enable_caching
        
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
        self.cache = cache_manager
        
        # Performance tracking
        self.rerank_stats = {
            'total_reranks': 0,
            'avg_rerank_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'model_calls': 0,
            'batch_processing_time': 0.0
        }
        
        # Available models with their characteristics
        self.available_models = {
            "cross-encoder/ms-marco-MiniLM-L-6-v2": {
                "description": "Fast and efficient cross-encoder for general ranking",
                "max_length": 512,
                "performance": "fast",
                "accuracy": "good"
            },
            "cross-encoder/ms-marco-MiniLM-L-12-v2": {
                "description": "Balanced cross-encoder with better accuracy",
                "max_length": 512,
                "performance": "medium",
                "accuracy": "better"
            },
            "cross-encoder/ms-marco-electra-base": {
                "description": "High accuracy cross-encoder based on ELECTRA",
                "max_length": 512,
                "performance": "slow",
                "accuracy": "best"
            },
            "cross-encoder/stsb-roberta-large": {
                "description": "Semantic similarity focused cross-encoder",
                "max_length": 512,
                "performance": "slow",
                "accuracy": "specialized"
            }
        }
    
    async def initialize(self):
        """Initialize the cross-encoder model."""
        if self._initialized:
            return
        
        try:
            # Try to import sentence-transformers
            try:
                from sentence_transformers import CrossEncoder
                
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                
                # Load model in thread to avoid blocking
                def load_model():
                    return CrossEncoder(
                        self.model_name,
                        max_length=self.max_length,
                        device=self.device
                    )
                
                self.model = await trio.to_thread.run_sync(load_model)
                
                self._initialized = True
                logger.info(f"Cross-encoder model {self.model_name} loaded successfully")
                
            except ImportError:
                logger.warning("sentence-transformers not available, cross-encoder reranking disabled")
                self._initialized = False
                
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder model: {e}")
            self._initialized = False
            raise
    
    async def rerank_results(self,
                           query: str,
                           search_results: List[Dict[str, Any]],
                           top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank search results using cross-encoder model.
        
        Args:
            query: Search query
            search_results: List of search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked results with cross-encoder scores
        """
        start_time = time.time()
        
        try:
            if not search_results or not query.strip():
                return search_results
            
            # Initialize model if needed
            if not self._initialized:
                await self.initialize()
            
            if not self._initialized:
                logger.warning("Cross-encoder not available, returning original results")
                return search_results[:top_k] if top_k else search_results
            
            # Check cache
            if self.enable_caching:
                cache_key = self._generate_cache_key(query, search_results)
                cached_results = await self.cache.get(cache_key)
                
                if cached_results:
                    self.rerank_stats['cache_hits'] += 1
                    logger.debug(f"Using cached cross-encoder results for query: {query}")
                    return cached_results[:top_k] if top_k else cached_results
                
                self.rerank_stats['cache_misses'] += 1
            
            # Prepare query-document pairs
            pairs = []
            for result in search_results:
                content = result.get('content', '')
                # Truncate content if too long
                if len(content) > 1000:
                    content = content[:1000] + "..."
                pairs.append([query, content])
            
            # Get cross-encoder scores
            scores = await self._get_cross_encoder_scores(pairs)
            
            # Apply scores to results
            reranked_results = []
            for i, (result, score) in enumerate(zip(search_results, scores)):
                result_copy = result.copy()
                result_copy['cross_encoder_score'] = float(score)
                result_copy['original_score'] = result.get('score', 0.0)
                
                # Combine original and cross-encoder scores
                combined_score = self._combine_scores(
                    result.get('score', 0.0),
                    float(score)
                )
                result_copy['score'] = combined_score
                result_copy['combined_score'] = combined_score
                result_copy['rerank_method'] = 'cross_encoder'
                
                reranked_results.append(result_copy)
            
            # Sort by combined score
            reranked_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Apply top_k limit
            if top_k:
                reranked_results = reranked_results[:top_k]
            
            # Cache results
            if self.enable_caching:
                await self.cache.set(cache_key, reranked_results, ttl=3600)
            
            # Update stats
            rerank_time = time.time() - start_time
            self._update_rerank_stats(rerank_time)
            
            logger.debug(f"Cross-encoder reranked {len(search_results)} results in {rerank_time:.3f}s")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return search_results[:top_k] if top_k else search_results
    
    async def _get_cross_encoder_scores(self, pairs: List[List[str]]) -> List[float]:
        """Get scores from cross-encoder model."""
        try:
            if not self.model:
                return [0.5] * len(pairs)
            
            # Process in batches
            all_scores = []
            
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                
                # Run model prediction in thread
                def predict_batch():
                    return self.model.predict(batch)
                
                batch_scores = await trio.to_thread.run_sync(predict_batch)
                all_scores.extend(batch_scores.tolist())
                
                self.rerank_stats['model_calls'] += 1
            
            # Normalize scores to 0-1 range using sigmoid
            normalized_scores = []
            for score in all_scores:
                normalized_score = 1.0 / (1.0 + np.exp(-score))
                normalized_scores.append(normalized_score)
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            return [0.5] * len(pairs)
    
    def _combine_scores(self, original_score: float, cross_encoder_score: float) -> float:
        """Combine original search score with cross-encoder score."""
        try:
            # Weighted combination: 70% cross-encoder, 30% original
            combined = 0.7 * cross_encoder_score + 0.3 * original_score
            return min(1.0, max(0.0, combined))
            
        except Exception:
            return original_score
    
    def _generate_cache_key(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate cache key for cross-encoder results."""
        try:
            result_ids = [r.get('chunk_id', r.get('id', str(i))) for i, r in enumerate(results)]
            content_hash = hash(f"{query}:{':'.join(result_ids)}:{self.model_name}")
            
            return f"cross_encoder_rerank:{content_hash}"
            
        except Exception:
            return f"cross_encoder_rerank:{hash(query)}:{self.model_name}"
    
    def _update_rerank_stats(self, rerank_time: float):
        """Update reranking performance statistics."""
        try:
            self.rerank_stats['total_reranks'] += 1
            
            # Update average rerank time
            total_time = self.rerank_stats['avg_rerank_time'] * (self.rerank_stats['total_reranks'] - 1)
            self.rerank_stats['avg_rerank_time'] = (total_time + rerank_time) / self.rerank_stats['total_reranks']
            
            self.rerank_stats['batch_processing_time'] += rerank_time
            
        except Exception as e:
            logger.error(f"Rerank stats update failed: {e}")
    
    async def compare_models(self,
                           test_queries: List[str],
                           test_results: List[List[Dict[str, Any]]],
                           ground_truth: List[List[str]],
                           models_to_test: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare performance of different cross-encoder models.
        
        Args:
            test_queries: List of test queries
            test_results: List of search results for each query
            ground_truth: List of relevant result IDs for each query
            models_to_test: List of model names to test (default: all available)
            
        Returns:
            Comparison results
        """
        try:
            if models_to_test is None:
                models_to_test = list(self.available_models.keys())
            
            comparison_results = {}
            
            for model_name in models_to_test:
                logger.info(f"Testing model: {model_name}")
                
                # Create temporary reranker with this model
                temp_reranker = CrossEncoderReranker(
                    model_name=model_name,
                    batch_size=self.batch_size,
                    device=self.device,
                    enable_caching=False  # Disable caching for fair comparison
                )
                
                try:
                    await temp_reranker.initialize()
                    
                    if not temp_reranker._initialized:
                        logger.warning(f"Failed to initialize {model_name}, skipping")
                        continue
                    
                    # Test on all queries
                    total_ndcg = 0.0
                    total_time = 0.0
                    
                    for query, results, truth in zip(test_queries, test_results, ground_truth):
                        start_time = time.time()
                        
                        reranked = await temp_reranker.rerank_results(query, results)
                        
                        # Calculate NDCG
                        ndcg = self._calculate_ndcg(reranked, truth)
                        total_ndcg += ndcg
                        
                        total_time += (time.time() - start_time)
                    
                    avg_ndcg = total_ndcg / len(test_queries)
                    avg_time = total_time / len(test_queries)
                    
                    comparison_results[model_name] = {
                        'avg_ndcg': avg_ndcg,
                        'avg_time': avg_time,
                        'model_info': self.available_models.get(model_name, {}),
                        'initialized': True
                    }
                    
                except Exception as model_error:
                    logger.error(f"Error testing model {model_name}: {model_error}")
                    comparison_results[model_name] = {
                        'error': str(model_error),
                        'initialized': False
                    }
            
            # Find best model
            best_model = None
            best_score = 0.0
            
            for model_name, results in comparison_results.items():
                if results.get('initialized') and results.get('avg_ndcg', 0) > best_score:
                    best_score = results['avg_ndcg']
                    best_model = model_name
            
            return {
                'comparison_results': comparison_results,
                'best_model': best_model,
                'best_score': best_score,
                'test_queries_count': len(test_queries)
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
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
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        return {
            'current_model': self.model_name,
            'initialized': self._initialized,
            'available_models': self.available_models,
            'configuration': {
                'max_length': self.max_length,
                'batch_size': self.batch_size,
                'device': self.device,
                'enable_caching': self.enable_caching
            },
            'performance_stats': self.rerank_stats.copy()
        }
    
    async def switch_model(self, new_model_name: str) -> Dict[str, Any]:
        """
        Switch to a different cross-encoder model.
        
        Args:
            new_model_name: Name of the new model to load
            
        Returns:
            Switch status
        """
        try:
            if new_model_name not in self.available_models:
                return {
                    'success': False,
                    'error': f"Model {new_model_name} not available",
                    'available_models': list(self.available_models.keys())
                }
            
            old_model = self.model_name
            
            # Update model name and reinitialize
            self.model_name = new_model_name
            self.model = None
            self._initialized = False
            
            # Clear cache since we're switching models
            await self._clear_model_cache()
            
            # Initialize new model
            await self.initialize()
            
            if self._initialized:
                logger.info(f"Successfully switched from {old_model} to {new_model_name}")
                return {
                    'success': True,
                    'old_model': old_model,
                    'new_model': new_model_name,
                    'model_info': self.available_models[new_model_name]
                }
            else:
                # Rollback on failure
                self.model_name = old_model
                self.model = None
                self._initialized = False
                
                return {
                    'success': False,
                    'error': f"Failed to initialize {new_model_name}, rolled back to {old_model}"
                }
                
        except Exception as e:
            logger.error(f"Model switching failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _clear_model_cache(self):
        """Clear cache entries for current model."""
        try:
            cache_keys = await self.cache.keys(f"cross_encoder_rerank:*")
            for key in cache_keys:
                await self.cache.delete(key)
            
            # Reset cache stats
            self.rerank_stats['cache_hits'] = 0
            self.rerank_stats['cache_misses'] = 0
            
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")


# Global instance
cross_encoder_reranker = CrossEncoderReranker()