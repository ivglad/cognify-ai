"""
Enhanced dense retrieval system with advanced vector search capabilities.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import time

import trio

from app.services.embeddings.embedding_service import embedding_service
from app.services.embeddings.vector_store import vector_store
from app.services.nlp.rag_tokenizer import rag_tokenizer
from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Enhanced dense retrieval system with advanced vector search and optimization.
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.7,
                 max_results: int = 100,
                 enable_query_expansion: bool = True,
                 enable_reranking: bool = True):
        """
        Initialize dense retriever.
        
        Args:
            similarity_threshold: Minimum similarity threshold for results
            max_results: Maximum number of results to return
            enable_query_expansion: Whether to expand queries with similar terms
            enable_reranking: Whether to apply semantic reranking
        """
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self.enable_query_expansion = enable_query_expansion
        self.enable_reranking = enable_reranking
        
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.tokenizer = rag_tokenizer
        self.cache = cache_manager
        
        # Query expansion cache
        self.expansion_cache = {}
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'avg_search_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def search(self, 
                    query: str,
                    filters: Optional[Dict[str, Any]] = None,
                    top_k: Optional[int] = None,
                    include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Perform dense vector search.
        
        Args:
            query: Search query
            filters: Optional filters to apply
            top_k: Number of results to return
            include_metadata: Whether to include chunk metadata
            
        Returns:
            List of search results with similarity scores
        """
        start_time = time.time()
        
        try:
            if not query or not query.strip():
                return []
            
            # Initialize services
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            top_k = top_k or self.max_results
            
            # Check cache first
            cache_key = self._generate_cache_key(query, filters, top_k)
            cached_results = await self.cache.get(cache_key)
            
            if cached_results:
                self.search_stats['cache_hits'] += 1
                logger.debug(f"Using cached results for query: {query}")
                return cached_results
            
            self.search_stats['cache_misses'] += 1
            
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(query)
            
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []
            
            # Perform vector search
            search_results = await self._vector_search(query_embedding, filters, top_k * 2)  # Get more for reranking
            
            # Apply query expansion if enabled
            if self.enable_query_expansion and len(search_results) < top_k:
                expanded_results = await self._expand_and_search(query, query_embedding, filters, top_k)
                search_results.extend(expanded_results)
            
            # Remove duplicates and sort by similarity
            search_results = self._deduplicate_results(search_results)
            search_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Apply similarity threshold
            filtered_results = [r for r in search_results if r['similarity'] >= self.similarity_threshold]
            
            # Apply semantic reranking if enabled
            if self.enable_reranking and len(filtered_results) > 1:
                filtered_results = await self._semantic_rerank(query, filtered_results)
            
            # Limit results
            final_results = filtered_results[:top_k]
            
            # Add metadata if requested
            if include_metadata:
                final_results = await self._enrich_with_metadata(final_results)
            
            # Cache results
            await self.cache.set(cache_key, final_results, ttl=3600)  # 1 hour
            
            # Update stats
            search_time = time.time() - start_time
            self._update_search_stats(search_time)
            
            logger.debug(f"Dense search returned {len(final_results)} results for query: {query}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    async def _generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for search query with optimization."""
        try:
            # Check embedding cache
            embedding_cache_key = f"query_embedding:{hash(query)}"
            cached_embedding = await self.cache.get(embedding_cache_key)
            
            if cached_embedding is not None:
                return np.array(cached_embedding)
            
            # Generate new embedding
            embedding = await self.embedding_service.generate_embedding(query)
            
            if embedding is not None:
                # Cache embedding
                await self.cache.set(embedding_cache_key, embedding.tolist(), ttl=86400)  # 24 hours
                return embedding
            
            return None
            
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            return None
    
    async def _vector_search(self, 
                           query_embedding: np.ndarray,
                           filters: Optional[Dict[str, Any]] = None,
                           top_k: int = 100) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        try:
            # Search in vector store
            search_results = await self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            # Convert to standard format
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    'chunk_id': result.get('id', ''),
                    'document_id': result.get('metadata', {}).get('document_id', ''),
                    'content': result.get('metadata', {}).get('content', ''),
                    'similarity': float(result.get('score', 0.0)),
                    'method': 'dense',
                    'metadata': result.get('metadata', {})
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _expand_and_search(self, 
                               original_query: str,
                               query_embedding: np.ndarray,
                               filters: Optional[Dict[str, Any]] = None,
                               top_k: int = 100) -> List[Dict[str, Any]]:
        """Expand query and perform additional search."""
        try:
            # Generate query expansions
            expanded_queries = await self._generate_query_expansions(original_query)
            
            if not expanded_queries:
                return []
            
            all_results = []
            
            # Search with each expanded query
            for expanded_query in expanded_queries:
                # Generate embedding for expanded query
                expanded_embedding = await self._generate_query_embedding(expanded_query)
                
                if expanded_embedding is not None:
                    # Perform search
                    expanded_results = await self._vector_search(expanded_embedding, filters, top_k // 2)
                    
                    # Mark as expanded results
                    for result in expanded_results:
                        result['expanded_query'] = expanded_query
                        result['is_expanded'] = True
                        result['similarity'] *= 0.9  # Slight penalty for expanded results
                    
                    all_results.extend(expanded_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Query expansion search failed: {e}")
            return []
    
    async def _generate_query_expansions(self, query: str) -> List[str]:
        """Generate expanded versions of the query."""
        try:
            # Check cache
            if query in self.expansion_cache:
                return self.expansion_cache[query]
            
            expansions = []
            
            # Tokenize query
            tokens = await self.tokenizer.tokenize_text(query, remove_stopwords=False, stem_words=False)
            
            if not tokens:
                return []
            
            # Generate n-grams for expansion
            if len(tokens) > 1:
                # Add bigrams
                for i in range(len(tokens) - 1):
                    bigram = f"{tokens[i]} {tokens[i + 1]}"
                    if bigram != query:
                        expansions.append(bigram)
            
            # Add individual important terms
            important_tokens = [token for token in tokens if len(token) > 3]
            for token in important_tokens[:3]:  # Limit to top 3
                if token != query:
                    expansions.append(token)
            
            # Add stemmed versions
            stemmed_tokens = await self.tokenizer.tokenize_text(query, remove_stopwords=False, stem_words=True)
            if stemmed_tokens != tokens:
                stemmed_query = ' '.join(stemmed_tokens)
                if stemmed_query != query:
                    expansions.append(stemmed_query)
            
            # Cache expansions
            self.expansion_cache[query] = expansions[:5]  # Limit to 5 expansions
            
            return expansions[:5]
            
        except Exception as e:
            logger.error(f"Query expansion generation failed: {e}")
            return []
    
    async def _semantic_rerank(self, 
                             query: str, 
                             results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply semantic reranking to improve result quality."""
        try:
            if len(results) <= 1:
                return results
            
            # Generate query embedding if not cached
            query_embedding = await self._generate_query_embedding(query)
            
            if query_embedding is None:
                return results
            
            # Calculate semantic similarity for reranking
            reranked_results = []
            
            for result in results:
                content = result.get('content', '')
                
                if content:
                    # Generate content embedding
                    content_embedding = await self.embedding_service.generate_embedding(content)
                    
                    if content_embedding is not None:
                        # Calculate semantic similarity
                        semantic_similarity = np.dot(query_embedding, content_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)
                        )
                        
                        # Combine with original similarity
                        original_similarity = result.get('similarity', 0.0)
                        combined_similarity = (original_similarity * 0.7 + semantic_similarity * 0.3)
                        
                        result['semantic_similarity'] = float(semantic_similarity)
                        result['combined_similarity'] = float(combined_similarity)
                        result['similarity'] = float(combined_similarity)  # Update main similarity
                
                reranked_results.append(result)
            
            # Sort by combined similarity
            reranked_results.sort(key=lambda x: x.get('combined_similarity', x.get('similarity', 0)), reverse=True)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Semantic reranking failed: {e}")
            return results
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on chunk_id."""
        try:
            seen_chunks = set()
            deduplicated = []
            
            for result in results:
                chunk_id = result.get('chunk_id', '')
                
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    deduplicated.append(result)
                elif not chunk_id:
                    # Keep results without chunk_id
                    deduplicated.append(result)
            
            return deduplicated
            
        except Exception as e:
            logger.error(f"Result deduplication failed: {e}")
            return results
    
    async def _enrich_with_metadata(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich results with additional metadata."""
        try:
            enriched_results = []
            
            for result in results:
                # Add relevance metrics
                result['relevance_metrics'] = await self._calculate_relevance_metrics(result)
                
                # Add search method info
                result['search_method'] = 'dense'
                result['embedding_model'] = self.embedding_service.model_name if hasattr(self.embedding_service, 'model_name') else 'unknown'
                
                # Add timestamp
                result['retrieved_at'] = time.time()
                
                enriched_results.append(result)
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Result enrichment failed: {e}")
            return results
    
    async def _calculate_relevance_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate relevance metrics for a result."""
        try:
            content = result.get('content', '')
            similarity = result.get('similarity', 0.0)
            
            metrics = {
                'similarity_score': similarity,
                'content_length': len(content),
                'is_expanded_result': result.get('is_expanded', False)
            }
            
            # Add semantic similarity if available
            if 'semantic_similarity' in result:
                metrics['semantic_similarity'] = result['semantic_similarity']
            
            # Add combined similarity if available
            if 'combined_similarity' in result:
                metrics['combined_similarity'] = result['combined_similarity']
            
            # Calculate confidence based on similarity
            if similarity >= 0.9:
                metrics['confidence'] = 'high'
            elif similarity >= 0.7:
                metrics['confidence'] = 'medium'
            else:
                metrics['confidence'] = 'low'
            
            return metrics
            
        except Exception as e:
            logger.error(f"Relevance metrics calculation failed: {e}")
            return {}
    
    def _generate_cache_key(self, 
                          query: str, 
                          filters: Optional[Dict[str, Any]] = None, 
                          top_k: int = 100) -> str:
        """Generate cache key for search results."""
        try:
            key_parts = [f"dense_search:{hash(query)}", f"top_k:{top_k}"]
            
            if filters:
                filter_str = str(sorted(filters.items()))
                key_parts.append(f"filters:{hash(filter_str)}")
            
            return ":".join(key_parts)
            
        except Exception:
            return f"dense_search:{hash(query)}:{top_k}"
    
    def _update_search_stats(self, search_time: float):
        """Update search performance statistics."""
        try:
            self.search_stats['total_searches'] += 1
            
            # Update average search time
            total_time = self.search_stats['avg_search_time'] * (self.search_stats['total_searches'] - 1)
            self.search_stats['avg_search_time'] = (total_time + search_time) / self.search_stats['total_searches']
            
        except Exception as e:
            logger.error(f"Search stats update failed: {e}")
    
    async def optimize_similarity_threshold(self, 
                                          test_queries: List[str],
                                          ground_truth: Optional[Dict[str, List[str]]] = None) -> float:
        """Optimize similarity threshold based on test queries."""
        try:
            if not test_queries:
                return self.similarity_threshold
            
            best_threshold = self.similarity_threshold
            best_score = 0.0
            
            # Test different thresholds
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
            
            for threshold in thresholds:
                original_threshold = self.similarity_threshold
                self.similarity_threshold = threshold
                
                # Evaluate performance
                total_results = 0
                relevant_results = 0
                
                for query in test_queries:
                    results = await self.search(query, top_k=10)
                    total_results += len(results)
                    
                    # If ground truth available, calculate precision
                    if ground_truth and query in ground_truth:
                        relevant_chunks = set(ground_truth[query])
                        found_chunks = set(r.get('chunk_id', '') for r in results)
                        relevant_results += len(relevant_chunks & found_chunks)
                
                # Calculate score (precision if ground truth available, otherwise result count)
                if ground_truth:
                    score = relevant_results / max(1, total_results)
                else:
                    score = total_results / len(test_queries)  # Average results per query
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                
                # Restore original threshold
                self.similarity_threshold = original_threshold
            
            # Set optimal threshold
            self.similarity_threshold = best_threshold
            
            logger.info(f"Optimized similarity threshold to {best_threshold} (score: {best_score:.3f})")
            
            return best_threshold
            
        except Exception as e:
            logger.error(f"Threshold optimization failed: {e}")
            return self.similarity_threshold
    
    async def get_retriever_stats(self) -> Dict[str, Any]:
        """Get dense retriever statistics."""
        try:
            return {
                'similarity_threshold': self.similarity_threshold,
                'max_results': self.max_results,
                'enable_query_expansion': self.enable_query_expansion,
                'enable_reranking': self.enable_reranking,
                'expansion_cache_size': len(self.expansion_cache),
                'search_stats': self.search_stats.copy(),
                'embedding_service_available': self.embedding_service is not None,
                'vector_store_available': self.vector_store is not None,
                'tokenizer_initialized': self.tokenizer._initialized if hasattr(self.tokenizer, '_initialized') else False
            }
            
        except Exception as e:
            logger.error(f"Retriever stats retrieval failed: {e}")
            return {'error': str(e)}
    
    async def clear_caches(self):
        """Clear all caches."""
        try:
            self.expansion_cache.clear()
            # Note: Redis cache clearing would need to be implemented separately
            logger.info("Cleared dense retriever caches")
            
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")


# Global instance
dense_retriever = DenseRetriever()