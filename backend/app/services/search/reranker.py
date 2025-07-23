"""
LLM-based reranking system for improving search result relevance.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from collections import defaultdict

import trio

from app.services.nlp.rag_tokenizer import rag_tokenizer
from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMReranker:
    """
    LLM-based reranking system for improving search result relevance.
    """
    
    def __init__(self,
                 max_rerank_candidates: int = 50,
                 batch_size: int = 10,
                 relevance_threshold: float = 0.6,
                 enable_caching: bool = True):
        """
        Initialize LLM reranker.
        
        Args:
            max_rerank_candidates: Maximum number of candidates to rerank
            batch_size: Batch size for LLM processing
            relevance_threshold: Minimum relevance score threshold
            enable_caching: Whether to cache reranking results
        """
        self.max_rerank_candidates = max_rerank_candidates
        self.batch_size = batch_size
        self.relevance_threshold = relevance_threshold
        self.enable_caching = enable_caching
        
        self.tokenizer = rag_tokenizer
        self.cache = cache_manager
        
        # Reranking prompts for different scenarios
        self.reranking_prompts = {
            'general': """
You are an expert at evaluating the relevance of search results to user queries.

Query: "{query}"

Please evaluate how relevant each of the following search results is to the query. 
Rate each result on a scale from 0.0 to 1.0, where:
- 1.0 = Perfectly relevant and directly answers the query
- 0.8 = Highly relevant with good information
- 0.6 = Moderately relevant with some useful information
- 0.4 = Somewhat relevant but limited usefulness
- 0.2 = Barely relevant with minimal connection
- 0.0 = Not relevant at all

Results to evaluate:
{results}

Respond with a JSON object containing the relevance scores:
{{"scores": [score1, score2, score3, ...]}}
""",
            
            'technical': """
You are a technical expert evaluating search results for technical queries.

Query: "{query}"

Evaluate the technical relevance and accuracy of each result. Consider:
- Technical accuracy and correctness
- Depth of technical detail
- Practical applicability
- Code examples and implementation details

Rate each result from 0.0 to 1.0:

Results:
{results}

Respond with JSON: {{"scores": [score1, score2, ...]}}
""",
            
            'factual': """
You are evaluating search results for factual accuracy and completeness.

Query: "{query}"

Evaluate each result based on:
- Factual accuracy
- Completeness of information
- Source credibility indicators
- Clarity of explanation

Rate from 0.0 to 1.0:

Results:
{results}

Respond with JSON: {{"scores": [score1, score2, ...]}}
"""
        }
        
        # Performance tracking
        self.rerank_stats = {
            'total_reranks': 0,
            'avg_rerank_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_calls': 0
        }
    
    async def batch_rerank_results(self,
                                 queries_and_results: List[Tuple[str, List[Dict[str, Any]]]],
                                 rerank_type: str = 'general',
                                 top_k: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        Batch rerank multiple queries for performance optimization.
        
        Args:
            queries_and_results: List of (query, search_results) tuples
            rerank_type: Type of reranking (general, technical, factual)
            top_k: Number of top results to return per query
            
        Returns:
            List of reranked results for each query
        """
        try:
            if not queries_and_results:
                return []
            
            # Process all queries concurrently using trio
            async with trio.open_nursery() as nursery:
                results = []
                
                async def process_single_query(query_result_pair, index):
                    query, search_results = query_result_pair
                    reranked = await self.rerank_results(query, search_results, rerank_type, top_k)
                    results.append((index, reranked))
                
                # Start all reranking tasks
                for i, query_result_pair in enumerate(queries_and_results):
                    nursery.start_soon(process_single_query, query_result_pair, i)
            
            # Sort results by original order
            results.sort(key=lambda x: x[0])
            return [result[1] for result in results]
            
        except Exception as e:
            logger.error(f"Batch reranking failed: {e}")
            # Return original results if batch processing fails
            return [search_results for _, search_results in queries_and_results]
    
    async def rerank_results(self, 
                           query: str,
                           search_results: List[Dict[str, Any]],
                           rerank_type: str = 'general',
                           top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank search results using LLM evaluation.
        
        Args:
            query: Original search query
            search_results: List of search results to rerank
            rerank_type: Type of reranking (general, technical, factual)
            top_k: Number of top results to return
            
        Returns:
            Reranked results with relevance scores
        """
        start_time = time.time()
        
        try:
            if not search_results or not query.strip():
                return search_results
            
            # Initialize services
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            # Limit candidates for reranking
            candidates = search_results[:self.max_rerank_candidates]
            remaining_results = search_results[self.max_rerank_candidates:]
            
            # Check cache
            if self.enable_caching:
                cache_key = self._generate_cache_key(query, candidates, rerank_type)
                cached_results = await self.cache.get(cache_key)
                
                if cached_results:
                    self.rerank_stats['cache_hits'] += 1
                    logger.debug(f"Using cached reranking results for query: {query}")
                    return cached_results + remaining_results
                
                self.rerank_stats['cache_misses'] += 1
            
            # Perform LLM-based reranking
            reranked_candidates = await self._llm_rerank(query, candidates, rerank_type)
            
            # Combine with remaining results
            final_results = reranked_candidates + remaining_results
            
            # Apply top_k limit if specified
            if top_k:
                final_results = final_results[:top_k]
            
            # Cache results with enhanced metadata
            if self.enable_caching and reranked_candidates:
                cache_data = {
                    'results': reranked_candidates,
                    'rerank_type': rerank_type,
                    'timestamp': time.time(),
                    'query_hash': hash(query),
                    'result_count': len(reranked_candidates)
                }
                await self.cache.set(cache_key, cache_data, ttl=3600)  # 1 hour
            
            # Update stats
            rerank_time = time.time() - start_time
            self._update_rerank_stats(rerank_time)
            
            logger.debug(f"Reranked {len(candidates)} results in {rerank_time:.3f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return search_results
    
    async def _llm_rerank(self, 
                        query: str,
                        results: List[Dict[str, Any]],
                        rerank_type: str) -> List[Dict[str, Any]]:
        """Perform LLM-based reranking."""
        try:
            if not results:
                return []
            
            # Process in batches
            reranked_results = []
            
            for i in range(0, len(results), self.batch_size):
                batch = results[i:i + self.batch_size]
                batch_scores = await self._evaluate_batch(query, batch, rerank_type)
                
                # Apply scores to results
                for j, result in enumerate(batch):
                    if j < len(batch_scores):
                        result_copy = result.copy()
                        result_copy['llm_relevance_score'] = batch_scores[j]
                        result_copy['original_score'] = result.get('score', 0.0)
                        result_copy['rerank_type'] = rerank_type
                        
                        # Combine original and LLM scores
                        combined_score = self._combine_scores(
                            result.get('score', 0.0),
                            batch_scores[j]
                        )
                        result_copy['score'] = combined_score
                        result_copy['combined_score'] = combined_score
                        
                        reranked_results.append(result_copy)
                    else:
                        reranked_results.append(result)
            
            # Sort by combined score
            reranked_results.sort(key=lambda x: x.get('combined_score', x.get('score', 0)), reverse=True)
            
            # Filter by relevance threshold
            filtered_results = [r for r in reranked_results 
                              if r.get('llm_relevance_score', 0) >= self.relevance_threshold]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"LLM reranking process failed: {e}")
            return results
    
    async def _evaluate_batch(self, 
                            query: str,
                            batch: List[Dict[str, Any]],
                            rerank_type: str) -> List[float]:
        """Evaluate a batch of results using LLM."""
        try:
            # Prepare results for LLM evaluation
            formatted_results = []
            for i, result in enumerate(batch):
                content = result.get('content', result.get('text', ''))
                title = result.get('title', '')
                
                # Truncate content if too long
                max_content_length = 500
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                
                formatted_result = f"{i+1}. "
                if title:
                    formatted_result += f"Title: {title}\n"
                formatted_result += f"Content: {content}"
                
                formatted_results.append(formatted_result)
            
            # Get appropriate prompt
            prompt_template = self.reranking_prompts.get(rerank_type, self.reranking_prompts['general'])
            
            # Format prompt
            results_text = "\n\n".join(formatted_results)
            prompt = prompt_template.format(query=query, results=results_text)
            
            # Call LLM (placeholder implementation)
            scores = await self._call_llm_for_scoring(prompt, len(batch))
            
            self.rerank_stats['llm_calls'] += 1
            
            return scores
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            # Return default scores
            return [0.5] * len(batch)
    
    async def _call_llm_for_scoring(self, prompt: str, expected_count: int) -> List[float]:
        """
        Call LLM for scoring with actual Yandex LLM integration.
        
        This implementation calls the actual Yandex LLM service for relevance scoring.
        Falls back to heuristic scoring if LLM is unavailable.
        """
        try:
            # Try to use actual LLM service
            from app.core.config import settings
            
            # Import Yandex LLM client if available
            try:
                from app.services.llm.yandex_client import YandexLLMClient
                
                if not hasattr(self, '_llm_client'):
                    self._llm_client = YandexLLMClient(
                        api_key=settings.YANDEX_API_KEY,
                        folder_id=settings.YANDEX_FOLDER_ID
                    )
                
                # Call LLM with structured prompt
                response = await self._llm_client.generate_text(
                    prompt=prompt,
                    temperature=0.1,  # Low temperature for consistent scoring
                    max_tokens=200
                )
                
                # Parse JSON response
                import json
                import re
                
                # Extract JSON from response
                json_match = re.search(r'\{[^}]*"scores"[^}]*\}', response)
                if json_match:
                    json_str = json_match.group()
                    parsed_response = json.loads(json_str)
                    
                    if 'scores' in parsed_response:
                        scores = parsed_response['scores']
                        
                        # Validate scores
                        validated_scores = []
                        for score in scores[:expected_count]:
                            if isinstance(score, (int, float)) and 0 <= score <= 1:
                                validated_scores.append(float(score))
                            else:
                                validated_scores.append(0.5)
                        
                        # Pad if needed
                        while len(validated_scores) < expected_count:
                            validated_scores.append(0.5)
                        
                        logger.debug(f"LLM scoring successful: {validated_scores}")
                        return validated_scores[:expected_count]
                
                logger.warning("LLM response format invalid, falling back to heuristic scoring")
                
            except ImportError:
                logger.warning("Yandex LLM client not available, using heuristic scoring")
            except Exception as llm_error:
                logger.warning(f"LLM scoring failed: {llm_error}, falling back to heuristic scoring")
            
            # Fallback to enhanced heuristic scoring
            return await self._heuristic_scoring(prompt, expected_count)
            
        except Exception as e:
            logger.error(f"LLM scoring call failed: {e}")
            return [0.5] * expected_count
    
    async def _heuristic_scoring(self, prompt: str, expected_count: int) -> List[float]:
        """Enhanced heuristic scoring as fallback."""
        try:
            # Extract query and results from prompt
            lines = prompt.split('\n')
            query_line = next((line for line in lines if line.startswith('Query:')), '')
            query = query_line.replace('Query:', '').strip().strip('"')
            
            # Find results section
            results_start = False
            result_texts = []
            current_result = ""
            
            for line in lines:
                if "Results" in line and "evaluate" in line:
                    results_start = True
                    continue
                
                if results_start and line.strip():
                    if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                        if current_result:
                            result_texts.append(current_result)
                        current_result = line
                    else:
                        current_result += " " + line
            
            if current_result:
                result_texts.append(current_result)
            
            # Enhanced heuristic scoring
            scores = []
            query_tokens = set(query.lower().split())
            
            # Use tokenizer for better analysis
            if hasattr(self, 'tokenizer') and self.tokenizer._initialized:
                query_processed = await self._process_query_tokens(query)
                query_tokens = set(query_processed)
            
            for result_text in result_texts[:expected_count]:
                score = await self._calculate_heuristic_score(query, query_tokens, result_text)
                scores.append(score)
            
            # Pad with default scores if needed
            while len(scores) < expected_count:
                scores.append(0.5)
            
            return scores[:expected_count]
            
        except Exception as e:
            logger.error(f"Heuristic scoring failed: {e}")
            return [0.5] * expected_count
    
    async def _process_query_tokens(self, query: str) -> List[str]:
        """Process query tokens using RagTokenizer."""
        try:
            processed_query = self.tokenizer.tokenize(query)
            return processed_query.split()
        except Exception:
            return query.lower().split()
    
    async def _calculate_heuristic_score(self, query: str, query_tokens: set, result_text: str) -> float:
        """Calculate enhanced heuristic score for a result."""
        try:
            result_tokens = set(result_text.lower().split())
            
            # Base score from token overlap
            overlap = len(query_tokens & result_tokens)
            total_query_tokens = len(query_tokens)
            
            if total_query_tokens > 0:
                base_score = overlap / total_query_tokens
            else:
                base_score = 0.5
            
            # Title match bonus
            title_bonus = 0.0
            if 'title:' in result_text.lower():
                title_part = result_text.lower().split('title:')[1].split('content:')[0]
                if any(token in title_part for token in query_tokens):
                    title_bonus = 0.2
            
            # Content length bonus (moderate length preferred)
            length_bonus = 0.0
            content_length = len(result_text)
            if 200 <= content_length <= 800:
                length_bonus = 0.1
            elif 100 <= content_length <= 1200:
                length_bonus = 0.05
            
            # Exact phrase match bonus
            phrase_bonus = 0.0
            if len(query.split()) > 1:
                if query.lower() in result_text.lower():
                    phrase_bonus = 0.15
            
            # Position bonus (earlier results get slight boost)
            position_bonus = 0.02
            
            # Combine all scores
            final_score = base_score + title_bonus + length_bonus + phrase_bonus + position_bonus
            
            # Normalize to [0, 1]
            return min(1.0, max(0.0, final_score))
            
        except Exception:
            return 0.5
    
    def _combine_scores(self, original_score: float, llm_score: float) -> float:
        """Combine original search score with LLM relevance score."""
        try:
            # Weighted combination: 60% LLM score, 40% original score
            combined = 0.6 * llm_score + 0.4 * original_score
            return min(1.0, max(0.0, combined))
            
        except Exception:
            return original_score
    
    def _generate_cache_key(self, 
                          query: str, 
                          results: List[Dict[str, Any]], 
                          rerank_type: str) -> str:
        """Generate cache key for reranking results."""
        try:
            # Create hash from query and result IDs
            result_ids = [r.get('chunk_id', r.get('id', str(i))) for i, r in enumerate(results)]
            content_hash = hash(f"{query}:{':'.join(result_ids)}:{rerank_type}")
            
            return f"llm_rerank:{content_hash}"
            
        except Exception:
            return f"llm_rerank:{hash(query)}:{rerank_type}"
    
    def _update_rerank_stats(self, rerank_time: float):
        """Update reranking performance statistics."""
        try:
            self.rerank_stats['total_reranks'] += 1
            
            # Update average rerank time
            total_time = self.rerank_stats['avg_rerank_time'] * (self.rerank_stats['total_reranks'] - 1)
            self.rerank_stats['avg_rerank_time'] = (total_time + rerank_time) / self.rerank_stats['total_reranks']
            
        except Exception as e:
            logger.error(f"Rerank stats update failed: {e}")
    
    async def evaluate_reranking_quality(self, 
                                       original_results: List[Dict[str, Any]],
                                       reranked_results: List[Dict[str, Any]],
                                       ground_truth: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of reranking.
        
        Args:
            original_results: Original search results
            reranked_results: Reranked results
            ground_truth: Optional list of relevant result IDs
            
        Returns:
            Quality metrics
        """
        try:
            metrics = {
                'original_count': len(original_results),
                'reranked_count': len(reranked_results),
                'rank_changes': 0,
                'avg_llm_score': 0.0,
                'score_improvement': 0.0
            }
            
            if not reranked_results:
                return metrics
            
            # Calculate average LLM score
            llm_scores = [r.get('llm_relevance_score', 0) for r in reranked_results if 'llm_relevance_score' in r]
            if llm_scores:
                metrics['avg_llm_score'] = sum(llm_scores) / len(llm_scores)
            
            # Calculate rank changes
            original_ids = [self._get_result_id(r) for r in original_results]
            reranked_ids = [self._get_result_id(r) for r in reranked_results]
            
            rank_changes = 0
            for i, result_id in enumerate(reranked_ids):
                if result_id in original_ids:
                    original_rank = original_ids.index(result_id)
                    if original_rank != i:
                        rank_changes += 1
            
            metrics['rank_changes'] = rank_changes
            metrics['rank_change_ratio'] = rank_changes / len(reranked_results) if reranked_results else 0
            
            # Calculate score improvement
            original_scores = [r.get('score', 0) for r in original_results]
            reranked_scores = [r.get('combined_score', r.get('score', 0)) for r in reranked_results]
            
            if original_scores and reranked_scores:
                avg_original = sum(original_scores) / len(original_scores)
                avg_reranked = sum(reranked_scores) / len(reranked_scores)
                metrics['score_improvement'] = avg_reranked - avg_original
            
            # Ground truth evaluation if available
            if ground_truth:
                # Calculate NDCG improvement
                original_ndcg = self._calculate_ndcg(original_results, ground_truth)
                reranked_ndcg = self._calculate_ndcg(reranked_results, ground_truth)
                
                metrics['original_ndcg'] = original_ndcg
                metrics['reranked_ndcg'] = reranked_ndcg
                metrics['ndcg_improvement'] = reranked_ndcg - original_ndcg
                
                # Calculate precision@k improvements
                for k in [1, 3, 5, 10]:
                    if len(original_results) >= k and len(reranked_results) >= k:
                        orig_p_at_k = self._calculate_precision_at_k(original_results[:k], ground_truth)
                        rerank_p_at_k = self._calculate_precision_at_k(reranked_results[:k], ground_truth)
                        
                        metrics[f'precision_at_{k}_improvement'] = rerank_p_at_k - orig_p_at_k
            
            return metrics
            
        except Exception as e:
            logger.error(f"Reranking quality evaluation failed: {e}")
            return {'error': str(e)}
    
    def _get_result_id(self, result: Dict[str, Any]) -> str:
        """Get unique ID for a result."""
        return result.get('chunk_id', result.get('id', str(hash(str(result)))))
    
    def _calculate_ndcg(self, results: List[Dict[str, Any]], ground_truth: List[str]) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        try:
            if not results or not ground_truth:
                return 0.0
            
            # Calculate DCG
            dcg = 0.0
            for i, result in enumerate(results):
                result_id = self._get_result_id(result)
                if result_id in ground_truth:
                    dcg += 1.0 / (1 + i)  # Simple relevance = 1 for relevant, 0 for not
            
            # Calculate IDCG (ideal DCG)
            idcg = sum(1.0 / (1 + i) for i in range(min(len(results), len(ground_truth))))
            
            return dcg / idcg if idcg > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_precision_at_k(self, results: List[Dict[str, Any]], ground_truth: List[str]) -> float:
        """Calculate Precision@K."""
        try:
            if not results or not ground_truth:
                return 0.0
            
            relevant_count = 0
            for result in results:
                result_id = self._get_result_id(result)
                if result_id in ground_truth:
                    relevant_count += 1
            
            return relevant_count / len(results)
            
        except Exception:
            return 0.0
    
    async def optimize_reranking_parameters(self, 
                                          test_queries: List[str],
                                          test_results: List[List[Dict[str, Any]]],
                                          ground_truth: List[List[str]]) -> Dict[str, Any]:
        """
        Optimize reranking parameters using test data.
        
        Args:
            test_queries: List of test queries
            test_results: List of search results for each query
            ground_truth: List of relevant result IDs for each query
            
        Returns:
            Optimal parameters and performance metrics
        """
        try:
            if len(test_queries) != len(test_results) or len(test_queries) != len(ground_truth):
                raise ValueError("Test data lengths must match")
            
            best_params = {
                'relevance_threshold': self.relevance_threshold,
                'batch_size': self.batch_size,
                'score_combination_weight': 0.6
            }
            best_score = 0.0
            
            # Test different relevance thresholds
            for threshold in [0.4, 0.5, 0.6, 0.7, 0.8]:
                original_threshold = self.relevance_threshold
                self.relevance_threshold = threshold
                
                total_ndcg = 0.0
                for query, results, truth in zip(test_queries, test_results, ground_truth):
                    reranked = await self.rerank_results(query, results)
                    ndcg = self._calculate_ndcg(reranked, truth)
                    total_ndcg += ndcg
                
                avg_ndcg = total_ndcg / len(test_queries)
                if avg_ndcg > best_score:
                    best_score = avg_ndcg
                    best_params['relevance_threshold'] = threshold
                
                self.relevance_threshold = original_threshold
            
            # Test different batch sizes
            for batch_size in [5, 10, 15, 20]:
                original_batch_size = self.batch_size
                self.batch_size = batch_size
                
                total_ndcg = 0.0
                for query, results, truth in zip(test_queries, test_results, ground_truth):
                    reranked = await self.rerank_results(query, results)
                    ndcg = self._calculate_ndcg(reranked, truth)
                    total_ndcg += ndcg
                
                avg_ndcg = total_ndcg / len(test_queries)
                if avg_ndcg > best_score:
                    best_score = avg_ndcg
                    best_params['batch_size'] = batch_size
                
                self.batch_size = original_batch_size
            
            # Apply best parameters
            self.relevance_threshold = best_params['relevance_threshold']
            self.batch_size = best_params['batch_size']
            
            return {
                'best_parameters': best_params,
                'best_ndcg_score': best_score,
                'optimization_completed': True
            }
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return {'error': str(e)}
    
    async def validate_reranking_quality(self,
                                       validation_queries: List[str],
                                       validation_results: List[List[Dict[str, Any]]],
                                       ground_truth: List[List[str]]) -> Dict[str, Any]:
        """
        Validate reranking quality on a validation set.
        
        Args:
            validation_queries: List of validation queries
            validation_results: List of search results for each query
            ground_truth: List of relevant result IDs for each query
            
        Returns:
            Validation metrics
        """
        try:
            if not validation_queries:
                return {'error': 'No validation data provided'}
            
            metrics = {
                'total_queries': len(validation_queries),
                'avg_ndcg_improvement': 0.0,
                'avg_precision_at_1_improvement': 0.0,
                'avg_precision_at_5_improvement': 0.0,
                'avg_rerank_time': 0.0,
                'cache_hit_rate': 0.0,
                'quality_score': 0.0
            }
            
            total_ndcg_improvement = 0.0
            total_p1_improvement = 0.0
            total_p5_improvement = 0.0
            total_rerank_time = 0.0
            
            for query, results, truth in zip(validation_queries, validation_results, ground_truth):
                start_time = time.time()
                
                # Get original metrics
                original_ndcg = self._calculate_ndcg(results, truth)
                original_p1 = self._calculate_precision_at_k(results[:1], truth)
                original_p5 = self._calculate_precision_at_k(results[:5], truth)
                
                # Rerank and get new metrics
                reranked = await self.rerank_results(query, results)
                reranked_ndcg = self._calculate_ndcg(reranked, truth)
                reranked_p1 = self._calculate_precision_at_k(reranked[:1], truth)
                reranked_p5 = self._calculate_precision_at_k(reranked[:5], truth)
                
                # Calculate improvements
                total_ndcg_improvement += (reranked_ndcg - original_ndcg)
                total_p1_improvement += (reranked_p1 - original_p1)
                total_p5_improvement += (reranked_p5 - original_p5)
                
                rerank_time = time.time() - start_time
                total_rerank_time += rerank_time
            
            # Calculate averages
            num_queries = len(validation_queries)
            metrics['avg_ndcg_improvement'] = total_ndcg_improvement / num_queries
            metrics['avg_precision_at_1_improvement'] = total_p1_improvement / num_queries
            metrics['avg_precision_at_5_improvement'] = total_p5_improvement / num_queries
            metrics['avg_rerank_time'] = total_rerank_time / num_queries
            
            # Calculate cache hit rate
            total_requests = self.rerank_stats['cache_hits'] + self.rerank_stats['cache_misses']
            if total_requests > 0:
                metrics['cache_hit_rate'] = self.rerank_stats['cache_hits'] / total_requests
            
            # Calculate overall quality score
            quality_score = (
                metrics['avg_ndcg_improvement'] * 0.4 +
                metrics['avg_precision_at_1_improvement'] * 0.3 +
                metrics['avg_precision_at_5_improvement'] * 0.2 +
                min(1.0, metrics['cache_hit_rate']) * 0.1
            )
            metrics['quality_score'] = max(0.0, quality_score)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Reranking quality validation failed: {e}")
            return {'error': str(e)}
    
    async def clear_reranking_cache(self) -> Dict[str, Any]:
        """Clear all reranking cache entries."""
        try:
            # Clear cache entries with rerank prefix
            cache_keys = await self.cache.keys("llm_rerank:*")
            cleared_count = 0
            
            for key in cache_keys:
                await self.cache.delete(key)
                cleared_count += 1
            
            # Reset cache stats
            self.rerank_stats['cache_hits'] = 0
            self.rerank_stats['cache_misses'] = 0
            
            logger.info(f"Cleared {cleared_count} reranking cache entries")
            
            return {
                'cleared_entries': cleared_count,
                'cache_cleared': True
            }
            
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
            return {'error': str(e)}
    
    async def update_reranking_prompts(self, new_prompts: Dict[str, str]) -> Dict[str, Any]:
        """
        Update reranking prompts for different scenarios.
        
        Args:
            new_prompts: Dictionary of prompt type -> prompt template
            
        Returns:
            Update status
        """
        try:
            updated_types = []
            
            for prompt_type, prompt_template in new_prompts.items():
                if prompt_type in self.reranking_prompts:
                    self.reranking_prompts[prompt_type] = prompt_template
                    updated_types.append(prompt_type)
                else:
                    logger.warning(f"Unknown prompt type: {prompt_type}")
            
            # Clear cache after prompt updates
            await self.clear_reranking_cache()
            
            return {
                'updated_types': updated_types,
                'total_updated': len(updated_types),
                'available_types': list(self.reranking_prompts.keys())
            }
            
        except Exception as e:
            logger.error(f"Prompt update failed: {e}")
            return {'error': str(e)}
    
    async def get_reranker_stats(self) -> Dict[str, Any]:
        """Get comprehensive reranker statistics."""
        try:
            # Calculate cache hit rate
            total_requests = self.rerank_stats['cache_hits'] + self.rerank_stats['cache_misses']
            cache_hit_rate = self.rerank_stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'configuration': {
                    'max_rerank_candidates': self.max_rerank_candidates,
                    'batch_size': self.batch_size,
                    'relevance_threshold': self.relevance_threshold,
                    'enable_caching': self.enable_caching
                },
                'available_rerank_types': list(self.reranking_prompts.keys()),
                'performance_stats': {
                    **self.rerank_stats,
                    'cache_hit_rate': cache_hit_rate,
                    'avg_batch_processing_time': self.rerank_stats['avg_rerank_time'] / max(1, self.batch_size)
                },
                'system_status': {
                    'tokenizer_initialized': self.tokenizer._initialized if hasattr(self.tokenizer, '_initialized') else False,
                    'cache_available': self.cache is not None,
                    'llm_client_available': hasattr(self, '_llm_client')
                }
            }
            
        except Exception as e:
            logger.error(f"Reranker stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
llm_reranker = LLMReranker()