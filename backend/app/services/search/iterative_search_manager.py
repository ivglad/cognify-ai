"""
Iterative search refinement system for improved query processing.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import time
import re
from enum import Enum
from collections import defaultdict

import trio

from app.services.search.retriever import basic_retriever
from app.services.nlp.rag_tokenizer import rag_tokenizer
from app.core.config import settings
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class SearchRefinementStrategy(Enum):
    """Search refinement strategies."""
    EXPANSION = "expansion"
    NARROWING = "narrowing"
    REFORMULATION = "reformulation"
    SEMANTIC_SHIFT = "semantic_shift"
    CONTEXT_ADDITION = "context_addition"


class IterativeSearchManager:
    """
    Manager for iterative search refinement with query optimization.
    """
    
    def __init__(self,
                 max_iterations: int = 5,
                 min_results_threshold: int = 3,
                 max_results_threshold: int = 20,
                 similarity_threshold: float = 0.8,
                 enable_caching: bool = True):
        """
        Initialize iterative search manager.
        
        Args:
            max_iterations: Maximum search iterations
            min_results_threshold: Minimum results to continue iteration
            max_results_threshold: Maximum results before stopping
            similarity_threshold: Threshold for result similarity detection
            enable_caching: Whether to cache search results
        """
        self.max_iterations = max_iterations
        self.min_results_threshold = min_results_threshold
        self.max_results_threshold = max_results_threshold
        self.similarity_threshold = similarity_threshold
        self.enable_caching = enable_caching
        
        self.retriever = basic_retriever
        self.tokenizer = rag_tokenizer
        self.cache = cache_manager
        
        self._initialized = False
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'avg_iterations': 0.0,
            'avg_search_time': 0.0,
            'successful_refinements': 0,
            'failed_refinements': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def initialize(self):
        """Initialize the iterative search manager."""
        if self._initialized:
            return
        
        try:
            # Initialize retriever
            if not self.retriever._initialized:
                await self.retriever.initialize()
            
            # Initialize tokenizer
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            self._initialized = True
            logger.info("IterativeSearchManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize IterativeSearchManager: {e}")
            self._initialized = False
            raise 
   
    async def iterative_search(self,
                             initial_query: str,
                             document_ids: Optional[List[str]] = None,
                             target_result_count: int = 10,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform iterative search with query refinement.
        
        Args:
            initial_query: Initial search query
            document_ids: Optional document IDs to limit search scope
            target_result_count: Target number of results to achieve
            context: Optional context for refinement
            
        Returns:
            Comprehensive search results with refinement history
        """
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            if not initial_query.strip():
                return self._create_error_response("Empty query provided", start_time)
            
            # Check cache
            if self.enable_caching:
                cache_key = self._generate_cache_key(initial_query, document_ids, target_result_count)
                cached_result = await self.cache.get(cache_key)
                
                if cached_result:
                    self.search_stats['cache_hits'] += 1
                    logger.debug(f"Using cached iterative search result for query: {initial_query}")
                    return cached_result
                
                self.search_stats['cache_misses'] += 1
            
            # Initialize search state
            search_state = {
                'original_query': initial_query,
                'current_query': initial_query,
                'iteration': 0,
                'all_results': [],
                'unique_results': set(),
                'refinement_history': [],
                'performance_metrics': {
                    'query_variations': [],
                    'result_counts': [],
                    'search_times': []
                }
            }
            
            # Perform iterative search
            final_results = await self._execute_iterative_search(
                search_state, document_ids, target_result_count, context
            )
            
            # Create comprehensive response
            response = {
                'original_query': initial_query,
                'final_query': search_state['current_query'],
                'results': final_results,
                'total_results': len(final_results),
                'iterations_performed': search_state['iteration'],
                'refinement_history': search_state['refinement_history'],
                'performance_metrics': search_state['performance_metrics'],
                'search_time': time.time() - start_time,
                'success': True
            }
            
            # Cache result
            if self.enable_caching:
                await self.cache.set(cache_key, response, ttl=1800)  # 30 minutes
            
            # Update stats
            self._update_search_stats(response)
            
            logger.info(f"Iterative search completed in {response['search_time']:.3f}s with {response['iterations_performed']} iterations")
            
            return response
            
        except Exception as e:
            logger.error(f"Iterative search failed: {e}")
            return self._create_error_response(str(e), start_time)
    
    async def _execute_iterative_search(self,
                                      search_state: Dict[str, Any],
                                      document_ids: Optional[List[str]],
                                      target_result_count: int,
                                      context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute the iterative search process."""
        try:
            while search_state['iteration'] < self.max_iterations:
                search_state['iteration'] += 1
                iteration_start = time.time()
                
                # Perform search with current query
                search_result = await self.retriever.search(
                    query=search_state['current_query'],
                    document_ids=document_ids,
                    search_type="hybrid",
                    top_k=min(target_result_count * 2, 30),  # Get extra results for filtering
                    enable_reranking=True
                )
                
                iteration_time = time.time() - iteration_start
                
                # Process search results
                new_results = self._process_search_results(
                    search_result.get('results', []),
                    search_state['unique_results']
                )
                
                # Update search state
                search_state['all_results'].extend(new_results)
                search_state['performance_metrics']['query_variations'].append(search_state['current_query'])
                search_state['performance_metrics']['result_counts'].append(len(new_results))
                search_state['performance_metrics']['search_times'].append(iteration_time)
                
                # Check termination conditions
                total_unique_results = len(search_state['all_results'])
                
                if total_unique_results >= target_result_count:
                    logger.debug(f"Target result count reached: {total_unique_results}")
                    break
                
                if total_unique_results >= self.max_results_threshold:
                    logger.debug(f"Maximum results threshold reached: {total_unique_results}")
                    break
                
                if len(new_results) < self.min_results_threshold and search_state['iteration'] > 1:
                    logger.debug(f"Insufficient new results: {len(new_results)}")
                    break
                
                # Generate refined query for next iteration
                refinement_result = await self._refine_query(
                    search_state, new_results, target_result_count, context
                )
                
                if not refinement_result['success']:
                    logger.debug("Query refinement failed, stopping iterations")
                    break
                
                search_state['current_query'] = refinement_result['refined_query']
                search_state['refinement_history'].append(refinement_result['refinement_info'])
                
                # Check if query hasn't changed significantly
                if self._queries_too_similar(
                    search_state['refinement_history'][-2:] if len(search_state['refinement_history']) >= 2 else []
                ):
                    logger.debug("Query refinement not producing significant changes")
                    break
            
            # Return top results sorted by relevance
            final_results = sorted(
                search_state['all_results'],
                key=lambda x: x.get('score', 0),
                reverse=True
            )[:target_result_count]
            
            return final_results
            
        except Exception as e:
            logger.error(f"Iterative search execution failed: {e}")
            return search_state['all_results'][:target_result_count]
    
    def _process_search_results(self,
                              results: List[Dict[str, Any]],
                              seen_results: Set[str]) -> List[Dict[str, Any]]:
        """Process and deduplicate search results."""
        new_results = []
        
        for result in results:
            chunk_id = result.get('chunk_id', '')
            if chunk_id and chunk_id not in seen_results:
                seen_results.add(chunk_id)
                new_results.append(result)
        
        return new_results
    
    async def _refine_query(self,
                          search_state: Dict[str, Any],
                          recent_results: List[Dict[str, Any]],
                          target_result_count: int,
                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Refine query based on search results and context."""
        try:
            current_query = search_state['current_query']
            original_query = search_state['original_query']
            iteration = search_state['iteration']
            total_results = len(search_state['all_results'])
            
            # Determine refinement strategy
            strategy = self._select_refinement_strategy(
                total_results, target_result_count, iteration, recent_results
            )
            
            # Apply refinement strategy
            if strategy == SearchRefinementStrategy.EXPANSION:
                refined_query = await self._expand_query(current_query, original_query, context)
            elif strategy == SearchRefinementStrategy.NARROWING:
                refined_query = await self._narrow_query(current_query, recent_results, context)
            elif strategy == SearchRefinementStrategy.REFORMULATION:
                refined_query = await self._reformulate_query(current_query, original_query, recent_results)
            elif strategy == SearchRefinementStrategy.SEMANTIC_SHIFT:
                refined_query = await self._semantic_shift_query(current_query, recent_results)
            elif strategy == SearchRefinementStrategy.CONTEXT_ADDITION:
                refined_query = await self._add_context_to_query(current_query, context, recent_results)
            else:
                refined_query = current_query
            
            # Validate refined query
            if not refined_query or refined_query.strip() == current_query.strip():
                return {'success': False, 'error': 'No meaningful refinement generated'}
            
            refinement_info = {
                'iteration': iteration,
                'strategy': strategy.value,
                'original_query': current_query,
                'refined_query': refined_query,
                'reason': self._get_refinement_reason(strategy, total_results, target_result_count),
                'timestamp': time.time()
            }
            
            return {
                'success': True,
                'refined_query': refined_query,
                'refinement_info': refinement_info
            }
            
        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _select_refinement_strategy(self,
                                  total_results: int,
                                  target_result_count: int,
                                  iteration: int,
                                  recent_results: List[Dict[str, Any]]) -> SearchRefinementStrategy:
        """Select appropriate refinement strategy."""
        # Too few results - expand search
        if total_results < target_result_count * 0.5:
            return SearchRefinementStrategy.EXPANSION
        
        # Too many results - narrow search
        if total_results > target_result_count * 1.5:
            return SearchRefinementStrategy.NARROWING
        
        # Early iterations - try reformulation
        if iteration <= 2:
            return SearchRefinementStrategy.REFORMULATION
        
        # Later iterations - try semantic shift
        if iteration <= 4:
            return SearchRefinementStrategy.SEMANTIC_SHIFT
        
        # Final iterations - add context
        return SearchRefinementStrategy.CONTEXT_ADDITION 
   
    async def _expand_query(self,
                          current_query: str,
                          original_query: str,
                          context: Optional[Dict[str, Any]]) -> str:
        """Expand query to find more results."""
        try:
            # Tokenize current query
            if self.tokenizer._initialized:
                tokens = self.tokenizer.tokenize(current_query).split()
            else:
                tokens = current_query.split()
            
            # Add synonyms and related terms
            expanded_terms = []
            for token in tokens:
                expanded_terms.append(token)
                # Add common variations
                if len(token) > 4:
                    expanded_terms.append(f"{token}*")  # Wildcard expansion
            
            # Add context terms if available
            if context and 'related_terms' in context:
                expanded_terms.extend(context['related_terms'][:3])
            
            # Create expanded query
            expanded_query = " OR ".join(expanded_terms[:8])  # Limit to avoid too broad search
            
            return expanded_query if expanded_query != current_query else f"{current_query} OR {original_query}"
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return current_query
    
    async def _narrow_query(self,
                          current_query: str,
                          recent_results: List[Dict[str, Any]],
                          context: Optional[Dict[str, Any]]) -> str:
        """Narrow query to reduce result count."""
        try:
            # Extract key terms from high-scoring results
            key_terms = self._extract_key_terms_from_results(recent_results[:5])
            
            # Combine with current query
            if key_terms:
                narrowed_query = f"{current_query} AND ({' OR '.join(key_terms[:3])})"
            else:
                # Add specific constraints
                narrowed_query = f'"{current_query}"'  # Exact phrase search
            
            return narrowed_query
            
        except Exception as e:
            logger.error(f"Query narrowing failed: {e}")
            return current_query
    
    async def _reformulate_query(self,
                               current_query: str,
                               original_query: str,
                               recent_results: List[Dict[str, Any]]) -> str:
        """Reformulate query using different terms."""
        try:
            # Extract important terms from results
            result_terms = self._extract_key_terms_from_results(recent_results)
            
            # Tokenize original query
            if self.tokenizer._initialized:
                original_tokens = set(self.tokenizer.tokenize(original_query).split())
            else:
                original_tokens = set(original_query.split())
            
            # Find new terms that weren't in original query
            new_terms = [term for term in result_terms if term.lower() not in {t.lower() for t in original_tokens}]
            
            if new_terms:
                # Combine original intent with new terms
                reformulated = f"{original_query} {' '.join(new_terms[:3])}"
            else:
                # Try different phrasing
                reformulated = self._rephrase_query(current_query)
            
            return reformulated
            
        except Exception as e:
            logger.error(f"Query reformulation failed: {e}")
            return current_query
    
    async def _semantic_shift_query(self,
                                  current_query: str,
                                  recent_results: List[Dict[str, Any]]) -> str:
        """Shift query semantically based on result patterns."""
        try:
            # Analyze result content for semantic patterns
            content_analysis = self._analyze_result_content(recent_results)
            
            # Generate semantic variations
            if content_analysis['dominant_topics']:
                topic = content_analysis['dominant_topics'][0]
                semantic_query = f"{current_query} {topic}"
            else:
                # Try different semantic approach
                semantic_query = self._generate_semantic_variation(current_query)
            
            return semantic_query
            
        except Exception as e:
            logger.error(f"Semantic shift failed: {e}")
            return current_query
    
    async def _add_context_to_query(self,
                                  current_query: str,
                                  context: Optional[Dict[str, Any]],
                                  recent_results: List[Dict[str, Any]]) -> str:
        """Add contextual information to query."""
        try:
            context_terms = []
            
            # Add context from provided context
            if context:
                if 'domain' in context:
                    context_terms.append(context['domain'])
                if 'specific_focus' in context:
                    context_terms.append(context['specific_focus'])
            
            # Add context from result patterns
            result_context = self._extract_context_from_results(recent_results)
            context_terms.extend(result_context[:2])
            
            if context_terms:
                contextual_query = f"{current_query} ({' OR '.join(context_terms)})"
            else:
                contextual_query = current_query
            
            return contextual_query
            
        except Exception as e:
            logger.error(f"Context addition failed: {e}")
            return current_query
    
    def _extract_key_terms_from_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract key terms from search results."""
        try:
            term_frequency = defaultdict(int)
            
            for result in results:
                content = result.get('content', '')
                if self.tokenizer._initialized:
                    tokens = self.tokenizer.tokenize(content).split()
                else:
                    tokens = content.split()
                
                # Count significant terms (length > 3, not common words)
                for token in tokens:
                    if len(token) > 3 and token.lower() not in {'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were'}:
                        term_frequency[token.lower()] += 1
            
            # Return most frequent terms
            sorted_terms = sorted(term_frequency.items(), key=lambda x: x[1], reverse=True)
            return [term for term, freq in sorted_terms[:10] if freq > 1]
            
        except Exception as e:
            logger.error(f"Key term extraction failed: {e}")
            return []
    
    def _analyze_result_content(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content patterns in results."""
        try:
            analysis = {
                'dominant_topics': [],
                'content_types': [],
                'avg_content_length': 0,
                'common_phrases': []
            }
            
            if not results:
                return analysis
            
            # Analyze content types and topics
            content_lengths = []
            all_content = []
            
            for result in results:
                content = result.get('content', '')
                all_content.append(content)
                content_lengths.append(len(content))
                
                # Extract potential topics (simple heuristic)
                if 'metadata' in result:
                    chunk_type = result['metadata'].get('chunk_type', '')
                    if chunk_type:
                        analysis['content_types'].append(chunk_type)
            
            analysis['avg_content_length'] = sum(content_lengths) / len(content_lengths) if content_lengths else 0
            
            # Find common phrases (simple implementation)
            combined_content = ' '.join(all_content)
            words = combined_content.split()
            
            # Extract potential topics (words that appear frequently)
            word_freq = defaultdict(int)
            for word in words:
                if len(word) > 4:
                    word_freq[word.lower()] += 1
            
            analysis['dominant_topics'] = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5] if freq > 2]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {'dominant_topics': [], 'content_types': [], 'avg_content_length': 0, 'common_phrases': []}
    
    def _rephrase_query(self, query: str) -> str:
        """Simple query rephrasing."""
        try:
            # Simple rephrasing strategies
            if query.startswith('what is'):
                return query.replace('what is', 'define')
            elif query.startswith('how to'):
                return query.replace('how to', 'steps for')
            elif query.startswith('why'):
                return query.replace('why', 'reasons for')
            else:
                # Add question words
                return f"explain {query}"
                
        except Exception:
            return query
    
    def _generate_semantic_variation(self, query: str) -> str:
        """Generate semantic variation of query."""
        try:
            # Simple semantic variations
            variations = [
                f"concept of {query}",
                f"understanding {query}",
                f"principles of {query}",
                f"fundamentals of {query}"
            ]
            
            # Return first variation that's different from original
            for variation in variations:
                if variation != query:
                    return variation
            
            return query
            
        except Exception:
            return query
    
    def _extract_context_from_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract contextual terms from results."""
        try:
            context_terms = []
            
            for result in results:
                # Extract from metadata
                metadata = result.get('metadata', {})
                if 'keywords' in metadata:
                    context_terms.extend(metadata['keywords'][:2])
                if 'tags' in metadata:
                    context_terms.extend(metadata['tags'][:2])
            
            # Remove duplicates and return
            return list(set(context_terms))[:5]
            
        except Exception as e:
            logger.error(f"Context extraction failed: {e}")
            return []
    
    def _queries_too_similar(self, recent_refinements: List[Dict[str, Any]]) -> bool:
        """Check if recent query refinements are too similar."""
        try:
            if len(recent_refinements) < 2:
                return False
            
            query1 = recent_refinements[-2]['refined_query']
            query2 = recent_refinements[-1]['refined_query']
            
            # Simple similarity check
            words1 = set(query1.lower().split())
            words2 = set(query2.lower().split())
            
            if not words1 or not words2:
                return True
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            similarity = intersection / union if union > 0 else 0
            
            return similarity > self.similarity_threshold
            
        except Exception:
            return False
    
    def _get_refinement_reason(self,
                             strategy: SearchRefinementStrategy,
                             total_results: int,
                             target_result_count: int) -> str:
        """Get human-readable reason for refinement strategy."""
        if strategy == SearchRefinementStrategy.EXPANSION:
            return f"Expanding query to find more results (current: {total_results}, target: {target_result_count})"
        elif strategy == SearchRefinementStrategy.NARROWING:
            return f"Narrowing query to reduce result count (current: {total_results}, target: {target_result_count})"
        elif strategy == SearchRefinementStrategy.REFORMULATION:
            return "Reformulating query with different terms for better results"
        elif strategy == SearchRefinementStrategy.SEMANTIC_SHIFT:
            return "Shifting query semantically based on result patterns"
        elif strategy == SearchRefinementStrategy.CONTEXT_ADDITION:
            return "Adding contextual information to improve relevance"
        else:
            return "Unknown refinement strategy"    
  
  def _generate_cache_key(self,
                          query: str,
                          document_ids: Optional[List[str]],
                          target_result_count: int) -> str:
        """Generate cache key for iterative search."""
        try:
            doc_ids_str = ':'.join(sorted(document_ids)) if document_ids else 'all'
            content_hash = hash(f"{query}:{doc_ids_str}:{target_result_count}")
            return f"iterative_search:{content_hash}"
        except Exception:
            return f"iterative_search:{hash(query)}"
    
    def _create_error_response(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create error response for iterative search."""
        return {
            'original_query': '',
            'final_query': '',
            'results': [],
            'total_results': 0,
            'iterations_performed': 0,
            'refinement_history': [],
            'performance_metrics': {
                'query_variations': [],
                'result_counts': [],
                'search_times': []
            },
            'search_time': time.time() - start_time,
            'success': False,
            'error': error_message
        }
    
    def _update_search_stats(self, response: Dict[str, Any]):
        """Update search performance statistics."""
        try:
            self.search_stats['total_searches'] += 1
            
            # Update average iterations
            iterations = response.get('iterations_performed', 0)
            total_iterations = self.search_stats['avg_iterations'] * (self.search_stats['total_searches'] - 1)
            self.search_stats['avg_iterations'] = (total_iterations + iterations) / self.search_stats['total_searches']
            
            # Update average search time
            search_time = response.get('search_time', 0)
            total_time = self.search_stats['avg_search_time'] * (self.search_stats['total_searches'] - 1)
            self.search_stats['avg_search_time'] = (total_time + search_time) / self.search_stats['total_searches']
            
            # Update success/failure counts
            if response.get('success', False) and response.get('total_results', 0) > 0:
                self.search_stats['successful_refinements'] += 1
            else:
                self.search_stats['failed_refinements'] += 1
                
        except Exception as e:
            logger.error(f"Search stats update failed: {e}")
    
    async def analyze_query_performance(self,
                                      queries: List[str],
                                      document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze performance of iterative search for multiple queries.
        
        Args:
            queries: List of queries to analyze
            document_ids: Optional document IDs to limit search scope
            
        Returns:
            Performance analysis results
        """
        try:
            analysis_results = {
                'total_queries': len(queries),
                'successful_searches': 0,
                'failed_searches': 0,
                'avg_iterations': 0.0,
                'avg_search_time': 0.0,
                'avg_results_found': 0.0,
                'refinement_strategy_usage': defaultdict(int),
                'query_performance': []
            }
            
            total_iterations = 0
            total_search_time = 0.0
            total_results = 0
            
            for query in queries:
                try:
                    result = await self.iterative_search(
                        initial_query=query,
                        document_ids=document_ids,
                        target_result_count=10
                    )
                    
                    if result.get('success', False):
                        analysis_results['successful_searches'] += 1
                    else:
                        analysis_results['failed_searches'] += 1
                    
                    # Collect metrics
                    iterations = result.get('iterations_performed', 0)
                    search_time = result.get('search_time', 0)
                    results_count = result.get('total_results', 0)
                    
                    total_iterations += iterations
                    total_search_time += search_time
                    total_results += results_count
                    
                    # Count refinement strategies
                    for refinement in result.get('refinement_history', []):
                        strategy = refinement.get('strategy', 'unknown')
                        analysis_results['refinement_strategy_usage'][strategy] += 1
                    
                    # Store individual query performance
                    analysis_results['query_performance'].append({
                        'query': query,
                        'success': result.get('success', False),
                        'iterations': iterations,
                        'search_time': search_time,
                        'results_found': results_count,
                        'refinement_strategies': [r.get('strategy') for r in result.get('refinement_history', [])]
                    })
                    
                except Exception as query_error:
                    logger.error(f"Query analysis failed for '{query}': {query_error}")
                    analysis_results['failed_searches'] += 1
            
            # Calculate averages
            if analysis_results['total_queries'] > 0:
                analysis_results['avg_iterations'] = total_iterations / analysis_results['total_queries']
                analysis_results['avg_search_time'] = total_search_time / analysis_results['total_queries']
                analysis_results['avg_results_found'] = total_results / analysis_results['total_queries']
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Query performance analysis failed: {e}")
            return {'error': str(e)}
    
    async def optimize_search_parameters(self,
                                       test_queries: List[str],
                                       document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Optimize search parameters based on test queries.
        
        Args:
            test_queries: List of test queries for optimization
            document_ids: Optional document IDs to limit search scope
            
        Returns:
            Optimization results with recommended parameters
        """
        try:
            optimization_results = {
                'original_parameters': {
                    'max_iterations': self.max_iterations,
                    'min_results_threshold': self.min_results_threshold,
                    'max_results_threshold': self.max_results_threshold,
                    'similarity_threshold': self.similarity_threshold
                },
                'tested_configurations': [],
                'best_configuration': None,
                'best_score': 0.0
            }
            
            # Test different parameter combinations
            test_configurations = [
                {'max_iterations': 3, 'min_results_threshold': 2, 'max_results_threshold': 15},
                {'max_iterations': 5, 'min_results_threshold': 3, 'max_results_threshold': 20},
                {'max_iterations': 7, 'min_results_threshold': 4, 'max_results_threshold': 25},
            ]
            
            for config in test_configurations:
                # Temporarily update parameters
                original_max_iterations = self.max_iterations
                original_min_threshold = self.min_results_threshold
                original_max_threshold = self.max_results_threshold
                
                self.max_iterations = config['max_iterations']
                self.min_results_threshold = config['min_results_threshold']
                self.max_results_threshold = config['max_results_threshold']
                
                try:
                    # Test configuration
                    performance = await self.analyze_query_performance(test_queries[:5], document_ids)  # Limit for speed
                    
                    # Calculate score (higher is better)
                    success_rate = performance['successful_searches'] / max(performance['total_queries'], 1)
                    avg_results = performance['avg_results_found']
                    efficiency = 1.0 / max(performance['avg_search_time'], 0.1)  # Inverse of time
                    
                    score = (success_rate * 0.5) + (min(avg_results / 10.0, 1.0) * 0.3) + (min(efficiency, 1.0) * 0.2)
                    
                    config_result = {
                        'configuration': config.copy(),
                        'performance': performance,
                        'score': score
                    }
                    
                    optimization_results['tested_configurations'].append(config_result)
                    
                    if score > optimization_results['best_score']:
                        optimization_results['best_score'] = score
                        optimization_results['best_configuration'] = config.copy()
                    
                finally:
                    # Restore original parameters
                    self.max_iterations = original_max_iterations
                    self.min_results_threshold = original_min_threshold
                    self.max_results_threshold = original_max_threshold
            
            # Apply best configuration if found
            if optimization_results['best_configuration']:
                best_config = optimization_results['best_configuration']
                self.max_iterations = best_config['max_iterations']
                self.min_results_threshold = best_config['min_results_threshold']
                self.max_results_threshold = best_config['max_results_threshold']
                
                logger.info(f"Applied optimized parameters: {best_config}")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return {'error': str(e)}
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        try:
            return {
                'configuration': {
                    'max_iterations': self.max_iterations,
                    'min_results_threshold': self.min_results_threshold,
                    'max_results_threshold': self.max_results_threshold,
                    'similarity_threshold': self.similarity_threshold,
                    'enable_caching': self.enable_caching
                },
                'performance_stats': self.search_stats.copy(),
                'available_strategies': [strategy.value for strategy in SearchRefinementStrategy],
                'system_status': {
                    'initialized': self._initialized,
                    'retriever_available': self.retriever._initialized if hasattr(self.retriever, '_initialized') else False,
                    'tokenizer_available': self.tokenizer._initialized if hasattr(self.tokenizer, '_initialized') else False,
                    'cache_available': self.cache is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Search stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
iterative_search_manager = IterativeSearchManager()