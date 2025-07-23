"""
Agentic response generation with multi-step reasoning and iterative search.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import time
import re
from enum import Enum

import trio

from app.services.search.retriever import basic_retriever
from app.services.search.iterative_search_manager import iterative_search_manager
from app.services.external_search.tavily_client import tavily_client
from app.core.config import settings
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class ReasoningStep(Enum):
    """Types of reasoning steps."""
    ANALYSIS = "analysis"
    SEARCH = "search"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    CONCLUSION = "conclusion"


class AgenticProcessor:
    """
    Multi-step agentic reasoning system with iterative search and refinement.
    """
    
    def __init__(self,
                 max_reasoning_steps: int = 5,
                 max_search_iterations: int = 3,
                 confidence_threshold: float = 0.7,
                 enable_caching: bool = True):
        """
        Initialize agentic processor.
        
        Args:
            max_reasoning_steps: Maximum number of reasoning steps
            max_search_iterations: Maximum search iterations per step
            confidence_threshold: Minimum confidence for conclusions
            enable_caching: Whether to cache reasoning results
        """
        self.max_reasoning_steps = max_reasoning_steps
        self.max_search_iterations = max_search_iterations
        self.confidence_threshold = confidence_threshold
        self.enable_caching = enable_caching
        
        self.retriever = basic_retriever
        self.cache = cache_manager
        
        # Reasoning prompts for different steps
        self.reasoning_prompts = {
            'initial_analysis': """
Analyze the following query and break it down into key components that need to be researched:

Query: "{query}"

Provide your analysis in the following JSON format:
{{
    "query_type": "factual|analytical|comparative|procedural|creative",
    "key_components": ["component1", "component2", ...],
    "search_queries": ["search1", "search2", ...],
    "complexity_level": "simple|moderate|complex",
    "reasoning_approach": "direct|multi_step|comparative|synthesis"
}}
""",
            
            'step_reasoning': """
Based on the current information and previous reasoning steps, continue the analysis:

Original Query: "{query}"
Current Step: {step_number}/{total_steps}
Previous Steps: {previous_steps}
Available Information: {available_info}

Generate the next reasoning step in JSON format:
{{
    "step_type": "analysis|search|synthesis|validation|conclusion",
    "reasoning": "Your reasoning for this step",
    "search_needed": true/false,
    "search_queries": ["query1", "query2", ...] if search_needed,
    "confidence": 0.0-1.0,
    "next_action": "continue|search|conclude"
}}
""",
            
            'information_synthesis': """
Synthesize the following information to answer the original query:

Original Query: "{query}"
Reasoning Steps: {reasoning_steps}
Search Results: {search_results}

Provide a comprehensive synthesis in JSON format:
{{
    "answer": "Your comprehensive answer",
    "confidence": 0.0-1.0,
    "supporting_evidence": ["evidence1", "evidence2", ...],
    "limitations": ["limitation1", "limitation2", ...],
    "additional_context": "Any additional relevant context"
}}
""",
            
            'quality_validation': """
Validate the quality and accuracy of the following response:

Original Query: "{query}"
Generated Answer: "{answer}"
Supporting Evidence: {evidence}
Reasoning Process: {reasoning_process}

Provide validation assessment in JSON format:
{{
    "accuracy_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "relevance_score": 0.0-1.0,
    "overall_quality": 0.0-1.0,
    "issues_found": ["issue1", "issue2", ...],
    "improvement_suggestions": ["suggestion1", "suggestion2", ...],
    "validation_passed": true/false
}}
"""
        }
        
        # Performance tracking
        self.processing_stats = {
            'total_queries': 0,
            'avg_processing_time': 0.0,
            'avg_reasoning_steps': 0.0,
            'avg_search_iterations': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'successful_conclusions': 0,
            'failed_conclusions': 0
        }
    
    async def process_query(self,
                          query: str,
                          context: Optional[List[Dict[str, Any]]] = None,
                          document_ids: Optional[List[str]] = None,
                          enable_external_search: bool = False) -> Dict[str, Any]:
        """
        Process query using agentic reasoning with multi-step analysis.
        
        Args:
            query: User query to process
            context: Optional context from previous interactions
            document_ids: Optional document IDs to limit search scope
            enable_external_search: Whether to enable external search
            
        Returns:
            Agentic response with reasoning steps and final answer
        """
        start_time = time.time()
        
        try:
            if not query.strip():
                return self._create_error_response("Empty query provided", start_time)
            
            # Initialize retriever
            if not self.retriever._initialized:
                await self.retriever.initialize()
            
            # Check cache
            if self.enable_caching:
                cache_key = self._generate_cache_key(query, document_ids)
                cached_result = await self.cache.get(cache_key)
                
                if cached_result:
                    self.processing_stats['cache_hits'] += 1
                    logger.debug(f"Using cached agentic result for query: {query}")
                    return cached_result
                
                self.processing_stats['cache_misses'] += 1
            
            # Step 1: Initial query analysis
            analysis_result = await self._analyze_query(query)
            
            if not analysis_result['success']:
                return self._create_error_response(analysis_result['error'], start_time)
            
            analysis = analysis_result['analysis']
            
            # Step 2: Multi-step reasoning process
            reasoning_result = await self._execute_reasoning_process(
                query, analysis, context, document_ids, enable_external_search
            )
            
            if not reasoning_result['success']:
                return self._create_error_response(reasoning_result['error'], start_time)
            
            # Step 3: Generate final response
            final_response = await self._generate_final_response(
                query, analysis, reasoning_result['reasoning_steps'], reasoning_result['collected_info']
            )
            
            # Step 4: Validate response quality
            validation_result = await self._validate_response_quality(
                query, final_response, reasoning_result['reasoning_steps']
            )
            
            # Create comprehensive result
            result = {
                'query': query,
                'answer': final_response['answer'],
                'confidence': final_response['confidence'],
                'reasoning_steps': reasoning_result['reasoning_steps'],
                'search_iterations': reasoning_result['search_iterations'],
                'supporting_evidence': final_response.get('supporting_evidence', []),
                'limitations': final_response.get('limitations', []),
                'additional_context': final_response.get('additional_context', ''),
                'validation': validation_result,
                'processing_time': time.time() - start_time,
                'query_analysis': analysis,
                'success': True
            }
            
            # Cache result
            if self.enable_caching:
                await self.cache.set(cache_key, result, ttl=3600)
            
            # Update stats
            self._update_processing_stats(result)
            
            logger.info(f"Agentic processing completed for query in {result['processing_time']:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Agentic processing failed: {e}")
            return self._create_error_response(str(e), start_time)
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine processing approach."""
        try:
            prompt = self.reasoning_prompts['initial_analysis'].format(query=query)
            
            # Call LLM for analysis
            analysis_response = await self._call_llm(prompt)
            
            if not analysis_response:
                return {'success': False, 'error': 'Failed to analyze query'}
            
            # Parse JSON response
            try:
                analysis = json.loads(analysis_response)
                
                # Validate required fields
                required_fields = ['query_type', 'key_components', 'search_queries', 'complexity_level', 'reasoning_approach']
                if not all(field in analysis for field in required_fields):
                    raise ValueError("Missing required analysis fields")
                
                return {'success': True, 'analysis': analysis}
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse analysis response: {e}")
                # Fallback to heuristic analysis
                return {'success': True, 'analysis': self._heuristic_query_analysis(query)}
                
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _heuristic_query_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback heuristic query analysis."""
        query_lower = query.lower()
        
        # Determine query type
        if any(word in query_lower for word in ['what is', 'define', 'explain', 'meaning']):
            query_type = 'factual'
        elif any(word in query_lower for word in ['how to', 'steps', 'process', 'procedure']):
            query_type = 'procedural'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            query_type = 'comparative'
        elif any(word in query_lower for word in ['analyze', 'evaluate', 'assess', 'why']):
            query_type = 'analytical'
        else:
            query_type = 'factual'
        
        # Extract key components (simple word extraction)
        key_components = [word for word in query.split() if len(word) > 3 and word.lower() not in ['what', 'how', 'why', 'when', 'where', 'which']]
        
        # Generate search queries
        search_queries = [query]
        if len(key_components) > 1:
            search_queries.extend(key_components[:3])
        
        # Determine complexity
        complexity_level = 'complex' if len(query.split()) > 10 else 'moderate' if len(query.split()) > 5 else 'simple'
        
        # Determine reasoning approach
        reasoning_approach = 'comparative' if query_type == 'comparative' else 'multi_step' if complexity_level == 'complex' else 'direct'
        
        return {
            'query_type': query_type,
            'key_components': key_components,
            'search_queries': search_queries,
            'complexity_level': complexity_level,
            'reasoning_approach': reasoning_approach
        }
    
    async def _execute_reasoning_process(self,
                                       query: str,
                                       analysis: Dict[str, Any],
                                       context: Optional[List[Dict[str, Any]]],
                                       document_ids: Optional[List[str]],
                                       enable_external_search: bool) -> Dict[str, Any]:
        """Execute multi-step reasoning process."""
        try:
            reasoning_steps = []
            collected_info = []
            search_iterations = 0
            
            # Determine number of steps based on complexity
            complexity = analysis.get('complexity_level', 'moderate')
            if complexity == 'simple':
                max_steps = 2
            elif complexity == 'moderate':
                max_steps = 3
            else:
                max_steps = self.max_reasoning_steps
            
            for step_num in range(1, max_steps + 1):
                step_result = await self._execute_reasoning_step(
                    query, analysis, reasoning_steps, collected_info, 
                    step_num, max_steps, document_ids, enable_external_search
                )
                
                if not step_result['success']:
                    break
                
                reasoning_steps.append(step_result['step'])
                
                # Perform search if needed
                if step_result['step'].get('search_needed', False):
                    search_result = await self._perform_iterative_search(
                        step_result['step'].get('search_queries', []),
                        document_ids,
                        enable_external_search
                    )
                    
                    collected_info.extend(search_result['results'])
                    search_iterations += search_result['iterations']
                
                # Check if we should conclude
                if step_result['step'].get('next_action') == 'conclude':
                    break
                
                # Check confidence threshold
                if step_result['step'].get('confidence', 0) >= self.confidence_threshold:
                    if step_num >= 2:  # Minimum 2 steps
                        break
            
            return {
                'success': True,
                'reasoning_steps': reasoning_steps,
                'collected_info': collected_info,
                'search_iterations': search_iterations
            }
            
        except Exception as e:
            logger.error(f"Reasoning process failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_reasoning_step(self,
                                    query: str,
                                    analysis: Dict[str, Any],
                                    previous_steps: List[Dict[str, Any]],
                                    available_info: List[Dict[str, Any]],
                                    step_number: int,
                                    total_steps: int,
                                    document_ids: Optional[List[str]],
                                    enable_external_search: bool) -> Dict[str, Any]:
        """Execute a single reasoning step."""
        try:
            # Prepare context for reasoning
            previous_steps_text = json.dumps(previous_steps, indent=2) if previous_steps else "None"
            available_info_text = json.dumps(available_info[:5], indent=2) if available_info else "None"  # Limit context
            
            prompt = self.reasoning_prompts['step_reasoning'].format(
                query=query,
                step_number=step_number,
                total_steps=total_steps,
                previous_steps=previous_steps_text,
                available_info=available_info_text
            )
            
            # Call LLM for reasoning step
            step_response = await self._call_llm(prompt)
            
            if not step_response:
                return {'success': False, 'error': 'Failed to generate reasoning step'}
            
            # Parse JSON response
            try:
                step_data = json.loads(step_response)
                
                # Validate step data
                required_fields = ['step_type', 'reasoning', 'confidence', 'next_action']
                if not all(field in step_data for field in required_fields):
                    raise ValueError("Missing required step fields")
                
                # Add metadata
                step_data['step_number'] = step_number
                step_data['timestamp'] = time.time()
                
                return {'success': True, 'step': step_data}
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse reasoning step: {e}")
                # Fallback step
                fallback_step = {
                    'step_type': 'analysis',
                    'reasoning': f'Analyzing query components for step {step_number}',
                    'search_needed': step_number <= 2,
                    'search_queries': analysis.get('search_queries', [query])[:2],
                    'confidence': 0.5,
                    'next_action': 'continue' if step_number < total_steps else 'conclude',
                    'step_number': step_number,
                    'timestamp': time.time()
                }
                return {'success': True, 'step': fallback_step}
                
        except Exception as e:
            logger.error(f"Reasoning step execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _perform_iterative_search(self,
                                      search_queries: List[str],
                                      document_ids: Optional[List[str]],
                                      enable_external_search: bool) -> Dict[str, Any]:
        """Perform iterative search with query refinement using iterative search manager."""
        try:
            # Initialize iterative search manager if needed
            if not iterative_search_manager._initialized:
                await iterative_search_manager.initialize()
            
            all_results = []
            total_iterations = 0
            
            # Use iterative search manager for each query
            for query in search_queries[:self.max_search_iterations]:
                # Perform internal search
                search_result = await iterative_search_manager.iterative_search(
                    initial_query=query,
                    document_ids=document_ids,
                    target_result_count=8,  # Target fewer results per query
                    context={'enable_external_search': enable_external_search}
                )
                
                if search_result.get('success', False) and search_result.get('results'):
                    # Add search metadata to results
                    for result in search_result['results']:
                        result['search_query'] = query
                        result['iterative_search_used'] = True
                        result['refinement_iterations'] = search_result.get('iterations_performed', 0)
                    
                    all_results.extend(search_result['results'])
                    total_iterations += search_result.get('iterations_performed', 0)
                
                # Perform external search if enabled and we need more results
                if enable_external_search and len(all_results) < 10:
                    external_results = await self._perform_external_search(query)
                    if external_results.get('success', False):
                        all_results.extend(external_results['results'])
                
                # Break if we have enough results
                if len(all_results) >= 15:
                    break
            
            # Deduplicate results by chunk_id
            seen_chunks = set()
            unique_results = []
            for result in all_results:
                chunk_id = result.get('chunk_id')
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    unique_results.append(result)
            
            # Sort by relevance score
            unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            return {
                'results': unique_results[:10],  # Limit final results
                'iterations': total_iterations,
                'total_found': len(unique_results),
                'iterative_refinement_used': True
            }
            
        except Exception as e:
            logger.error(f"Iterative search failed: {e}")
            # Fallback to basic search
            return await self._fallback_basic_search(search_queries, document_ids)
    
    async def _fallback_basic_search(self,
                                   search_queries: List[str],
                                   document_ids: Optional[List[str]]) -> Dict[str, Any]:
        """Fallback to basic search when iterative search fails."""
        try:
            all_results = []
            iterations = 0
            
            for query in search_queries[:self.max_search_iterations]:
                iterations += 1
                
                # Perform basic search
                search_result = await self.retriever.search(
                    query=query,
                    document_ids=document_ids,
                    search_type="hybrid",
                    top_k=5,
                    enable_reranking=True
                )
                
                if search_result.get('results'):
                    for result in search_result['results']:
                        result['search_query'] = query
                        result['search_iteration'] = iterations
                        result['iterative_search_used'] = False
                    all_results.extend(search_result['results'])
                
                # Break if we have enough results
                if len(all_results) >= 10:
                    break
            
            # Deduplicate results by chunk_id
            seen_chunks = set()
            unique_results = []
            for result in all_results:
                chunk_id = result.get('chunk_id')
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    unique_results.append(result)
            
            return {
                'results': unique_results,
                'iterations': iterations,
                'total_found': len(unique_results),
                'iterative_refinement_used': False
            }
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return {'results': [], 'iterations': 0, 'total_found': 0, 'iterative_refinement_used': False}
    
    async def _perform_external_search(self, query: str) -> Dict[str, Any]:
        """Perform external search using Tavily API."""
        try:
            # Initialize Tavily client if needed
            if not tavily_client._initialized:
                await tavily_client.initialize()
            
            if not tavily_client._initialized:
                logger.warning("Tavily client not initialized, skipping external search")
                return {'success': False, 'results': []}
            
            # Perform external search
            search_result = await tavily_client.retrieve_chunks(
                question=query,
                max_results=5
            )
            
            if search_result.get('success', False):
                # Convert chunks to standard format
                external_results = []
                for chunk in search_result.get('chunks', []):
                    external_result = {
                        'chunk_id': chunk['chunk_id'],
                        'content': chunk['content'],
                        'title': chunk.get('title', ''),
                        'url': chunk.get('url', ''),
                        'score': chunk.get('score', 0.5),
                        'search_type': 'external',
                        'source': 'tavily',
                        'metadata': {
                            **chunk.get('metadata', {}),
                            'external_search': True,
                            'search_query': query
                        }
                    }
                    external_results.append(external_result)
                
                logger.info(f"External search found {len(external_results)} results for query: {query}")
                
                return {
                    'success': True,
                    'results': external_results,
                    'total_found': len(external_results),
                    'search_time': search_result.get('search_time', 0)
                }
            else:
                logger.warning(f"External search failed: {search_result.get('error', 'Unknown error')}")
                return {'success': False, 'results': []}
                
        except Exception as e:
            logger.error(f"External search failed: {e}")
            return {'success': False, 'results': []}
    
    async def _generate_final_response(self,
                                     query: str,
                                     analysis: Dict[str, Any],
                                     reasoning_steps: List[Dict[str, Any]],
                                     collected_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final comprehensive response."""
        try:
            # Prepare context for synthesis
            reasoning_steps_text = json.dumps(reasoning_steps, indent=2)
            search_results_text = json.dumps(collected_info[:10], indent=2)  # Limit context
            
            prompt = self.reasoning_prompts['information_synthesis'].format(
                query=query,
                reasoning_steps=reasoning_steps_text,
                search_results=search_results_text
            )
            
            # Call LLM for synthesis
            synthesis_response = await self._call_llm(prompt)
            
            if not synthesis_response:
                return self._create_fallback_response(query, collected_info)
            
            # Parse JSON response
            try:
                synthesis = json.loads(synthesis_response)
                
                # Validate synthesis
                required_fields = ['answer', 'confidence']
                if not all(field in synthesis for field in required_fields):
                    raise ValueError("Missing required synthesis fields")
                
                return synthesis
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse synthesis response: {e}")
                return self._create_fallback_response(query, collected_info)
                
        except Exception as e:
            logger.error(f"Final response generation failed: {e}")
            return self._create_fallback_response(query, collected_info)
    
    def _create_fallback_response(self, query: str, collected_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create fallback response when synthesis fails."""
        if collected_info:
            # Extract content from search results
            content_pieces = []
            for info in collected_info[:5]:
                content = info.get('content', '')
                if content and len(content) > 50:
                    content_pieces.append(content[:200] + "..." if len(content) > 200 else content)
            
            answer = f"Based on the available information: {' '.join(content_pieces)}"
            confidence = 0.6
        else:
            answer = "I don't have enough information to provide a comprehensive answer to your query."
            confidence = 0.1
        
        return {
            'answer': answer,
            'confidence': confidence,
            'supporting_evidence': [info.get('content', '')[:100] for info in collected_info[:3]],
            'limitations': ['Limited information available', 'Fallback response generated'],
            'additional_context': 'This response was generated using fallback logic due to processing limitations.'
        }
    
    async def _validate_response_quality(self,
                                       query: str,
                                       response: Dict[str, Any],
                                       reasoning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of the generated response."""
        try:
            prompt = self.reasoning_prompts['quality_validation'].format(
                query=query,
                answer=response.get('answer', ''),
                evidence=json.dumps(response.get('supporting_evidence', []), indent=2),
                reasoning_process=json.dumps(reasoning_steps, indent=2)
            )
            
            # Call LLM for validation
            validation_response = await self._call_llm(prompt)
            
            if not validation_response:
                return self._create_fallback_validation(response)
            
            # Parse JSON response
            try:
                validation = json.loads(validation_response)
                
                # Validate validation response
                required_fields = ['accuracy_score', 'completeness_score', 'relevance_score', 'overall_quality']
                if not all(field in validation for field in required_fields):
                    raise ValueError("Missing required validation fields")
                
                return validation
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse validation response: {e}")
                return self._create_fallback_validation(response)
                
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return self._create_fallback_validation(response)
    
    def _create_fallback_validation(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback validation when LLM validation fails."""
        confidence = response.get('confidence', 0.5)
        
        return {
            'accuracy_score': confidence,
            'completeness_score': min(confidence + 0.1, 1.0),
            'relevance_score': confidence,
            'overall_quality': confidence,
            'issues_found': ['Validation performed using fallback logic'],
            'improvement_suggestions': ['Consider manual review of response quality'],
            'validation_passed': confidence >= self.confidence_threshold
        }
    
    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM service for reasoning."""
        try:
            # Try to use actual LLM service
            from app.core.config import settings
            
            try:
                from app.services.llm.yandex_client import YandexLLMClient
                
                if not hasattr(self, '_llm_client'):
                    self._llm_client = YandexLLMClient(
                        api_key=settings.YANDEX_API_KEY,
                        folder_id=settings.YANDEX_FOLDER_ID
                    )
                
                response = await self._llm_client.generate_text(
                    prompt=prompt,
                    temperature=0.3,  # Moderate temperature for reasoning
                    max_tokens=1000
                )
                
                return response
                
            except ImportError:
                logger.warning("Yandex LLM client not available")
                return None
            except Exception as llm_error:
                logger.warning(f"LLM call failed: {llm_error}")
                return None
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
    
    def _generate_cache_key(self, query: str, document_ids: Optional[List[str]]) -> str:
        """Generate cache key for agentic processing."""
        try:
            doc_ids_str = ':'.join(sorted(document_ids)) if document_ids else 'all'
            content_hash = hash(f"{query}:{doc_ids_str}")
            return f"agentic_process:{content_hash}"
        except Exception:
            return f"agentic_process:{hash(query)}"
    
    def _create_error_response(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create error response."""
        return {
            'query': '',
            'answer': f"I encountered an error while processing your query: {error_message}",
            'confidence': 0.0,
            'reasoning_steps': [],
            'search_iterations': 0,
            'supporting_evidence': [],
            'limitations': [error_message],
            'additional_context': 'Error occurred during agentic processing',
            'validation': {'overall_quality': 0.0, 'validation_passed': False},
            'processing_time': time.time() - start_time,
            'success': False,
            'error': error_message
        }
    
    def _update_processing_stats(self, result: Dict[str, Any]):
        """Update processing statistics."""
        try:
            self.processing_stats['total_queries'] += 1
            
            # Update average processing time
            total_time = self.processing_stats['avg_processing_time'] * (self.processing_stats['total_queries'] - 1)
            self.processing_stats['avg_processing_time'] = (total_time + result['processing_time']) / self.processing_stats['total_queries']
            
            # Update average reasoning steps
            steps_count = len(result.get('reasoning_steps', []))
            total_steps = self.processing_stats['avg_reasoning_steps'] * (self.processing_stats['total_queries'] - 1)
            self.processing_stats['avg_reasoning_steps'] = (total_steps + steps_count) / self.processing_stats['total_queries']
            
            # Update average search iterations
            search_iterations = result.get('search_iterations', 0)
            total_iterations = self.processing_stats['avg_search_iterations'] * (self.processing_stats['total_queries'] - 1)
            self.processing_stats['avg_search_iterations'] = (total_iterations + search_iterations) / self.processing_stats['total_queries']
            
            # Update success/failure counts
            if result.get('success', False) and result.get('validation', {}).get('validation_passed', False):
                self.processing_stats['successful_conclusions'] += 1
            else:
                self.processing_stats['failed_conclusions'] += 1
                
        except Exception as e:
            logger.error(f"Stats update failed: {e}")
    
    async def get_processor_stats(self) -> Dict[str, Any]:
        """Get agentic processor statistics."""
        try:
            return {
                'configuration': {
                    'max_reasoning_steps': self.max_reasoning_steps,
                    'max_search_iterations': self.max_search_iterations,
                    'confidence_threshold': self.confidence_threshold,
                    'enable_caching': self.enable_caching
                },
                'performance_stats': self.processing_stats.copy(),
                'available_reasoning_types': list(self.reasoning_prompts.keys()),
                'system_status': {
                    'retriever_initialized': self.retriever._initialized if hasattr(self.retriever, '_initialized') else False,
                    'cache_available': self.cache is not None,
                    'llm_client_available': hasattr(self, '_llm_client')
                }
            }
            
        except Exception as e:
            logger.error(f"Processor stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
agentic_processor = AgenticProcessor()