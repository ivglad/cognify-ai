"""
Tavily API integration for external web search capabilities.
"""
import logging
from typing import List, Dict, Any, Optional
import time
import json
import re

import trio
import httpx

from app.core.config import settings
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class TavilySearchClient:
    """
    Client for integrating with Tavily API for external web search.
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 max_results: int = 10,
                 search_depth: str = "basic",
                 enable_caching: bool = True,
                 cache_ttl: int = 3600):
        """
        Initialize Tavily search client.
        
        Args:
            api_key: Tavily API key (defaults to settings)
            max_results: Maximum results per search
            search_depth: Search depth ("basic" or "advanced")
            enable_caching: Whether to cache search results
            cache_ttl: Cache time-to-live in seconds
        """
        self.api_key = api_key or getattr(settings, 'TAVILY_API_KEY', None)
        self.max_results = max_results
        self.search_depth = search_depth
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        self.base_url = "https://api.tavily.com"
        self.cache = cache_manager
        
        self._initialized = False
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'avg_search_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_results_retrieved': 0
        }
    
    async def initialize(self):
        """Initialize the Tavily client."""
        if self._initialized:
            return
        
        try:
            if not self.api_key:
                logger.warning("Tavily API key not provided, external search disabled")
                self._initialized = False
                return
            
            # Test API connection
            test_result = await self._test_api_connection()
            
            if test_result:
                self._initialized = True
                logger.info("Tavily client initialized successfully")
            else:
                logger.error("Failed to initialize Tavily client - API test failed")
                self._initialized = False
                
        except Exception as e:
            logger.error(f"Failed to initialize Tavily client: {e}")
            self._initialized = False
    
    async def search(self,
                    query: str,
                    max_results: Optional[int] = None,
                    search_depth: Optional[str] = None,
                    include_domains: Optional[List[str]] = None,
                    exclude_domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform external search using Tavily API.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            search_depth: Search depth ("basic" or "advanced")
            include_domains: Domains to include in search
            exclude_domains: Domains to exclude from search
            
        Returns:
            Search results with metadata
        """
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            if not self._initialized:
                return self._create_error_response("Tavily client not initialized", start_time)
            
            if not query.strip():
                return self._create_error_response("Empty query provided", start_time)
            
            # Check cache
            if self.enable_caching:
                cache_key = self._generate_cache_key(query, max_results, search_depth, include_domains, exclude_domains)
                cached_result = await self.cache.get(cache_key)
                
                if cached_result:
                    self.search_stats['cache_hits'] += 1
                    logger.debug(f"Using cached Tavily result for query: {query}")
                    return cached_result
                
                self.search_stats['cache_misses'] += 1
            
            # Prepare search parameters
            search_params = {
                "query": query,
                "max_results": max_results or self.max_results,
                "search_depth": search_depth or self.search_depth,
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False
            }
            
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains
            
            # Perform search
            search_result = await self._perform_tavily_search(search_params)
            
            if not search_result['success']:
                return search_result
            
            # Process and format results
            processed_results = await self._process_search_results(
                search_result['raw_results'], query
            )
            
            # Create response
            response = {
                'query': query,
                'results': processed_results['results'],
                'total_results': len(processed_results['results']),
                'answer': search_result['raw_results'].get('answer', ''),
                'search_time': time.time() - start_time,
                'source': 'tavily',
                'search_params': search_params,
                'success': True
            }
            
            # Cache result
            if self.enable_caching:
                await self.cache.set(cache_key, response, ttl=self.cache_ttl)
            
            # Update stats
            self._update_search_stats(response, True)
            
            logger.info(f"Tavily search completed: {response['total_results']} results in {response['search_time']:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            error_response = self._create_error_response(str(e), start_time)
            self._update_search_stats(error_response, False)
            return error_response
    
    async def _perform_tavily_search(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual Tavily API search."""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "api_key": self.api_key,
                **search_params
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    return {
                        'success': True,
                        'raw_results': result_data
                    }
                else:
                    error_msg = f"Tavily API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {
                        'success': False,
                        'error': error_msg
                    }
                    
        except Exception as e:
            logger.error(f"Tavily API request failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _process_search_results(self,
                                    raw_results: Dict[str, Any],
                                    query: str) -> Dict[str, Any]:
        """Process and format Tavily search results."""
        try:
            processed_results = []
            
            results = raw_results.get('results', [])
            
            for i, result in enumerate(results):
                processed_result = {
                    'id': f"tavily_{i}",
                    'title': result.get('title', ''),
                    'content': result.get('content', ''),
                    'url': result.get('url', ''),
                    'score': result.get('score', 0.0),
                    'published_date': result.get('published_date', ''),
                    'source': 'tavily',
                    'search_query': query,
                    'metadata': {
                        'domain': self._extract_domain(result.get('url', '')),
                        'content_length': len(result.get('content', '')),
                        'has_title': bool(result.get('title', '')),
                        'relevance_score': result.get('score', 0.0)
                    }
                }
                
                # Add snippet if content is long
                content = processed_result['content']
                if len(content) > 500:
                    processed_result['snippet'] = content[:500] + "..."
                else:
                    processed_result['snippet'] = content
                
                processed_results.append(processed_result)
            
            return {
                'results': processed_results,
                'total_found': len(processed_results)
            }
            
        except Exception as e:
            logger.error(f"Result processing failed: {e}")
            return {
                'results': [],
                'total_found': 0
            }
    
    async def retrieve_chunks(self,
                            question: str,
                            max_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve and format search results as chunks for RAG integration.
        
        Args:
            question: Question to search for
            max_results: Maximum results to return
            
        Returns:
            Formatted chunks for RAG processing
        """
        try:
            # Perform search
            search_result = await self.search(
                query=question,
                max_results=max_results or 5
            )
            
            if not search_result.get('success', False):
                return {
                    'chunks': [],
                    'total_chunks': 0,
                    'success': False,
                    'error': search_result.get('error', 'Search failed')
                }
            
            # Convert results to chunks
            chunks = []
            for result in search_result.get('results', []):
                chunk = {
                    'chunk_id': f"external_{result['id']}",
                    'content': result['content'],
                    'title': result['title'],
                    'url': result['url'],
                    'score': result['score'],
                    'source': 'external_search',
                    'search_type': 'tavily',
                    'metadata': {
                        **result['metadata'],
                        'external_source': True,
                        'search_query': question,
                        'retrieved_at': time.time()
                    }
                }
                chunks.append(chunk)
            
            return {
                'chunks': chunks,
                'total_chunks': len(chunks),
                'answer': search_result.get('answer', ''),
                'search_time': search_result.get('search_time', 0),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Chunk retrieval failed: {e}")
            return {
                'chunks': [],
                'total_chunks': 0,
                'success': False,
                'error': str(e)
            }
    
    async def _test_api_connection(self) -> bool:
        """Test Tavily API connection."""
        try:
            test_result = await self._perform_tavily_search({
                "query": "test",
                "max_results": 1,
                "search_depth": "basic"
            })
            
            return test_result.get('success', False)
            
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc
        except Exception:
            return ""
    
    def _generate_cache_key(self,
                          query: str,
                          max_results: Optional[int],
                          search_depth: Optional[str],
                          include_domains: Optional[List[str]],
                          exclude_domains: Optional[List[str]]) -> str:
        """Generate cache key for search results."""
        try:
            key_parts = [
                query,
                str(max_results or self.max_results),
                search_depth or self.search_depth,
                ':'.join(sorted(include_domains or [])),
                ':'.join(sorted(exclude_domains or []))
            ]
            
            content_hash = hash(':'.join(key_parts))
            return f"tavily_search:{content_hash}"
            
        except Exception:
            return f"tavily_search:{hash(query)}"
    
    def _create_error_response(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create error response."""
        return {
            'query': '',
            'results': [],
            'total_results': 0,
            'answer': '',
            'search_time': time.time() - start_time,
            'source': 'tavily',
            'success': False,
            'error': error_message
        }
    
    def _update_search_stats(self, response: Dict[str, Any], success: bool):
        """Update search performance statistics."""
        try:
            self.search_stats['total_searches'] += 1
            
            if success:
                self.search_stats['successful_searches'] += 1
                self.search_stats['total_results_retrieved'] += response.get('total_results', 0)
            else:
                self.search_stats['failed_searches'] += 1
            
            # Update average search time
            search_time = response.get('search_time', 0)
            total_time = self.search_stats['avg_search_time'] * (self.search_stats['total_searches'] - 1)
            self.search_stats['avg_search_time'] = (total_time + search_time) / self.search_stats['total_searches']
            
        except Exception as e:
            logger.error(f"Search stats update failed: {e}")
    
    async def get_client_stats(self) -> Dict[str, Any]:
        """Get comprehensive client statistics."""
        try:
            return {
                'configuration': {
                    'api_key_configured': bool(self.api_key),
                    'max_results': self.max_results,
                    'search_depth': self.search_depth,
                    'enable_caching': self.enable_caching,
                    'cache_ttl': self.cache_ttl
                },
                'performance_stats': self.search_stats.copy(),
                'system_status': {
                    'initialized': self._initialized,
                    'api_available': bool(self.api_key),
                    'cache_available': self.cache is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Client stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
tavily_client = TavilySearchClient()