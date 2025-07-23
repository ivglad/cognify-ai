"""
Tests for Tavily external search integration.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import json

from app.services.external_search.tavily_client import TavilySearchClient


@pytest.fixture
def sample_tavily_response():
    """Sample Tavily API response for testing."""
    return {
        "answer": "Machine learning is a method of data analysis that automates analytical model building.",
        "results": [
            {
                "title": "What is Machine Learning? | IBM",
                "url": "https://www.ibm.com/topics/machine-learning",
                "content": "Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.",
                "score": 0.95,
                "published_date": "2023-01-15"
            },
            {
                "title": "Machine Learning Explained | MIT",
                "url": "https://web.mit.edu/machine-learning",
                "content": "Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.",
                "score": 0.88,
                "published_date": "2023-02-20"
            },
            {
                "title": "Introduction to Machine Learning",
                "url": "https://example.com/ml-intro",
                "content": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
                "score": 0.82,
                "published_date": "2023-03-10"
            }
        ]
    }


@pytest.fixture
def tavily_client():
    """Create Tavily client instance for testing."""
    with patch('app.services.external_search.tavily_client.cache_manager') as mock_cache:
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        
        client = TavilySearchClient(
            api_key="test_api_key",
            max_results=10,
            search_depth="basic",
            enable_caching=True,
            cache_ttl=3600
        )
        return client


class TestTavilySearchClient:
    """Test cases for Tavily search integration."""
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, tavily_client):
        """Test successful initialization with API key."""
        with patch.object(tavily_client, '_test_api_connection', return_value=True):
            await tavily_client.initialize()
            
            assert tavily_client._initialized == True
    
    @pytest.mark.asyncio
    async def test_initialization_no_api_key(self):
        """Test initialization without API key."""
        client = TavilySearchClient(api_key=None)
        
        await client.initialize()
        
        assert client._initialized == False
    
    @pytest.mark.asyncio
    async def test_initialization_api_test_failure(self, tavily_client):
        """Test initialization when API test fails."""
        with patch.object(tavily_client, '_test_api_connection', return_value=False):
            await tavily_client.initialize()
            
            assert tavily_client._initialized == False
    
    @pytest.mark.asyncio
    async def test_search_success(self, tavily_client, sample_tavily_response):
        """Test successful search operation."""
        query = "what is machine learning"
        
        # Mock successful API response
        with patch.object(tavily_client, '_perform_tavily_search', return_value={
            'success': True,
            'raw_results': sample_tavily_response
        }):
            tavily_client._initialized = True
            
            result = await tavily_client.search(query)
            
            assert result['success'] == True
            assert result['query'] == query
            assert len(result['results']) == 3
            assert result['total_results'] == 3
            assert 'answer' in result
            assert result['source'] == 'tavily'
    
    @pytest.mark.asyncio
    async def test_search_empty_query(self, tavily_client):
        """Test search with empty query."""
        tavily_client._initialized = True
        
        result = await tavily_client.search("")
        
        assert result['success'] == False
        assert 'Empty query' in result.get('error', '')
    
    @pytest.mark.asyncio
    async def test_search_not_initialized(self, tavily_client):
        """Test search when client is not initialized."""
        tavily_client._initialized = False
        
        with patch.object(tavily_client, 'initialize', side_effect=Exception("Init failed")):
            result = await tavily_client.search("test query")
            
            assert result['success'] == False
            assert 'not initialized' in result.get('error', '')
    
    @pytest.mark.asyncio
    async def test_search_with_parameters(self, tavily_client, sample_tavily_response):
        """Test search with custom parameters."""
        query = "machine learning"
        
        with patch.object(tavily_client, '_perform_tavily_search', return_value={
            'success': True,
            'raw_results': sample_tavily_response
        }) as mock_search:
            tavily_client._initialized = True
            
            result = await tavily_client.search(
                query=query,
                max_results=5,
                search_depth="advanced",
                include_domains=["ibm.com", "mit.edu"],
                exclude_domains=["example.com"]
            )
            
            # Check that parameters were passed correctly
            call_args = mock_search.call_args[0][0]
            assert call_args['max_results'] == 5
            assert call_args['search_depth'] == "advanced"
            assert call_args['include_domains'] == ["ibm.com", "mit.edu"]
            assert call_args['exclude_domains'] == ["example.com"]
            
            assert result['success'] == True
    
    @pytest.mark.asyncio
    async def test_api_request_success(self, tavily_client, sample_tavily_response):
        """Test successful API request."""
        search_params = {
            "query": "machine learning",
            "max_results": 5,
            "search_depth": "basic"
        }
        
        # Mock httpx response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_tavily_response
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            result = await tavily_client._perform_tavily_search(search_params)
            
            assert result['success'] == True
            assert result['raw_results'] == sample_tavily_response
    
    @pytest.mark.asyncio
    async def test_api_request_failure(self, tavily_client):
        """Test API request failure."""
        search_params = {
            "query": "test",
            "max_results": 5,
            "search_depth": "basic"
        }
        
        # Mock httpx error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            result = await tavily_client._perform_tavily_search(search_params)
            
            assert result['success'] == False
            assert 'API error' in result['error']
    
    @pytest.mark.asyncio
    async def test_result_processing(self, tavily_client, sample_tavily_response):
        """Test processing of search results."""
        query = "machine learning"
        
        result = await tavily_client._process_search_results(sample_tavily_response, query)
        
        assert result['total_found'] == 3
        assert len(result['results']) == 3
        
        # Check first result processing
        first_result = result['results'][0]
        assert first_result['id'] == 'tavily_0'
        assert first_result['title'] == "What is Machine Learning? | IBM"
        assert first_result['url'] == "https://www.ibm.com/topics/machine-learning"
        assert first_result['source'] == 'tavily'
        assert first_result['search_query'] == query
        assert 'metadata' in first_result
        assert first_result['metadata']['domain'] == 'www.ibm.com'
    
    @pytest.mark.asyncio
    async def test_retrieve_chunks(self, tavily_client, sample_tavily_response):
        """Test chunk retrieval for RAG integration."""
        question = "what is machine learning"
        
        with patch.object(tavily_client, 'search', return_value={
            'success': True,
            'results': [
                {
                    'id': 'tavily_0',
                    'title': 'ML Title',
                    'content': 'ML content',
                    'url': 'https://example.com',
                    'score': 0.9,
                    'metadata': {'domain': 'example.com'}
                }
            ],
            'answer': 'ML is a subset of AI',
            'search_time': 0.5
        }):
            result = await tavily_client.retrieve_chunks(question, max_results=5)
            
            assert result['success'] == True
            assert len(result['chunks']) == 1
            assert result['total_chunks'] == 1
            assert result['answer'] == 'ML is a subset of AI'
            
            # Check chunk format
            chunk = result['chunks'][0]
            assert chunk['chunk_id'] == 'external_tavily_0'
            assert chunk['content'] == 'ML content'
            assert chunk['source'] == 'external_search'
            assert chunk['search_type'] == 'tavily'
            assert chunk['metadata']['external_source'] == True
    
    @pytest.mark.asyncio
    async def test_retrieve_chunks_search_failure(self, tavily_client):
        """Test chunk retrieval when search fails."""
        question = "test question"
        
        with patch.object(tavily_client, 'search', return_value={
            'success': False,
            'error': 'Search failed'
        }):
            result = await tavily_client.retrieve_chunks(question)
            
            assert result['success'] == False
            assert result['total_chunks'] == 0
            assert len(result['chunks']) == 0
            assert 'Search failed' in result['error']
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, tavily_client, sample_tavily_response):
        """Test caching of search results."""
        query = "machine learning"
        
        # Mock cache hit
        cached_result = {
            'query': query,
            'results': [],
            'success': True
        }
        
        with patch.object(tavily_client.cache, 'get', return_value=cached_result):
            tavily_client._initialized = True
            
            result = await tavily_client.search(query)
            
            assert result == cached_result
            assert tavily_client.search_stats['cache_hits'] > 0
    
    @pytest.mark.asyncio
    async def test_api_connection_test(self, tavily_client):
        """Test API connection testing."""
        # Mock successful test
        with patch.object(tavily_client, '_perform_tavily_search', return_value={
            'success': True,
            'raw_results': {'results': []}
        }):
            result = await tavily_client._test_api_connection()
            
            assert result == True
        
        # Mock failed test
        with patch.object(tavily_client, '_perform_tavily_search', return_value={
            'success': False,
            'error': 'API error'
        }):
            result = await tavily_client._test_api_connection()
            
            assert result == False
    
    def test_domain_extraction(self, tavily_client):
        """Test domain extraction from URLs."""
        test_cases = [
            ("https://www.ibm.com/topics/machine-learning", "www.ibm.com"),
            ("http://example.com/page", "example.com"),
            ("https://subdomain.example.org/path?query=1", "subdomain.example.org"),
            ("invalid-url", "")
        ]
        
        for url, expected_domain in test_cases:
            domain = tavily_client._extract_domain(url)
            assert domain == expected_domain
    
    def test_cache_key_generation(self, tavily_client):
        """Test cache key generation."""
        query = "machine learning"
        max_results = 10
        search_depth = "basic"
        include_domains = ["ibm.com"]
        exclude_domains = ["example.com"]
        
        cache_key = tavily_client._generate_cache_key(
            query, max_results, search_depth, include_domains, exclude_domains
        )
        
        assert cache_key.startswith("tavily_search:")
        assert isinstance(cache_key, str)
        
        # Same inputs should generate same key
        cache_key2 = tavily_client._generate_cache_key(
            query, max_results, search_depth, include_domains, exclude_domains
        )
        assert cache_key == cache_key2
        
        # Different inputs should generate different keys
        cache_key3 = tavily_client._generate_cache_key(
            "different query", max_results, search_depth, include_domains, exclude_domains
        )
        assert cache_key != cache_key3
    
    def test_error_response_creation(self, tavily_client):
        """Test error response creation."""
        error_message = "Test error"
        start_time = 1000.0
        
        with patch('time.time', return_value=1001.5):
            error_response = tavily_client._create_error_response(error_message, start_time)
            
            assert error_response['success'] == False
            assert error_response['error'] == error_message
            assert error_response['search_time'] == 1.5
            assert error_response['source'] == 'tavily'
            assert error_response['total_results'] == 0
    
    def test_search_stats_update(self, tavily_client):
        """Test search statistics update."""
        initial_searches = tavily_client.search_stats['total_searches']
        
        # Test successful search stats
        response = {
            'success': True,
            'total_results': 5,
            'search_time': 1.2
        }
        
        tavily_client._update_search_stats(response, True)
        
        assert tavily_client.search_stats['total_searches'] == initial_searches + 1
        assert tavily_client.search_stats['successful_searches'] > 0
        assert tavily_client.search_stats['total_results_retrieved'] >= 5
        
        # Test failed search stats
        error_response = {
            'success': False,
            'search_time': 0.5
        }
        
        tavily_client._update_search_stats(error_response, False)
        
        assert tavily_client.search_stats['failed_searches'] > 0
    
    @pytest.mark.asyncio
    async def test_get_client_stats(self, tavily_client):
        """Test client statistics retrieval."""
        stats = await tavily_client.get_client_stats()
        
        assert 'configuration' in stats
        assert 'performance_stats' in stats
        assert 'system_status' in stats
        
        # Check configuration values
        config = stats['configuration']
        assert config['api_key_configured'] == True
        assert config['max_results'] == 10
        assert config['search_depth'] == "basic"
        assert config['enable_caching'] == True
        assert config['cache_ttl'] == 3600
        
        # Check system status
        status = stats['system_status']
        assert 'initialized' in status
        assert 'api_available' in status
        assert 'cache_available' in status
    
    @pytest.mark.asyncio
    async def test_long_content_snippet_creation(self, tavily_client):
        """Test snippet creation for long content."""
        long_content_response = {
            "results": [
                {
                    "title": "Long Content Test",
                    "url": "https://example.com",
                    "content": "A" * 1000,  # Very long content
                    "score": 0.8,
                    "published_date": "2023-01-01"
                }
            ]
        }
        
        result = await tavily_client._process_search_results(long_content_response, "test")
        
        processed_result = result['results'][0]
        assert len(processed_result['snippet']) <= 503  # 500 + "..."
        assert processed_result['snippet'].endswith("...")
        assert len(processed_result['content']) == 1000  # Original content preserved
    
    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self, tavily_client):
        """Test error handling in result processing."""
        # Test with malformed response
        malformed_response = {
            "results": [
                {
                    # Missing required fields
                    "title": "Test"
                }
            ]
        }
        
        result = await tavily_client._process_search_results(malformed_response, "test")
        
        # Should handle gracefully
        assert result['total_found'] >= 0
        assert isinstance(result['results'], list)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])