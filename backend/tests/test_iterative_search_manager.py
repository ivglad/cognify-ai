"""
Tests for iterative search refinement system.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from app.services.search.iterative_search_manager import (
    IterativeSearchManager, 
    SearchRefinementStrategy
)


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "content": "Machine learning is a method of data analysis that automates analytical model building using algorithms that iteratively learn from data.",
            "score": 0.9,
            "metadata": {"chunk_type": "text", "page_number": 1, "keywords": ["machine learning", "algorithms"]}
        },
        {
            "chunk_id": "chunk_2",
            "document_id": "doc_2",
            "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns.",
            "score": 0.8,
            "metadata": {"chunk_type": "text", "page_number": 2, "keywords": ["deep learning", "neural networks"]}
        },
        {
            "chunk_id": "chunk_3",
            "document_id": "doc_3",
            "content": "Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data.",
            "score": 0.7,
            "metadata": {"chunk_type": "text", "page_number": 1, "keywords": ["supervised learning", "training data"]}
        }
    ]


@pytest.fixture
def iterative_search_manager():
    """Create iterative search manager instance for testing."""
    with patch('app.services.search.iterative_search_manager.basic_retriever') as mock_retriever:
        mock_retriever._initialized = True
        mock_retriever.initialize = AsyncMock()
        mock_retriever.search = AsyncMock()
        
        with patch('app.services.search.iterative_search_manager.rag_tokenizer') as mock_tokenizer:
            mock_tokenizer._initialized = True
            mock_tokenizer.initialize = AsyncMock()
            mock_tokenizer.tokenize = Mock(return_value="machine learning algorithms")
            
            with patch('app.services.search.iterative_search_manager.cache_manager') as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()
                
                manager = IterativeSearchManager(
                    max_iterations=3,
                    min_results_threshold=2,
                    max_results_threshold=15,
                    similarity_threshold=0.8,
                    enable_caching=True
                )
                return manager


class TestIterativeSearchManager:
    """Test cases for iterative search refinement."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, iterative_search_manager):
        """Test manager initialization."""
        await iterative_search_manager.initialize()
        
        assert iterative_search_manager._initialized == True
    
    @pytest.mark.asyncio
    async def test_iterative_search_basic(self, iterative_search_manager, sample_search_results):
        """Test basic iterative search functionality."""
        query = "machine learning algorithms"
        
        # Mock search results for different iterations
        search_responses = [
            {'results': sample_search_results[:2], 'total': 2},
            {'results': sample_search_results[2:], 'total': 1}
        ]
        
        iterative_search_manager.retriever.search.side_effect = search_responses
        
        result = await iterative_search_manager.iterative_search(
            initial_query=query,
            target_result_count=5
        )
        
        assert result['success'] == True
        assert result['original_query'] == query
        assert len(result['results']) > 0
        assert result['iterations_performed'] > 0
        assert 'refinement_history' in result
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, iterative_search_manager):
        """Test handling of empty queries."""
        result = await iterative_search_manager.iterative_search("")
        
        assert result['success'] == False
        assert 'Empty query' in result.get('error', '')
    
    @pytest.mark.asyncio
    async def test_search_result_processing(self, iterative_search_manager, sample_search_results):
        """Test search result processing and deduplication."""
        seen_results = set()
        
        # Test with unique results
        new_results = iterative_search_manager._process_search_results(
            sample_search_results, seen_results
        )
        
        assert len(new_results) == len(sample_search_results)
        assert len(seen_results) == len(sample_search_results)
        
        # Test with duplicate results
        duplicate_results = iterative_search_manager._process_search_results(
            sample_search_results, seen_results
        )
        
        assert len(duplicate_results) == 0  # All should be duplicates
    
    @pytest.mark.asyncio
    async def test_query_expansion(self, iterative_search_manager):
        """Test query expansion strategy."""
        current_query = "machine learning"
        original_query = "machine learning algorithms"
        context = {'related_terms': ['neural networks', 'deep learning']}
        
        expanded_query = await iterative_search_manager._expand_query(
            current_query, original_query, context
        )
        
        assert expanded_query != current_query
        assert len(expanded_query) > len(current_query)
        # Should contain OR operators for expansion
        assert 'OR' in expanded_query or expanded_query != current_query
    
    @pytest.mark.asyncio
    async def test_query_narrowing(self, iterative_search_manager, sample_search_results):
        """Test query narrowing strategy."""
        current_query = "machine learning"
        
        narrowed_query = await iterative_search_manager._narrow_query(
            current_query, sample_search_results, None
        )
        
        assert narrowed_query != current_query
        # Should contain AND operators or exact phrase for narrowing
        assert 'AND' in narrowed_query or '"' in narrowed_query
    
    @pytest.mark.asyncio
    async def test_query_reformulation(self, iterative_search_manager, sample_search_results):
        """Test query reformulation strategy."""
        current_query = "machine learning"
        original_query = "what is machine learning"
        
        reformulated_query = await iterative_search_manager._reformulate_query(
            current_query, original_query, sample_search_results
        )
        
        assert reformulated_query != current_query
        assert len(reformulated_query) >= len(current_query)
    
    @pytest.mark.asyncio
    async def test_semantic_shift_query(self, iterative_search_manager, sample_search_results):
        """Test semantic shift strategy."""
        current_query = "machine learning"
        
        semantic_query = await iterative_search_manager._semantic_shift_query(
            current_query, sample_search_results
        )
        
        assert semantic_query != current_query
        # Should add semantic context
        assert len(semantic_query) > len(current_query)
    
    @pytest.mark.asyncio
    async def test_context_addition_query(self, iterative_search_manager, sample_search_results):
        """Test context addition strategy."""
        current_query = "machine learning"
        context = {'domain': 'artificial intelligence', 'specific_focus': 'algorithms'}
        
        contextual_query = await iterative_search_manager._add_context_to_query(
            current_query, context, sample_search_results
        )
        
        assert contextual_query != current_query
        # Should contain context terms
        assert 'artificial intelligence' in contextual_query or 'algorithms' in contextual_query
    
    def test_refinement_strategy_selection(self, iterative_search_manager):
        """Test refinement strategy selection logic."""
        # Test expansion strategy (too few results)
        strategy = iterative_search_manager._select_refinement_strategy(
            total_results=2, target_result_count=10, iteration=1, recent_results=[]
        )
        assert strategy == SearchRefinementStrategy.EXPANSION
        
        # Test narrowing strategy (too many results)
        strategy = iterative_search_manager._select_refinement_strategy(
            total_results=30, target_result_count=10, iteration=1, recent_results=[]
        )
        assert strategy == SearchRefinementStrategy.NARROWING
        
        # Test reformulation strategy (early iterations)
        strategy = iterative_search_manager._select_refinement_strategy(
            total_results=8, target_result_count=10, iteration=1, recent_results=[]
        )
        assert strategy == SearchRefinementStrategy.REFORMULATION
    
    def test_key_term_extraction(self, iterative_search_manager, sample_search_results):
        """Test key term extraction from results."""
        key_terms = iterative_search_manager._extract_key_terms_from_results(sample_search_results)
        
        assert isinstance(key_terms, list)
        assert len(key_terms) > 0
        # Should extract meaningful terms
        assert any(len(term) > 3 for term in key_terms)
    
    def test_content_analysis(self, iterative_search_manager, sample_search_results):
        """Test result content analysis."""
        analysis = iterative_search_manager._analyze_result_content(sample_search_results)
        
        assert 'dominant_topics' in analysis
        assert 'content_types' in analysis
        assert 'avg_content_length' in analysis
        assert 'common_phrases' in analysis
        
        assert isinstance(analysis['dominant_topics'], list)
        assert analysis['avg_content_length'] > 0
    
    def test_query_similarity_check(self, iterative_search_manager):
        """Test query similarity detection."""
        # Test similar queries
        similar_refinements = [
            {'refined_query': 'machine learning algorithms'},
            {'refined_query': 'machine learning algorithm'}
        ]
        
        is_similar = iterative_search_manager._queries_too_similar(similar_refinements)
        assert is_similar == True
        
        # Test different queries
        different_refinements = [
            {'refined_query': 'machine learning algorithms'},
            {'refined_query': 'deep neural networks'}
        ]
        
        is_similar = iterative_search_manager._queries_too_similar(different_refinements)
        assert is_similar == False
    
    def test_refinement_reason_generation(self, iterative_search_manager):
        """Test refinement reason generation."""
        reason = iterative_search_manager._get_refinement_reason(
            SearchRefinementStrategy.EXPANSION, 5, 10
        )
        
        assert isinstance(reason, str)
        assert len(reason) > 0
        assert 'Expanding' in reason
        assert '5' in reason and '10' in reason
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, iterative_search_manager, sample_search_results):
        """Test caching of search results."""
        query = "machine learning"
        
        # Mock cache hit
        cached_result = {
            'original_query': query,
            'results': sample_search_results,
            'success': True
        }
        
        with patch.object(iterative_search_manager.cache, 'get', return_value=cached_result):
            result = await iterative_search_manager.iterative_search(query)
            
            assert result == cached_result
            assert iterative_search_manager.search_stats['cache_hits'] > 0
    
    @pytest.mark.asyncio
    async def test_target_result_count_handling(self, iterative_search_manager, sample_search_results):
        """Test handling of different target result counts."""
        query = "machine learning"
        target_count = 2
        
        # Mock search to return more results than target
        iterative_search_manager.retriever.search.return_value = {
            'results': sample_search_results,
            'total': len(sample_search_results)
        }
        
        result = await iterative_search_manager.iterative_search(
            initial_query=query,
            target_result_count=target_count
        )
        
        assert result['success'] == True
        # Should stop when target is reached or exceeded
        assert result['total_results'] >= target_count or result['iterations_performed'] > 0
    
    @pytest.mark.asyncio
    async def test_max_iterations_limit(self, iterative_search_manager, sample_search_results):
        """Test that search respects maximum iterations limit."""
        query = "machine learning"
        
        # Mock search to always return few results to force iterations
        iterative_search_manager.retriever.search.return_value = {
            'results': sample_search_results[:1],  # Only one result per iteration
            'total': 1
        }
        
        result = await iterative_search_manager.iterative_search(
            initial_query=query,
            target_result_count=20  # High target to force max iterations
        )
        
        assert result['success'] == True
        assert result['iterations_performed'] <= iterative_search_manager.max_iterations
    
    @pytest.mark.asyncio
    async def test_query_performance_analysis(self, iterative_search_manager, sample_search_results):
        """Test query performance analysis."""
        test_queries = ["machine learning", "deep learning", "neural networks"]
        
        # Mock successful searches
        iterative_search_manager.retriever.search.return_value = {
            'results': sample_search_results,
            'total': len(sample_search_results)
        }
        
        analysis = await iterative_search_manager.analyze_query_performance(test_queries)
        
        assert 'total_queries' in analysis
        assert 'successful_searches' in analysis
        assert 'avg_iterations' in analysis
        assert 'avg_search_time' in analysis
        assert 'refinement_strategy_usage' in analysis
        assert 'query_performance' in analysis
        
        assert analysis['total_queries'] == len(test_queries)
        assert len(analysis['query_performance']) == len(test_queries)
    
    @pytest.mark.asyncio
    async def test_parameter_optimization(self, iterative_search_manager, sample_search_results):
        """Test search parameter optimization."""
        test_queries = ["machine learning", "deep learning"]
        
        # Mock search results
        iterative_search_manager.retriever.search.return_value = {
            'results': sample_search_results,
            'total': len(sample_search_results)
        }
        
        optimization = await iterative_search_manager.optimize_search_parameters(test_queries)
        
        assert 'original_parameters' in optimization
        assert 'tested_configurations' in optimization
        assert 'best_configuration' in optimization
        assert 'best_score' in optimization
        
        assert len(optimization['tested_configurations']) > 0
        assert optimization['best_score'] >= 0.0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, iterative_search_manager):
        """Test error handling in various scenarios."""
        query = "machine learning"
        
        # Test search failure
        iterative_search_manager.retriever.search.side_effect = Exception("Search failed")
        
        result = await iterative_search_manager.iterative_search(query)
        
        assert result['success'] == False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_search_stats_collection(self, iterative_search_manager, sample_search_results):
        """Test search statistics collection."""
        query = "machine learning"
        
        # Mock successful search
        iterative_search_manager.retriever.search.return_value = {
            'results': sample_search_results,
            'total': len(sample_search_results)
        }
        
        initial_searches = iterative_search_manager.search_stats['total_searches']
        
        await iterative_search_manager.iterative_search(query)
        
        assert iterative_search_manager.search_stats['total_searches'] == initial_searches + 1
        assert iterative_search_manager.search_stats['successful_refinements'] > 0
    
    @pytest.mark.asyncio
    async def test_get_search_stats(self, iterative_search_manager):
        """Test search statistics retrieval."""
        stats = await iterative_search_manager.get_search_stats()
        
        assert 'configuration' in stats
        assert 'performance_stats' in stats
        assert 'available_strategies' in stats
        assert 'system_status' in stats
        
        # Check configuration values
        config = stats['configuration']
        assert config['max_iterations'] == 3
        assert config['min_results_threshold'] == 2
        assert config['max_results_threshold'] == 15
        assert config['similarity_threshold'] == 0.8
    
    def test_cache_key_generation(self, iterative_search_manager):
        """Test cache key generation."""
        query = "machine learning"
        document_ids = ["doc1", "doc2"]
        target_count = 10
        
        cache_key = iterative_search_manager._generate_cache_key(query, document_ids, target_count)
        
        assert cache_key.startswith("iterative_search:")
        assert isinstance(cache_key, str)
        
        # Same inputs should generate same key
        cache_key2 = iterative_search_manager._generate_cache_key(query, document_ids, target_count)
        assert cache_key == cache_key2
        
        # Different inputs should generate different keys
        cache_key3 = iterative_search_manager._generate_cache_key(query, ["doc3"], target_count)
        assert cache_key != cache_key3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])