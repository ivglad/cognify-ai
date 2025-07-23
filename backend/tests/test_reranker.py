"""
Tests for LLM-based reranking system.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from app.services.search.reranker import LLMReranker


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "content": "Python is a programming language used for web development and data science.",
            "score": 0.8,
            "metadata": {"chunk_type": "text", "page_number": 1}
        },
        {
            "chunk_id": "chunk_2", 
            "document_id": "doc_1",
            "content": "JavaScript is primarily used for frontend web development.",
            "score": 0.7,
            "metadata": {"chunk_type": "text", "page_number": 2}
        },
        {
            "chunk_id": "chunk_3",
            "document_id": "doc_2", 
            "content": "Machine learning algorithms can be implemented in Python using libraries like scikit-learn.",
            "score": 0.6,
            "metadata": {"chunk_type": "text", "page_number": 1}
        }
    ]


@pytest.fixture
def reranker():
    """Create reranker instance for testing."""
    with patch('app.services.search.reranker.rag_tokenizer') as mock_tokenizer:
        mock_tokenizer._initialized = True
        mock_tokenizer.tokenize = Mock(return_value="python programming language")
        
        with patch('app.services.search.reranker.cache_manager') as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            
            reranker = LLMReranker(
                max_rerank_candidates=10,
                batch_size=5,
                relevance_threshold=0.5,
                enable_caching=True
            )
            return reranker


class TestLLMReranker:
    """Test cases for LLM-based reranking."""
    
    @pytest.mark.asyncio
    async def test_rerank_results_basic(self, reranker, sample_search_results):
        """Test basic reranking functionality."""
        query = "Python programming"
        
        # Mock LLM scoring to return predictable scores
        with patch.object(reranker, '_call_llm_for_scoring', return_value=[0.9, 0.3, 0.8]):
            results = await reranker.rerank_results(
                query=query,
                search_results=sample_search_results,
                rerank_type="general"
            )
        
        # Check that results are returned
        assert len(results) > 0
        assert all('llm_relevance_score' in result for result in results)
        assert all('combined_score' in result for result in results)
        
        # Check that results are sorted by combined score
        scores = [result['combined_score'] for result in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_rerank_empty_results(self, reranker):
        """Test reranking with empty results."""
        results = await reranker.rerank_results(
            query="test query",
            search_results=[],
            rerank_type="general"
        )
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_rerank_single_result(self, reranker, sample_search_results):
        """Test reranking with single result."""
        single_result = [sample_search_results[0]]
        
        with patch.object(reranker, '_call_llm_for_scoring', return_value=[0.8]):
            results = await reranker.rerank_results(
                query="Python programming",
                search_results=single_result,
                rerank_type="general"
            )
        
        assert len(results) == 1
        assert 'llm_relevance_score' in results[0]
    
    @pytest.mark.asyncio
    async def test_batch_reranking(self, reranker, sample_search_results):
        """Test batch reranking functionality."""
        queries_and_results = [
            ("Python programming", sample_search_results),
            ("JavaScript development", sample_search_results[:2])
        ]
        
        with patch.object(reranker, 'rerank_results', side_effect=[
            sample_search_results,  # First query results
            sample_search_results[:2]  # Second query results
        ]):
            results = await reranker.batch_rerank_results(
                queries_and_results=queries_and_results,
                rerank_type="general"
            )
        
        assert len(results) == 2
        assert len(results[0]) == 3  # First query results
        assert len(results[1]) == 2  # Second query results
    
    @pytest.mark.asyncio
    async def test_relevance_threshold_filtering(self, reranker, sample_search_results):
        """Test that results below relevance threshold are filtered out."""
        reranker.relevance_threshold = 0.7
        
        # Mock LLM to return scores where some are below threshold
        with patch.object(reranker, '_call_llm_for_scoring', return_value=[0.8, 0.5, 0.9]):
            results = await reranker.rerank_results(
                query="Python programming",
                search_results=sample_search_results,
                rerank_type="general"
            )
        
        # Should only return results with LLM score >= 0.7
        assert all(result['llm_relevance_score'] >= 0.7 for result in results)
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, reranker, sample_search_results):
        """Test that caching works correctly."""
        query = "Python programming"
        
        # First call - should miss cache and call LLM
        with patch.object(reranker, '_call_llm_for_scoring', return_value=[0.8, 0.7, 0.6]) as mock_llm:
            results1 = await reranker.rerank_results(
                query=query,
                search_results=sample_search_results,
                rerank_type="general"
            )
            
            # LLM should be called
            assert mock_llm.called
        
        # Mock cache hit for second call
        cached_data = {
            'results': results1,
            'rerank_type': 'general',
            'timestamp': 1234567890,
            'query_hash': hash(query),
            'result_count': len(results1)
        }
        
        with patch.object(reranker.cache, 'get', return_value=cached_data):
            with patch.object(reranker, '_call_llm_for_scoring') as mock_llm:
                results2 = await reranker.rerank_results(
                    query=query,
                    search_results=sample_search_results,
                    rerank_type="general"
                )
                
                # LLM should not be called due to cache hit
                assert not mock_llm.called
                assert len(results2) == len(results1)
    
    @pytest.mark.asyncio
    async def test_different_rerank_types(self, reranker, sample_search_results):
        """Test different reranking types use different prompts."""
        query = "Python programming"
        
        with patch.object(reranker, '_call_llm_for_scoring', return_value=[0.8, 0.7, 0.6]) as mock_llm:
            # Test general reranking
            await reranker.rerank_results(query, sample_search_results, "general")
            general_prompt = mock_llm.call_args[0][0]
            
            # Test technical reranking
            await reranker.rerank_results(query, sample_search_results, "technical")
            technical_prompt = mock_llm.call_args[0][0]
            
            # Prompts should be different
            assert general_prompt != technical_prompt
            assert "technical" in technical_prompt.lower()
    
    @pytest.mark.asyncio
    async def test_heuristic_fallback(self, reranker, sample_search_results):
        """Test fallback to heuristic scoring when LLM fails."""
        query = "Python programming"
        
        # Mock LLM to raise exception
        with patch.object(reranker, '_call_llm_for_scoring', side_effect=Exception("LLM failed")):
            with patch.object(reranker, '_heuristic_scoring', return_value=[0.7, 0.6, 0.5]) as mock_heuristic:
                results = await reranker.rerank_results(
                    query=query,
                    search_results=sample_search_results,
                    rerank_type="general"
                )
                
                # Should fallback to heuristic scoring
                assert mock_heuristic.called
                assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_score_combination(self, reranker, sample_search_results):
        """Test that original and LLM scores are combined correctly."""
        query = "Python programming"
        
        with patch.object(reranker, '_call_llm_for_scoring', return_value=[0.9, 0.8, 0.7]):
            results = await reranker.rerank_results(
                query=query,
                search_results=sample_search_results,
                rerank_type="general"
            )
        
        for result in results:
            # Check that both scores are present
            assert 'original_score' in result
            assert 'llm_relevance_score' in result
            assert 'combined_score' in result
            
            # Combined score should be weighted combination
            original = result['original_score']
            llm_score = result['llm_relevance_score']
            combined = result['combined_score']
            
            expected_combined = 0.6 * llm_score + 0.4 * original
            assert abs(combined - expected_combined) < 0.01
    
    @pytest.mark.asyncio
    async def test_reranker_stats(self, reranker):
        """Test reranker statistics collection."""
        stats = await reranker.get_reranker_stats()
        
        assert 'configuration' in stats
        assert 'available_rerank_types' in stats
        assert 'performance_stats' in stats
        assert 'system_status' in stats
        
        # Check configuration values
        config = stats['configuration']
        assert config['max_rerank_candidates'] == 10
        assert config['batch_size'] == 5
        assert config['relevance_threshold'] == 0.5
        assert config['enable_caching'] == True
    
    @pytest.mark.asyncio
    async def test_quality_evaluation(self, reranker, sample_search_results):
        """Test reranking quality evaluation."""
        original_results = sample_search_results.copy()
        
        with patch.object(reranker, '_call_llm_for_scoring', return_value=[0.9, 0.7, 0.8]):
            reranked_results = await reranker.rerank_results(
                query="Python programming",
                search_results=sample_search_results,
                rerank_type="general"
            )
        
        # Test quality evaluation
        ground_truth = ["chunk_1", "chunk_3"]  # Relevant chunks
        
        metrics = await reranker.evaluate_reranking_quality(
            original_results=original_results,
            reranked_results=reranked_results,
            ground_truth=ground_truth
        )
        
        assert 'original_count' in metrics
        assert 'reranked_count' in metrics
        assert 'rank_changes' in metrics
        assert 'avg_llm_score' in metrics
        assert metrics['original_count'] == 3
        assert metrics['reranked_count'] == len(reranked_results)
    
    @pytest.mark.asyncio
    async def test_cache_clearing(self, reranker):
        """Test cache clearing functionality."""
        with patch.object(reranker.cache, 'keys', return_value=['llm_rerank:key1', 'llm_rerank:key2']):
            with patch.object(reranker.cache, 'delete', return_value=True) as mock_delete:
                result = await reranker.clear_reranking_cache()
                
                assert result['cleared_entries'] == 2
                assert result['cache_cleared'] == True
                assert mock_delete.call_count == 2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])