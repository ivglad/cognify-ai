"""
Tests for cross-encoder based reranking system.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import numpy as np

from app.services.search.cross_encoder_reranker import CrossEncoderReranker


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "content": "Python is a high-level programming language used for web development, data science, and machine learning.",
            "score": 0.8,
            "metadata": {"chunk_type": "text", "page_number": 1}
        },
        {
            "chunk_id": "chunk_2", 
            "document_id": "doc_1",
            "content": "JavaScript is a dynamic programming language primarily used for frontend web development and user interfaces.",
            "score": 0.7,
            "metadata": {"chunk_type": "text", "page_number": 2}
        },
        {
            "chunk_id": "chunk_3",
            "document_id": "doc_2", 
            "content": "Machine learning algorithms like neural networks can be implemented using Python libraries such as TensorFlow and PyTorch.",
            "score": 0.6,
            "metadata": {"chunk_type": "text", "page_number": 1}
        }
    ]


@pytest.fixture
def mock_cross_encoder_model():
    """Mock cross-encoder model for testing."""
    mock_model = MagicMock()
    mock_model.predict = Mock(return_value=np.array([0.8, 0.6, 0.9]))
    return mock_model


@pytest.fixture
def cross_encoder_reranker():
    """Create cross-encoder reranker instance for testing."""
    with patch('app.services.search.cross_encoder_reranker.cache_manager') as mock_cache:
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        mock_cache.keys = AsyncMock(return_value=[])
        mock_cache.delete = AsyncMock()
        
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
            batch_size=8,
            device="cpu",
            enable_caching=True
        )
        return reranker


class TestCrossEncoderReranker:
    """Test cases for cross-encoder based reranking."""
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, cross_encoder_reranker, mock_cross_encoder_model):
        """Test successful initialization of cross-encoder model."""
        with patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder_model):
            await cross_encoder_reranker.initialize()
            
            assert cross_encoder_reranker._initialized == True
            assert cross_encoder_reranker.model is not None
    
    @pytest.mark.asyncio
    async def test_initialization_missing_dependency(self, cross_encoder_reranker):
        """Test initialization when sentence-transformers is not available."""
        with patch('sentence_transformers.CrossEncoder', side_effect=ImportError("No module named 'sentence_transformers'")):
            await cross_encoder_reranker.initialize()
            
            assert cross_encoder_reranker._initialized == False
            assert cross_encoder_reranker.model is None
    
    @pytest.mark.asyncio
    async def test_rerank_results_basic(self, cross_encoder_reranker, sample_search_results, mock_cross_encoder_model):
        """Test basic cross-encoder reranking functionality."""
        query = "Python machine learning"
        
        # Mock initialization
        cross_encoder_reranker.model = mock_cross_encoder_model
        cross_encoder_reranker._initialized = True
        
        results = await cross_encoder_reranker.rerank_results(
            query=query,
            search_results=sample_search_results
        )
        
        # Check that results are returned with cross-encoder scores
        assert len(results) > 0
        assert all('cross_encoder_score' in result for result in results)
        assert all('combined_score' in result for result in results)
        assert all('rerank_method' in result for result in results)
        
        # Check that results are sorted by combined score
        scores = [result['combined_score'] for result in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_rerank_empty_results(self, cross_encoder_reranker):
        """Test reranking with empty results."""
        results = await cross_encoder_reranker.rerank_results(
            query="test query",
            search_results=[]
        )
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_rerank_without_initialization(self, cross_encoder_reranker, sample_search_results):
        """Test reranking when model is not initialized."""
        cross_encoder_reranker._initialized = False
        
        with patch.object(cross_encoder_reranker, 'initialize', side_effect=Exception("Init failed")):
            results = await cross_encoder_reranker.rerank_results(
                query="Python programming",
                search_results=sample_search_results
            )
            
            # Should return original results when initialization fails
            assert len(results) == len(sample_search_results)
            assert results == sample_search_results
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, cross_encoder_reranker, sample_search_results, mock_cross_encoder_model):
        """Test batch processing of query-document pairs."""
        query = "Python programming"
        
        # Mock model with batch processing
        mock_cross_encoder_model.predict = Mock(return_value=np.array([0.9, 0.7, 0.8]))
        cross_encoder_reranker.model = mock_cross_encoder_model
        cross_encoder_reranker._initialized = True
        cross_encoder_reranker.batch_size = 2  # Small batch size for testing
        
        results = await cross_encoder_reranker.rerank_results(
            query=query,
            search_results=sample_search_results
        )
        
        # Model predict should be called (potentially multiple times for batches)
        assert mock_cross_encoder_model.predict.called
        assert len(results) == len(sample_search_results)
    
    @pytest.mark.asyncio
    async def test_score_combination(self, cross_encoder_reranker, sample_search_results, mock_cross_encoder_model):
        """Test that original and cross-encoder scores are combined correctly."""
        query = "Python programming"
        
        mock_cross_encoder_model.predict = Mock(return_value=np.array([0.9, 0.8, 0.7]))
        cross_encoder_reranker.model = mock_cross_encoder_model
        cross_encoder_reranker._initialized = True
        
        results = await cross_encoder_reranker.rerank_results(
            query=query,
            search_results=sample_search_results
        )
        
        for result in results:
            # Check that both scores are present
            assert 'original_score' in result
            assert 'cross_encoder_score' in result
            assert 'combined_score' in result
            
            # Combined score should be weighted combination (70% cross-encoder, 30% original)
            original = result['original_score']
            cross_encoder = result['cross_encoder_score']
            combined = result['combined_score']
            
            expected_combined = 0.7 * cross_encoder + 0.3 * original
            assert abs(combined - expected_combined) < 0.01
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, cross_encoder_reranker, sample_search_results, mock_cross_encoder_model):
        """Test that caching works correctly."""
        query = "Python programming"
        
        cross_encoder_reranker.model = mock_cross_encoder_model
        cross_encoder_reranker._initialized = True
        
        # First call - should miss cache
        results1 = await cross_encoder_reranker.rerank_results(
            query=query,
            search_results=sample_search_results
        )
        
        # Mock cache hit for second call
        with patch.object(cross_encoder_reranker.cache, 'get', return_value=results1):
            with patch.object(cross_encoder_reranker, '_get_cross_encoder_scores') as mock_scoring:
                results2 = await cross_encoder_reranker.rerank_results(
                    query=query,
                    search_results=sample_search_results
                )
                
                # Scoring should not be called due to cache hit
                assert not mock_scoring.called
                assert len(results2) == len(results1)
    
    @pytest.mark.asyncio
    async def test_top_k_limiting(self, cross_encoder_reranker, sample_search_results, mock_cross_encoder_model):
        """Test that top_k parameter limits results correctly."""
        query = "Python programming"
        top_k = 2
        
        cross_encoder_reranker.model = mock_cross_encoder_model
        cross_encoder_reranker._initialized = True
        
        results = await cross_encoder_reranker.rerank_results(
            query=query,
            search_results=sample_search_results,
            top_k=top_k
        )
        
        assert len(results) == top_k
        
        # Results should be the top scoring ones
        scores = [result['combined_score'] for result in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_model_comparison(self, cross_encoder_reranker):
        """Test model comparison functionality."""
        test_queries = ["Python programming", "JavaScript development"]
        test_results = [
            [{"chunk_id": "1", "content": "Python content", "score": 0.8}],
            [{"chunk_id": "2", "content": "JavaScript content", "score": 0.7}]
        ]
        ground_truth = [["1"], ["2"]]
        
        models_to_test = ["cross-encoder/ms-marco-MiniLM-L-6-v2"]
        
        # Mock the temporary reranker creation and testing
        with patch.object(CrossEncoderReranker, '__init__', return_value=None):
            with patch.object(CrossEncoderReranker, 'initialize', return_value=None):
                with patch.object(CrossEncoderReranker, '_initialized', True):
                    with patch.object(CrossEncoderReranker, 'rerank_results', return_value=[{"chunk_id": "1", "score": 0.9}]):
                        comparison = await cross_encoder_reranker.compare_models(
                            test_queries=test_queries,
                            test_results=test_results,
                            ground_truth=ground_truth,
                            models_to_test=models_to_test
                        )
                        
                        assert 'comparison_results' in comparison
                        assert 'best_model' in comparison
    
    @pytest.mark.asyncio
    async def test_model_switching(self, cross_encoder_reranker, mock_cross_encoder_model):
        """Test switching between different cross-encoder models."""
        new_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        
        # Mock successful model loading
        with patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder_model):
            result = await cross_encoder_reranker.switch_model(new_model_name)
            
            assert result['success'] == True
            assert result['new_model'] == new_model_name
            assert cross_encoder_reranker.model_name == new_model_name
    
    @pytest.mark.asyncio
    async def test_model_switching_invalid_model(self, cross_encoder_reranker):
        """Test switching to an invalid model name."""
        invalid_model_name = "invalid-model-name"
        
        result = await cross_encoder_reranker.switch_model(invalid_model_name)
        
        assert result['success'] == False
        assert 'error' in result
        assert 'not available' in result['error']
    
    @pytest.mark.asyncio
    async def test_get_model_info(self, cross_encoder_reranker):
        """Test getting model information."""
        info = await cross_encoder_reranker.get_model_info()
        
        assert 'current_model' in info
        assert 'initialized' in info
        assert 'available_models' in info
        assert 'configuration' in info
        assert 'performance_stats' in info
        
        # Check configuration values
        config = info['configuration']
        assert config['max_length'] == 512
        assert config['batch_size'] == 8
        assert config['device'] == "cpu"
        assert config['enable_caching'] == True
    
    @pytest.mark.asyncio
    async def test_content_truncation(self, cross_encoder_reranker, mock_cross_encoder_model):
        """Test that long content is properly truncated."""
        query = "Python programming"
        
        # Create result with very long content
        long_content_result = {
            "chunk_id": "chunk_long",
            "document_id": "doc_1",
            "content": "A" * 2000,  # Very long content
            "score": 0.8,
            "metadata": {"chunk_type": "text"}
        }
        
        cross_encoder_reranker.model = mock_cross_encoder_model
        cross_encoder_reranker._initialized = True
        
        results = await cross_encoder_reranker.rerank_results(
            query=query,
            search_results=[long_content_result]
        )
        
        # Should still process the result
        assert len(results) == 1
        assert 'cross_encoder_score' in results[0]
    
    @pytest.mark.asyncio
    async def test_error_handling_in_scoring(self, cross_encoder_reranker, sample_search_results):
        """Test error handling when model scoring fails."""
        query = "Python programming"
        
        # Mock model that raises exception
        mock_failing_model = MagicMock()
        mock_failing_model.predict = Mock(side_effect=Exception("Model prediction failed"))
        
        cross_encoder_reranker.model = mock_failing_model
        cross_encoder_reranker._initialized = True
        
        results = await cross_encoder_reranker.rerank_results(
            query=query,
            search_results=sample_search_results
        )
        
        # Should return original results when scoring fails
        assert len(results) == len(sample_search_results)
        # Results should have default scores
        for result in results:
            if 'cross_encoder_score' in result:
                assert result['cross_encoder_score'] == 0.5  # Default fallback score
    
    def test_ndcg_calculation(self, cross_encoder_reranker):
        """Test NDCG calculation."""
        results = [
            {"chunk_id": "1", "score": 0.9},
            {"chunk_id": "2", "score": 0.8},
            {"chunk_id": "3", "score": 0.7}
        ]
        ground_truth = ["1", "3"]  # First and third results are relevant
        
        ndcg = cross_encoder_reranker._calculate_ndcg(results, ground_truth)
        
        # NDCG should be > 0 since we have relevant results
        assert ndcg > 0.0
        assert ndcg <= 1.0
    
    def test_cache_key_generation(self, cross_encoder_reranker, sample_search_results):
        """Test cache key generation."""
        query = "Python programming"
        
        cache_key = cross_encoder_reranker._generate_cache_key(query, sample_search_results)
        
        assert cache_key.startswith("cross_encoder_rerank:")
        assert isinstance(cache_key, str)
        
        # Same inputs should generate same key
        cache_key2 = cross_encoder_reranker._generate_cache_key(query, sample_search_results)
        assert cache_key == cache_key2
    
    def test_stats_update(self, cross_encoder_reranker):
        """Test performance statistics update."""
        initial_count = cross_encoder_reranker.rerank_stats['total_reranks']
        
        cross_encoder_reranker._update_rerank_stats(0.5)
        
        assert cross_encoder_reranker.rerank_stats['total_reranks'] == initial_count + 1
        assert cross_encoder_reranker.rerank_stats['avg_rerank_time'] >= 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])