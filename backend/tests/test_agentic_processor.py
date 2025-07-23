"""
Tests for agentic response generation system.
"""
import pytest
import trio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import json

from app.services.response.agentic_processor import AgenticProcessor, ReasoningStep


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "content": "Python is a high-level programming language that is widely used for web development, data science, and artificial intelligence applications.",
            "score": 0.9,
            "metadata": {"chunk_type": "text", "page_number": 1}
        },
        {
            "chunk_id": "chunk_2",
            "document_id": "doc_2",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            "score": 0.8,
            "metadata": {"chunk_type": "text", "page_number": 2}
        },
        {
            "chunk_id": "chunk_3",
            "document_id": "doc_3",
            "content": "Deep learning uses neural networks with multiple layers to process complex patterns in data, making it particularly effective for image and speech recognition.",
            "score": 0.7,
            "metadata": {"chunk_type": "text", "page_number": 1}
        }
    ]


@pytest.fixture
def agentic_processor():
    """Create agentic processor instance for testing."""
    with patch('app.services.response.agentic_processor.basic_retriever') as mock_retriever:
        mock_retriever._initialized = True
        mock_retriever.initialize = AsyncMock()
        mock_retriever.search = AsyncMock()
        
        with patch('app.services.response.agentic_processor.cache_manager') as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            
            processor = AgenticProcessor(
                max_reasoning_steps=3,
                max_search_iterations=2,
                confidence_threshold=0.7,
                enable_caching=True
            )
            return processor


class TestAgenticProcessor:
    """Test cases for agentic response generation."""
    
    @pytest.mark.trio
    async def test_process_query_basic(self, agentic_processor, sample_search_results):
        """Test basic query processing functionality."""
        query = "What is machine learning?"
        
        # Mock LLM responses
        analysis_response = json.dumps({
            "query_type": "factual",
            "key_components": ["machine", "learning"],
            "search_queries": ["machine learning definition", "what is machine learning"],
            "complexity_level": "simple",
            "reasoning_approach": "direct"
        })
        
        reasoning_response = json.dumps({
            "step_type": "analysis",
            "reasoning": "Analyzing the query about machine learning definition",
            "search_needed": True,
            "search_queries": ["machine learning"],
            "confidence": 0.8,
            "next_action": "conclude"
        })
        
        synthesis_response = json.dumps({
            "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "confidence": 0.9,
            "supporting_evidence": ["ML is part of AI", "Learns from data"],
            "limitations": [],
            "additional_context": "Widely used in various applications"
        })
        
        validation_response = json.dumps({
            "accuracy_score": 0.9,
            "completeness_score": 0.8,
            "relevance_score": 0.9,
            "overall_quality": 0.87,
            "issues_found": [],
            "improvement_suggestions": [],
            "validation_passed": True
        })
        
        # Mock LLM calls
        with patch.object(agentic_processor, '_call_llm', side_effect=[
            analysis_response, reasoning_response, synthesis_response, validation_response
        ]):
            # Mock search results
            agentic_processor.retriever.search.return_value = {
                'results': sample_search_results,
                'total': len(sample_search_results)
            }
            
            result = await agentic_processor.process_query(query)
            
            assert result['success'] == True
            assert result['query'] == query
            assert 'answer' in result
            assert 'confidence' in result
            assert 'reasoning_steps' in result
            assert len(result['reasoning_steps']) > 0
            assert result['validation']['validation_passed'] == True
    
    @pytest.mark.trio
    async def test_process_empty_query(self, agentic_processor):
        """Test processing empty query."""
        result = await agentic_processor.process_query("")
        
        assert result['success'] == False
        assert 'Empty query' in result.get('error', '')
    
    @pytest.mark.trio
    async def test_query_analysis_success(self, agentic_processor):
        """Test successful query analysis."""
        query = "How does deep learning work?"
        
        analysis_response = json.dumps({
            "query_type": "procedural",
            "key_components": ["deep", "learning", "work"],
            "search_queries": ["deep learning process", "how deep learning works"],
            "complexity_level": "moderate",
            "reasoning_approach": "multi_step"
        })
        
        with patch.object(agentic_processor, '_call_llm', return_value=analysis_response):
            result = await agentic_processor._analyze_query(query)
            
            assert result['success'] == True
            assert result['analysis']['query_type'] == 'procedural'
            assert result['analysis']['complexity_level'] == 'moderate'
    
    @pytest.mark.trio
    async def test_query_analysis_fallback(self, agentic_processor):
        """Test query analysis fallback when LLM fails."""
        query = "What is artificial intelligence?"
        
        # Mock LLM failure
        with patch.object(agentic_processor, '_call_llm', return_value=None):
            result = await agentic_processor._analyze_query(query)
            
            assert result['success'] == True
            assert 'analysis' in result
            # Should use heuristic analysis
            assert result['analysis']['query_type'] == 'factual'
    
    @pytest.mark.trio
    async def test_heuristic_query_analysis(self, agentic_processor):
        """Test heuristic query analysis."""
        # Test different query types
        test_cases = [
            ("What is machine learning?", "factual"),
            ("How to implement neural networks?", "procedural"),
            ("Compare supervised vs unsupervised learning", "comparative"),
            ("Why is deep learning effective?", "analytical")
        ]
        
        for query, expected_type in test_cases:
            analysis = agentic_processor._heuristic_query_analysis(query)
            
            assert analysis['query_type'] == expected_type
            assert 'key_components' in analysis
            assert 'search_queries' in analysis
            assert 'complexity_level' in analysis
            assert 'reasoning_approach' in analysis
    
    @pytest.mark.trio
    async def test_reasoning_step_execution(self, agentic_processor):
        """Test execution of individual reasoning steps."""
        query = "What is machine learning?"
        analysis = {
            "query_type": "factual",
            "key_components": ["machine", "learning"],
            "search_queries": ["machine learning"],
            "complexity_level": "simple",
            "reasoning_approach": "direct"
        }
        
        step_response = json.dumps({
            "step_type": "analysis",
            "reasoning": "Analyzing machine learning concepts",
            "search_needed": True,
            "search_queries": ["machine learning definition"],
            "confidence": 0.8,
            "next_action": "continue"
        })
        
        with patch.object(agentic_processor, '_call_llm', return_value=step_response):
            result = await agentic_processor._execute_reasoning_step(
                query, analysis, [], [], 1, 3, None, False
            )
            
            assert result['success'] == True
            assert result['step']['step_type'] == 'analysis'
            assert result['step']['confidence'] == 0.8
            assert result['step']['step_number'] == 1
    
    @pytest.mark.asyncio
    async def test_iterative_search(self, agentic_processor, sample_search_results):
        """Test iterative search functionality."""
        search_queries = ["machine learning", "artificial intelligence"]
        
        # Mock search results
        agentic_processor.retriever.search.return_value = {
            'results': sample_search_results,
            'total': len(sample_search_results)
        }
        
        result = await agentic_processor._perform_iterative_search(
            search_queries, None, False
        )
        
        assert 'results' in result
        assert 'iterations' in result
        assert result['iterations'] > 0
        assert len(result['results']) > 0
        
        # Check that search was called
        assert agentic_processor.retriever.search.called
    
    @pytest.mark.asyncio
    async def test_final_response_generation(self, agentic_processor, sample_search_results):
        """Test final response generation."""
        query = "What is machine learning?"
        analysis = {"query_type": "factual"}
        reasoning_steps = [
            {
                "step_type": "analysis",
                "reasoning": "Analyzing ML concepts",
                "confidence": 0.8
            }
        ]
        
        synthesis_response = json.dumps({
            "answer": "Machine learning is a method of data analysis that automates analytical model building.",
            "confidence": 0.85,
            "supporting_evidence": ["Automates model building", "Uses data analysis"],
            "limitations": ["Requires quality data"],
            "additional_context": "Part of artificial intelligence field"
        })
        
        with patch.object(agentic_processor, '_call_llm', return_value=synthesis_response):
            result = await agentic_processor._generate_final_response(
                query, analysis, reasoning_steps, sample_search_results
            )
            
            assert 'answer' in result
            assert 'confidence' in result
            assert result['confidence'] == 0.85
            assert 'supporting_evidence' in result
    
    @pytest.mark.asyncio
    async def test_response_validation(self, agentic_processor):
        """Test response quality validation."""
        query = "What is machine learning?"
        response = {
            "answer": "Machine learning is a subset of AI",
            "confidence": 0.8,
            "supporting_evidence": ["Part of AI field"]
        }
        reasoning_steps = [{"step_type": "analysis", "reasoning": "Analyzed ML"}]
        
        validation_response = json.dumps({
            "accuracy_score": 0.9,
            "completeness_score": 0.7,
            "relevance_score": 0.9,
            "overall_quality": 0.83,
            "issues_found": ["Could be more detailed"],
            "improvement_suggestions": ["Add more examples"],
            "validation_passed": True
        })
        
        with patch.object(agentic_processor, '_call_llm', return_value=validation_response):
            result = await agentic_processor._validate_response_quality(
                query, response, reasoning_steps
            )
            
            assert 'accuracy_score' in result
            assert 'overall_quality' in result
            assert result['validation_passed'] == True
    
    @pytest.mark.asyncio
    async def test_fallback_response_creation(self, agentic_processor, sample_search_results):
        """Test fallback response creation."""
        query = "What is machine learning?"
        
        # Test with search results
        result = agentic_processor._create_fallback_response(query, sample_search_results)
        
        assert 'answer' in result
        assert 'confidence' in result
        assert result['confidence'] > 0.5  # Should have reasonable confidence with results
        assert 'supporting_evidence' in result
        assert 'limitations' in result
        
        # Test without search results
        result_empty = agentic_processor._create_fallback_response(query, [])
        
        assert result_empty['confidence'] < 0.5  # Lower confidence without results
        assert "don't have enough information" in result_empty['answer']
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, agentic_processor):
        """Test caching of agentic processing results."""
        query = "What is machine learning?"
        
        # Mock cached result
        cached_result = {
            'query': query,
            'answer': 'Cached answer',
            'confidence': 0.8,
            'success': True
        }
        
        with patch.object(agentic_processor.cache, 'get', return_value=cached_result):
            result = await agentic_processor.process_query(query)
            
            assert result == cached_result
            assert agentic_processor.processing_stats['cache_hits'] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agentic_processor):
        """Test error handling in various scenarios."""
        query = "What is machine learning?"
        
        # Test LLM failure
        with patch.object(agentic_processor, '_call_llm', return_value=None):
            result = await agentic_processor.process_query(query)
            
            # Should still complete with fallback logic
            assert 'answer' in result
            assert result['confidence'] >= 0.0
    
    @pytest.mark.asyncio
    async def test_complex_query_processing(self, agentic_processor, sample_search_results):
        """Test processing of complex queries requiring multiple steps."""
        query = "Compare supervised and unsupervised machine learning approaches and explain when to use each"
        
        # Mock complex analysis
        analysis_response = json.dumps({
            "query_type": "comparative",
            "key_components": ["supervised", "unsupervised", "machine learning"],
            "search_queries": ["supervised learning", "unsupervised learning", "comparison"],
            "complexity_level": "complex",
            "reasoning_approach": "comparative"
        })
        
        # Mock multiple reasoning steps
        step_responses = [
            json.dumps({
                "step_type": "analysis",
                "reasoning": "Analyzing supervised learning",
                "search_needed": True,
                "search_queries": ["supervised learning"],
                "confidence": 0.6,
                "next_action": "continue"
            }),
            json.dumps({
                "step_type": "analysis",
                "reasoning": "Analyzing unsupervised learning",
                "search_needed": True,
                "search_queries": ["unsupervised learning"],
                "confidence": 0.7,
                "next_action": "continue"
            }),
            json.dumps({
                "step_type": "synthesis",
                "reasoning": "Comparing both approaches",
                "search_needed": False,
                "confidence": 0.8,
                "next_action": "conclude"
            })
        ]
        
        synthesis_response = json.dumps({
            "answer": "Supervised learning uses labeled data while unsupervised learning finds patterns in unlabeled data.",
            "confidence": 0.85,
            "supporting_evidence": ["Labeled vs unlabeled data", "Different use cases"],
            "limitations": ["Depends on data availability"],
            "additional_context": "Choice depends on problem type and data availability"
        })
        
        validation_response = json.dumps({
            "accuracy_score": 0.9,
            "completeness_score": 0.8,
            "relevance_score": 0.9,
            "overall_quality": 0.87,
            "issues_found": [],
            "improvement_suggestions": [],
            "validation_passed": True
        })
        
        # Mock all LLM calls
        llm_responses = [analysis_response] + step_responses + [synthesis_response, validation_response]
        
        with patch.object(agentic_processor, '_call_llm', side_effect=llm_responses):
            # Mock search results
            agentic_processor.retriever.search.return_value = {
                'results': sample_search_results,
                'total': len(sample_search_results)
            }
            
            result = await agentic_processor.process_query(query)
            
            assert result['success'] == True
            assert len(result['reasoning_steps']) >= 2  # Should have multiple steps for complex query
            assert result['query_analysis']['complexity_level'] == 'complex'
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_handling(self, agentic_processor):
        """Test handling of confidence thresholds."""
        query = "What is machine learning?"
        
        # Set high confidence threshold
        agentic_processor.confidence_threshold = 0.9
        
        # Mock low confidence step
        step_response = json.dumps({
            "step_type": "analysis",
            "reasoning": "Initial analysis",
            "search_needed": True,
            "search_queries": ["machine learning"],
            "confidence": 0.5,  # Below threshold
            "next_action": "continue"
        })
        
        # Mock high confidence step
        high_confidence_step = json.dumps({
            "step_type": "synthesis",
            "reasoning": "Comprehensive analysis complete",
            "search_needed": False,
            "confidence": 0.95,  # Above threshold
            "next_action": "conclude"
        })
        
        with patch.object(agentic_processor, '_call_llm', side_effect=[step_response, high_confidence_step]):
            with patch.object(agentic_processor, '_perform_iterative_search', return_value={'results': [], 'iterations': 1}):
                result = await agentic_processor._execute_reasoning_process(
                    query, {"complexity_level": "moderate"}, None, None, False
                )
                
                assert result['success'] == True
                # Should continue until high confidence is reached
                assert len(result['reasoning_steps']) >= 1
    
    @pytest.mark.asyncio
    async def test_processor_stats(self, agentic_processor):
        """Test processor statistics collection."""
        stats = await agentic_processor.get_processor_stats()
        
        assert 'configuration' in stats
        assert 'performance_stats' in stats
        assert 'available_reasoning_types' in stats
        assert 'system_status' in stats
        
        # Check configuration values
        config = stats['configuration']
        assert config['max_reasoning_steps'] == 3
        assert config['max_search_iterations'] == 2
        assert config['confidence_threshold'] == 0.7
        assert config['enable_caching'] == True
    
    def test_cache_key_generation(self, agentic_processor):
        """Test cache key generation."""
        query = "What is machine learning?"
        document_ids = ["doc1", "doc2"]
        
        cache_key = agentic_processor._generate_cache_key(query, document_ids)
        
        assert cache_key.startswith("agentic_process:")
        assert isinstance(cache_key, str)
        
        # Same inputs should generate same key
        cache_key2 = agentic_processor._generate_cache_key(query, document_ids)
        assert cache_key == cache_key2
        
        # Different inputs should generate different keys
        cache_key3 = agentic_processor._generate_cache_key(query, ["doc3"])
        assert cache_key != cache_key3
    
    def test_stats_update(self, agentic_processor):
        """Test performance statistics update."""
        initial_count = agentic_processor.processing_stats['total_queries']
        
        result = {
            'processing_time': 2.5,
            'reasoning_steps': [{'step': 1}, {'step': 2}],
            'search_iterations': 3,
            'success': True,
            'validation': {'validation_passed': True}
        }
        
        agentic_processor._update_processing_stats(result)
        
        assert agentic_processor.processing_stats['total_queries'] == initial_count + 1
        assert agentic_processor.processing_stats['successful_conclusions'] > 0
        assert agentic_processor.processing_stats['avg_processing_time'] > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])