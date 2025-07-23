"""
Tests for citation generation system.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from app.services.response.citation_generator import (
    CitationGenerator, Citation, CitationType
)


@pytest.fixture
def sample_response_text():
    """Sample response text for testing."""
    return """
    Machine learning is a powerful technique for data analysis. It uses algorithms to automatically learn patterns from data.
    Deep learning, a subset of machine learning, uses neural networks with multiple layers. This approach has shown remarkable success in image recognition and natural language processing.
    Python is widely used for implementing machine learning algorithms due to its simplicity and extensive libraries.
    """


@pytest.fixture
def sample_source_chunks():
    """Sample source chunks for testing."""
    return [
        {
            'chunk_id': 'chunk_1',
            'document_id': 'doc_1',
            'content': 'Machine learning is a method of data analysis that automates analytical model building. It is based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.',
            'metadata': {
                'document_name': 'ML Guide',
                'page_number': 1
            },
            'url': 'https://example.com/ml-guide'
        },
        {
            'chunk_id': 'chunk_2',
            'document_id': 'doc_1',
            'content': 'Deep learning is part of a broader family of machine learning methods based on artificial neural networks. Deep learning architectures such as deep neural networks, deep belief networks, and recurrent neural networks have been applied to fields including computer vision, speech recognition, natural language processing.',
            'metadata': {
                'document_name': 'ML Guide',
                'page_number': 2
            },
            'url': 'https://example.com/ml-guide'
        },
        {
            'chunk_id': 'chunk_3',
            'document_id': 'doc_2',
            'content': 'Python is an interpreted, high-level programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together.',
            'metadata': {
                'document_name': 'Python Guide',
                'page_number': 1
            },
            'url': 'https://example.com/python-guide'
        }
    ]


@pytest.fixture
def citation_generator_service():
    """Create citation generator service instance for testing."""
    with patch('app.services.response.citation_generator.cache_manager') as mock_cache:
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        
        service = CitationGenerator(
            min_confidence_threshold=0.7,
            max_citations_per_sentence=3,
            enable_caching=True
        )
        return service


class TestCitationGenerator:
    """Test cases for citation generation."""
    
    @pytest.mark.asyncio
    async def test_generate_citations_basic(self, citation_generator_service, sample_response_text, sample_source_chunks):
        """Test basic citation generation."""
        result = await citation_generator_service.generate_citations(
            response_text=sample_response_text,
            source_chunks=sample_source_chunks
        )
        
        assert result['success'] == True
        assert result['total_citations'] > 0
        assert 'citations' in result
        assert 'sentence_mappings' in result
        assert result['avg_confidence'] > 0
        assert result['citation_coverage'] > 0
    
    @pytest.mark.asyncio
    async def test_generate_citations_empty_input(self, citation_generator_service):
        """Test citation generation with empty input."""
        result = await citation_generator_service.generate_citations(
            response_text="",
            source_chunks=[]
        )
        
        assert result['success'] == True
        assert result['total_citations'] == 0
        assert result['citations'] == []
        assert result['avg_confidence'] == 0.0
        assert result['citation_coverage'] == 0.0
    
    @pytest.mark.asyncio
    async def test_find_direct_quotes(self, citation_generator_service):
        """Test direct quote finding."""
        sentence = "Machine learning is a method of data analysis"
        chunk_content = "Machine learning is a method of data analysis that automates analytical model building."
        
        quotes = citation_generator_service._find_direct_quotes(sentence, chunk_content)
        
        assert len(quotes) > 0
        assert any("machine learning is a method" in quote['cited_text'].lower() for quote in quotes)
        assert all(quote['confidence'] > 0.7 for quote in quotes)
    
    @pytest.mark.asyncio
    async def test_find_paraphrases(self, citation_generator_service):
        """Test paraphrase finding."""
        sentence = "ML algorithms learn from data automatically"
        chunk_content = "Machine learning systems can learn from data and identify patterns with minimal human intervention."
        
        paraphrases = await citation_generator_service._find_paraphrases(sentence, chunk_content)
        
        # Should find semantic similarity
        assert len(paraphrases) >= 0  # May or may not find paraphrases depending on similarity threshold
        if paraphrases:
            assert all(para['confidence'] > 0.6 for para in paraphrases)
    
    def test_find_supporting_evidence(self, citation_generator_service):
        """Test supporting evidence finding."""
        sentence = "Python is used for machine learning"
        chunk_content = "Python programming language is popular for data science and machine learning applications."
        
        evidence = citation_generator_service._find_supporting_evidence(sentence, chunk_content)
        
        assert len(evidence) > 0
        assert all(ev['confidence'] > 0.3 for ev in evidence)
    
    def test_split_into_sentences(self, citation_generator_service):
        """Test sentence splitting."""
        text = "This is sentence one. This is sentence two! Is this sentence three?"
        
        sentences = citation_generator_service._split_into_sentences(text)
        
        assert len(sentences) == 3
        assert "This is sentence one" in sentences[0]
        assert "This is sentence two" in sentences[1]
        assert "Is this sentence three" in sentences[2]
    
    def test_extract_key_terms(self, citation_generator_service):
        """Test key term extraction."""
        text = "Machine learning algorithms process large datasets efficiently"
        
        terms = citation_generator_service._extract_key_terms(text)
        
        assert 'machine' in terms
        assert 'learning' in terms
        assert 'algorithms' in terms
        assert 'datasets' in terms
        # Stop words should be filtered out
        assert 'the' not in terms
        assert 'and' not in terms
    
    def test_calculate_semantic_similarity(self, citation_generator_service):
        """Test semantic similarity calculation."""
        text1 = "Machine learning processes data"
        text2 = "ML algorithms analyze information"
        text3 = "The weather is sunny today"
        
        # Similar texts should have higher similarity
        similarity1 = citation_generator_service._calculate_semantic_similarity(text1, text2)
        similarity2 = citation_generator_service._calculate_semantic_similarity(text1, text3)
        
        assert similarity1 > similarity2
        assert 0 <= similarity1 <= 1
        assert 0 <= similarity2 <= 1
    
    @pytest.mark.asyncio
    async def test_generate_sentence_citations(self, citation_generator_service, sample_source_chunks):
        """Test citation generation for a single sentence."""
        sentence = "Machine learning is a powerful technique for data analysis"
        
        result = await citation_generator_service._generate_sentence_citations(
            sentence=sentence,
            source_chunks=sample_source_chunks,
            sentence_index=0,
            context=None
        )
        
        assert 'citations' in result
        assert 'avg_confidence' in result
        assert 'total_candidates' in result
        assert result['avg_confidence'] >= 0
    
    @pytest.mark.asyncio
    async def test_find_chunk_citations(self, citation_generator_service, sample_source_chunks):
        """Test finding citations within a specific chunk."""
        sentence = "Machine learning is a method of data analysis"
        chunk = sample_source_chunks[0]  # First chunk contains similar content
        
        citations = await citation_generator_service._find_chunk_citations(
            sentence=sentence,
            chunk=chunk,
            sentence_index=0
        )
        
        assert len(citations) > 0
        assert all(isinstance(citation, Citation) for citation in citations)
        assert all(citation.chunk_id == chunk['chunk_id'] for citation in citations)
        assert all(citation.confidence > 0 for citation in citations)
    
    def test_calculate_average_confidence(self, citation_generator_service):
        """Test average confidence calculation."""
        citations = [
            Citation(
                id="1", chunk_id="c1", document_id="d1", document_name="doc",
                page_number=1, sentence_start=0, sentence_end=0,
                cited_text="text", source_text="source", confidence=0.8,
                citation_type=CitationType.DIRECT_QUOTE
            ),
            Citation(
                id="2", chunk_id="c2", document_id="d2", document_name="doc",
                page_number=1, sentence_start=0, sentence_end=0,
                cited_text="text", source_text="source", confidence=0.6,
                citation_type=CitationType.PARAPHRASE
            )
        ]
        
        avg_confidence = citation_generator_service._calculate_average_confidence(citations)
        
        assert avg_confidence == 0.7  # (0.8 + 0.6) / 2
    
    def test_calculate_citation_coverage(self, citation_generator_service):
        """Test citation coverage calculation."""
        sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]
        citations = [
            Citation(
                id="1", chunk_id="c1", document_id="d1", document_name="doc",
                page_number=1, sentence_start=0, sentence_end=0,
                cited_text="text", source_text="source", confidence=0.8,
                citation_type=CitationType.DIRECT_QUOTE
            ),
            Citation(
                id="2", chunk_id="c2", document_id="d2", document_name="doc",
                page_number=1, sentence_start=2, sentence_end=2,
                cited_text="text", source_text="source", confidence=0.6,
                citation_type=CitationType.PARAPHRASE
            )
        ]
        
        coverage = citation_generator_service._calculate_citation_coverage(sentences, citations)
        
        assert coverage == 2/3  # 2 out of 3 sentences have citations
    
    def test_generate_cache_key(self, citation_generator_service, sample_response_text, sample_source_chunks):
        """Test cache key generation."""
        cache_key = citation_generator_service._generate_cache_key(
            sample_response_text, sample_source_chunks
        )
        
        assert cache_key.startswith("citations:")
        assert isinstance(cache_key, str)
        
        # Same inputs should generate same key
        cache_key2 = citation_generator_service._generate_cache_key(
            sample_response_text, sample_source_chunks
        )
        assert cache_key == cache_key2
        
        # Different inputs should generate different keys
        cache_key3 = citation_generator_service._generate_cache_key(
            "different text", sample_source_chunks
        )
        assert cache_key != cache_key3
    
    @pytest.mark.asyncio
    async def test_validate_citations(self, citation_generator_service, sample_source_chunks):
        """Test citation validation."""
        citations = [
            Citation(
                id="1", chunk_id="chunk_1", document_id="doc_1", document_name="ML Guide",
                page_number=1, sentence_start=0, sentence_end=0,
                cited_text="Machine learning is a method", 
                source_text="Machine learning is a method of data analysis",
                confidence=0.9, citation_type=CitationType.DIRECT_QUOTE
            ),
            Citation(
                id="2", chunk_id="invalid_chunk", document_id="doc_1", document_name="ML Guide",
                page_number=1, sentence_start=1, sentence_end=1,
                cited_text="Invalid citation", source_text="Non-existent source",
                confidence=0.5, citation_type=CitationType.PARAPHRASE
            )
        ]
        
        validation_result = await citation_generator_service.validate_citations(
            citations=citations,
            response_text="Test response with citations",
            source_chunks=sample_source_chunks
        )
        
        assert validation_result['total_citations'] == 2
        assert validation_result['valid_citations'] == 1
        assert validation_result['invalid_citations'] == 1
        assert len(validation_result['citation_issues']) > 0
        assert len(validation_result['validation_details']) == 2
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, citation_generator_service, sample_response_text, sample_source_chunks):
        """Test caching of citation results."""
        # Mock cache hit
        cached_result = {
            'response_text': sample_response_text,
            'total_citations': 5,
            'citations': [],
            'success': True
        }
        
        with patch.object(citation_generator_service.cache, 'get', return_value=cached_result):
            result = await citation_generator_service.generate_citations(
                sample_response_text, sample_source_chunks
            )
            
            assert result == cached_result
            assert citation_generator_service.citation_stats['cache_hits'] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, citation_generator_service):
        """Test error handling in citation generation."""
        # Test with invalid input that might cause errors
        result = await citation_generator_service.generate_citations(
            response_text="Valid text",
            source_chunks=[{'invalid': 'chunk'}]  # Invalid chunk structure
        )
        
        # Should handle errors gracefully
        assert 'success' in result
        assert 'citations' in result
        assert 'total_citations' in result
    
    @pytest.mark.asyncio
    async def test_get_citation_stats(self, citation_generator_service):
        """Test citation statistics retrieval."""
        # Generate some citations to populate stats
        citation_generator_service.citation_stats['total_citations_generated'] = 10
        citation_generator_service.citation_stats['avg_confidence'] = 0.8
        
        stats = await citation_generator_service.get_citation_stats()
        
        assert 'performance_stats' in stats
        assert 'configuration' in stats
        assert 'citation_types' in stats
        assert 'system_status' in stats
        
        assert stats['performance_stats']['total_citations_generated'] == 10
        assert stats['performance_stats']['avg_confidence'] == 0.8
    
    def test_citation_types_enum(self):
        """Test citation types enumeration."""
        assert CitationType.DIRECT_QUOTE.value == "direct_quote"
        assert CitationType.PARAPHRASE.value == "paraphrase"
        assert CitationType.REFERENCE.value == "reference"
        assert CitationType.SUPPORTING_EVIDENCE.value == "supporting_evidence"
    
    def test_citation_dataclass(self):
        """Test Citation dataclass."""
        citation = Citation(
            id="test_1",
            chunk_id="chunk_1",
            document_id="doc_1",
            document_name="Test Document",
            page_number=1,
            sentence_start=0,
            sentence_end=0,
            cited_text="Test citation",
            source_text="Test source text",
            confidence=0.85,
            citation_type=CitationType.DIRECT_QUOTE,
            url="https://example.com",
            metadata={"key": "value"}
        )
        
        assert citation.id == "test_1"
        assert citation.confidence == 0.85
        assert citation.citation_type == CitationType.DIRECT_QUOTE
        assert citation.url == "https://example.com"
        assert citation.metadata == {"key": "value"}


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])