"""
Tests for citation quality validation system.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from app.services.response.citation_validator import (
    CitationValidator, ValidationLevel, ValidationIssueType, ValidationIssue, ValidationResult
)
from app.services.response.citation_generator import Citation, CitationType


@pytest.fixture
def sample_citations():
    """Sample citations for testing."""
    return [
        Citation(
            id="cite_1",
            chunk_id="chunk_1",
            document_id="doc_1",
            document_name="ML Guide",
            page_number=1,
            sentence_start=0,
            sentence_end=0,
            cited_text="Machine learning is a method of data analysis",
            source_text="Machine learning is a method of data analysis that automates analytical model building",
            confidence=0.9,
            citation_type=CitationType.DIRECT_QUOTE,
            url="https://example.com/ml-guide"
        ),
        Citation(
            id="cite_2",
            chunk_id="chunk_2",
            document_id="doc_1",
            document_name="ML Guide",
            page_number=2,
            sentence_start=1,
            sentence_end=1,
            cited_text="Deep learning uses neural networks",
            source_text="Deep learning is part of machine learning that uses artificial neural networks with multiple layers",
            confidence=0.8,
            citation_type=CitationType.PARAPHRASE,
            url="https://example.com/ml-guide"
        ),
        Citation(
            id="cite_3",
            chunk_id="chunk_3",
            document_id="doc_2",
            document_name="Python Guide",
            page_number=1,
            sentence_start=2,
            sentence_end=2,
            cited_text="Python is popular for data science",
            source_text="Python programming language is widely used in data science applications",
            confidence=0.7,
            citation_type=CitationType.SUPPORTING_EVIDENCE,
            url="https://example.com/python-guide"
        )
    ]


@pytest.fixture
def sample_response_text():
    """Sample response text for testing."""
    return """
    Machine learning is a method of data analysis that helps automate decision making.
    Deep learning uses neural networks to process complex patterns in data.
    Python is popular for data science due to its extensive libraries and ease of use.
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
            'content': 'Deep learning is part of machine learning that uses artificial neural networks with multiple layers. These networks can learn complex patterns in data through multiple processing layers.',
            'metadata': {
                'document_name': 'ML Guide',
                'page_number': 2
            },
            'url': 'https://example.com/ml-guide'
        },
        {
            'chunk_id': 'chunk_3',
            'document_id': 'doc_2',
            'content': 'Python programming language is widely used in data science applications. It offers extensive libraries like pandas, numpy, and scikit-learn for data analysis and machine learning.',
            'metadata': {
                'document_name': 'Python Guide',
                'page_number': 1
            },
            'url': 'https://example.com/python-guide'
        }
    ]


@pytest.fixture
def citation_validator_service():
    """Create citation validator service instance for testing."""
    with patch('app.services.response.citation_validator.document_store') as mock_document_store:
        mock_document_store._initialized = True
        mock_document_store.initialize = AsyncMock()
        
        with patch('app.services.response.citation_validator.embedding_service') as mock_embedding_service:
            mock_embedding_service._initialized = True
            mock_embedding_service.initialize = AsyncMock()
            
            with patch('app.services.response.citation_validator.cache_manager') as mock_cache:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock()
                
                service = CitationValidator(
                    min_accuracy_threshold=0.7,
                    min_relevance_threshold=0.6,
                    min_confidence_threshold=0.5,
                    enable_caching=True,
                    cache_ttl=3600
                )
                return service


class TestCitationValidator:
    """Test cases for citation quality validation."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, citation_validator_service):
        """Test service initialization."""
        await citation_validator_service.initialize()
        
        assert citation_validator_service._initialized == True
        assert citation_validator_service.document_store.initialize.called
        assert citation_validator_service.embedding_service.initialize.called
    
    @pytest.mark.asyncio
    async def test_validate_citations_basic(self, citation_validator_service, sample_citations, sample_response_text, sample_source_chunks):
        """Test basic citation validation."""
        citation_validator_service._initialized = True
        
        result = await citation_validator_service.validate_citations(
            citations=sample_citations,
            response_text=sample_response_text,
            source_chunks=sample_source_chunks,
            validation_level="basic"
        )
        
        assert result['success'] == True
        assert result['total_citations'] == len(sample_citations)
        assert 'valid_citations' in result
        assert 'invalid_citations' in result
        assert 'overall_metrics' in result
        assert 'citation_results' in result
        assert len(result['citation_results']) == len(sample_citations)
    
    @pytest.mark.asyncio
    async def test_validate_citations_standard(self, citation_validator_service, sample_citations, sample_response_text, sample_source_chunks):
        """Test standard citation validation."""
        citation_validator_service._initialized = True
        
        result = await citation_validator_service.validate_citations(
            citations=sample_citations,
            response_text=sample_response_text,
            source_chunks=sample_source_chunks,
            validation_level="standard"
        )
        
        assert result['success'] == True
        assert result['validation_level'] == "standard"
        assert 'overall_metrics' in result
        
        # Check that relevance scores are calculated
        for citation_result in result['citation_results']:
            assert 'relevance_score' in citation_result
            assert citation_result['relevance_score'] >= 0
    
    @pytest.mark.asyncio
    async def test_validate_citations_comprehensive(self, citation_validator_service, sample_citations, sample_response_text, sample_source_chunks):
        """Test comprehensive citation validation."""
        citation_validator_service._initialized = True
        
        result = await citation_validator_service.validate_citations(
            citations=sample_citations,
            response_text=sample_response_text,
            source_chunks=sample_source_chunks,
            validation_level="comprehensive"
        )
        
        assert result['success'] == True
        assert result['validation_level'] == "comprehensive"
        
        # Check that comprehensive validation details are included
        for citation_result in result['citation_results']:
            assert 'validation_details' in citation_result
            if 'comprehensive_checks' in citation_result['validation_details']:
                assert 'context_validation' in citation_result['validation_details']['comprehensive_checks']
                assert 'formatting_validation' in citation_result['validation_details']['comprehensive_checks']
    
    @pytest.mark.asyncio
    async def test_validate_single_citation_valid(self, citation_validator_service, sample_citations, sample_source_chunks, sample_response_text):
        """Test validation of a single valid citation."""
        citation_validator_service._initialized = True
        
        chunk_lookup = {chunk['chunk_id']: chunk for chunk in sample_source_chunks}
        
        result = await citation_validator_service._validate_single_citation(
            citation=sample_citations[0],  # Direct quote citation
            chunk_lookup=chunk_lookup,
            response_text=sample_response_text,
            validation_level=ValidationLevel.STANDARD
        )
        
        assert isinstance(result, ValidationResult)
        assert result.citation_id == sample_citations[0].id
        assert result.accuracy_score > 0.7  # Should be high for direct quote
        assert result.confidence_score > 0
        assert result.overall_quality_score > 0
    
    @pytest.mark.asyncio
    async def test_validate_single_citation_missing_source(self, citation_validator_service, sample_response_text):
        """Test validation of citation with missing source."""
        citation_validator_service._initialized = True
        
        # Citation with non-existent chunk
        invalid_citation = Citation(
            id="invalid_cite",
            chunk_id="non_existent_chunk",
            document_id="doc_1",
            document_name="Test Doc",
            page_number=1,
            sentence_start=0,
            sentence_end=0,
            cited_text="Some text",
            source_text="Some source text",
            confidence=0.8,
            citation_type=CitationType.DIRECT_QUOTE
        )
        
        result = await citation_validator_service._validate_single_citation(
            citation=invalid_citation,
            chunk_lookup={},  # Empty chunk lookup
            response_text=sample_response_text,
            validation_level=ValidationLevel.STANDARD
        )
        
        assert result.is_valid == False
        assert result.accuracy_score == 0.0
        assert len(result.issues) > 0
        assert any(issue.issue_type == ValidationIssueType.MISSING_SOURCE for issue in result.issues)
    
    @pytest.mark.asyncio
    async def test_validate_accuracy_direct_quote(self, citation_validator_service, sample_source_chunks):
        """Test accuracy validation for direct quotes."""
        citation_validator_service._initialized = True
        
        # Perfect direct quote
        perfect_citation = Citation(
            id="perfect_cite",
            chunk_id="chunk_1",
            document_id="doc_1",
            document_name="ML Guide",
            page_number=1,
            sentence_start=0,
            sentence_end=0,
            cited_text="Machine learning is a method of data analysis",
            source_text="Machine learning is a method of data analysis that automates analytical model building",
            confidence=0.9,
            citation_type=CitationType.DIRECT_QUOTE
        )
        
        issues = []
        accuracy_score = await citation_validator_service._validate_accuracy(
            perfect_citation,
            sample_source_chunks[0]['content'],
            issues
        )
        
        assert accuracy_score > 0.9  # Should be very high for exact match
        assert len(issues) == 0  # No issues for perfect quote
    
    @pytest.mark.asyncio
    async def test_validate_accuracy_inaccurate_quote(self, citation_validator_service, sample_source_chunks):
        """Test accuracy validation for inaccurate quotes."""
        citation_validator_service._initialized = True
        
        # Inaccurate direct quote
        inaccurate_citation = Citation(
            id="inaccurate_cite",
            chunk_id="chunk_1",
            document_id="doc_1",
            document_name="ML Guide",
            page_number=1,
            sentence_start=0,
            sentence_end=0,
            cited_text="Machine learning is completely different",  # Inaccurate
            source_text="Machine learning is a method of data analysis that automates analytical model building",
            confidence=0.9,
            citation_type=CitationType.DIRECT_QUOTE
        )
        
        issues = []
        accuracy_score = await citation_validator_service._validate_accuracy(
            inaccurate_citation,
            sample_source_chunks[0]['content'],
            issues
        )
        
        assert accuracy_score < 0.8  # Should be low for inaccurate quote
        assert len(issues) > 0  # Should have accuracy issues
        assert any(issue.issue_type == ValidationIssueType.INACCURATE_QUOTE for issue in issues)
    
    @pytest.mark.asyncio
    async def test_validate_relevance(self, citation_validator_service, sample_citations, sample_response_text):
        """Test relevance validation."""
        citation_validator_service._initialized = True
        
        issues = []
        relevance_score = await citation_validator_service._validate_relevance(
            sample_citations[0],
            sample_response_text,
            issues
        )
        
        assert 0 <= relevance_score <= 1
        # Should have reasonable relevance since citation matches response content
        assert relevance_score > 0.3
    
    @pytest.mark.asyncio
    async def test_validate_context(self, citation_validator_service, sample_citations, sample_source_chunks, sample_response_text):
        """Test context validation."""
        citation_validator_service._initialized = True
        
        issues = []
        await citation_validator_service._validate_context(
            sample_citations[0],
            sample_source_chunks[0],
            sample_response_text,
            issues
        )
        
        # Should not have context issues for properly matched citation
        context_issues = [issue for issue in issues if issue.issue_type == ValidationIssueType.CONTEXT_MISMATCH]
        assert len(context_issues) == 0
    
    @pytest.mark.asyncio
    async def test_validate_formatting(self, citation_validator_service, sample_citations):
        """Test formatting validation."""
        citation_validator_service._initialized = True
        
        # Test valid citation
        issues = []
        await citation_validator_service._validate_formatting(sample_citations[0], issues)
        
        formatting_issues = [issue for issue in issues if issue.issue_type == ValidationIssueType.FORMATTING_ERROR]
        assert len(formatting_issues) == 0  # Should have no formatting issues
        
        # Test citation with missing fields
        invalid_citation = Citation(
            id="",  # Missing ID
            chunk_id="",  # Missing chunk ID
            document_id="doc_1",
            document_name="Test Doc",
            page_number=1,
            sentence_start=0,
            sentence_end=0,
            cited_text="",  # Empty cited text
            source_text="",  # Empty source text
            confidence=0.8,
            citation_type=CitationType.DIRECT_QUOTE
        )
        
        issues = []
        await citation_validator_service._validate_formatting(invalid_citation, issues)
        
        formatting_issues = [issue for issue in issues if issue.issue_type == ValidationIssueType.FORMATTING_ERROR]
        assert len(formatting_issues) > 0  # Should have multiple formatting issues
    
    def test_validate_confidence(self, citation_validator_service, sample_citations):
        """Test confidence validation."""
        citation_validator_service._initialized = True
        
        # Test citation with good confidence
        issues = []
        confidence_score = citation_validator_service._validate_confidence(sample_citations[0], issues)
        
        assert confidence_score > 0
        assert confidence_score <= 1.0
        
        # Test citation with low confidence
        low_confidence_citation = Citation(
            id="low_conf_cite",
            chunk_id="chunk_1",
            document_id="doc_1",
            document_name="Test Doc",
            page_number=1,
            sentence_start=0,
            sentence_end=0,
            cited_text="Some text",
            source_text="Some source text",
            confidence=0.3,  # Below threshold
            citation_type=CitationType.DIRECT_QUOTE
        )
        
        issues = []
        confidence_score = citation_validator_service._validate_confidence(low_confidence_citation, issues)
        
        weak_confidence_issues = [issue for issue in issues if issue.issue_type == ValidationIssueType.WEAK_CONFIDENCE]
        assert len(weak_confidence_issues) > 0
    
    @pytest.mark.asyncio
    async def test_cross_citation_validation(self, citation_validator_service, sample_citations):
        """Test cross-citation validation."""
        citation_validator_service._initialized = True
        
        # Create duplicate citations
        duplicate_citations = sample_citations + [
            Citation(
                id="duplicate_cite",
                chunk_id="chunk_1",  # Same chunk
                document_id="doc_1",
                document_name="ML Guide",
                page_number=1,
                sentence_start=0,
                sentence_end=0,
                cited_text="Machine learning is a method of data analysis",  # Same text
                source_text="Machine learning is a method of data analysis that automates analytical model building",
                confidence=0.9,
                citation_type=CitationType.DIRECT_QUOTE
            )
        ]
        
        validation_results = []  # Mock validation results
        
        cross_issues = await citation_validator_service._perform_cross_citation_validation(
            duplicate_citations, validation_results
        )
        
        duplicate_issues = [issue for issue in cross_issues if issue.issue_type == ValidationIssueType.DUPLICATE_CITATION]
        assert len(duplicate_issues) > 0
    
    def test_calculate_quality_score(self, citation_validator_service):
        """Test quality score calculation."""
        citation_validator_service._initialized = True
        
        # Test with high scores and no issues
        quality_score = citation_validator_service._calculate_quality_score(
            accuracy_score=0.9,
            relevance_score=0.8,
            confidence_score=0.85,
            issues=[]
        )
        
        assert 0.8 <= quality_score <= 1.0  # Should be high
        
        # Test with issues
        issues = [
            ValidationIssue(
                citation_id="test",
                issue_type=ValidationIssueType.INACCURATE_QUOTE,
                severity="high",
                description="Test issue"
            )
        ]
        
        quality_score_with_issues = citation_validator_service._calculate_quality_score(
            accuracy_score=0.9,
            relevance_score=0.8,
            confidence_score=0.85,
            issues=issues
        )
        
        assert quality_score_with_issues < quality_score  # Should be lower due to issues
    
    def test_calculate_overall_metrics(self, citation_validator_service):
        """Test overall metrics calculation."""
        citation_validator_service._initialized = True
        
        validation_results = [
            ValidationResult(
                citation_id="1",
                is_valid=True,
                accuracy_score=0.9,
                relevance_score=0.8,
                confidence_score=0.85,
                overall_quality_score=0.85,
                issues=[],
                validation_details={}
            ),
            ValidationResult(
                citation_id="2",
                is_valid=False,
                accuracy_score=0.5,
                relevance_score=0.4,
                confidence_score=0.3,
                overall_quality_score=0.4,
                issues=[ValidationIssue(
                    citation_id="2",
                    issue_type=ValidationIssueType.LOW_RELEVANCE,
                    severity="medium",
                    description="Low relevance"
                )],
                validation_details={}
            )
        ]
        
        metrics = citation_validator_service._calculate_overall_metrics(validation_results)
        
        assert 'avg_accuracy' in metrics
        assert 'avg_relevance' in metrics
        assert 'avg_confidence' in metrics
        assert 'avg_quality' in metrics
        assert 'validation_rate' in metrics
        assert 'issue_distribution' in metrics
        
        assert metrics['validation_rate'] == 0.5  # 1 out of 2 valid
        assert metrics['avg_accuracy'] == 0.7  # (0.9 + 0.5) / 2
    
    @pytest.mark.asyncio
    async def test_calculate_semantic_similarity(self, citation_validator_service):
        """Test semantic similarity calculation."""
        citation_validator_service._initialized = True
        
        # Similar texts
        similarity1 = await citation_validator_service._calculate_semantic_similarity(
            "Machine learning processes data",
            "ML algorithms analyze information"
        )
        
        # Dissimilar texts
        similarity2 = await citation_validator_service._calculate_semantic_similarity(
            "Machine learning processes data",
            "The weather is sunny today"
        )
        
        assert 0 <= similarity1 <= 1
        assert 0 <= similarity2 <= 1
        assert similarity1 > similarity2  # Similar texts should have higher similarity
    
    def test_split_into_sentences(self, citation_validator_service):
        """Test sentence splitting."""
        citation_validator_service._initialized = True
        
        text = "This is sentence one. This is sentence two! Is this sentence three?"
        sentences = citation_validator_service._split_into_sentences(text)
        
        assert len(sentences) == 3
        assert "This is sentence one" in sentences[0]
        assert "This is sentence two" in sentences[1]
        assert "Is this sentence three" in sentences[2]
    
    def test_generate_cache_key(self, citation_validator_service, sample_citations, sample_response_text):
        """Test cache key generation."""
        citation_validator_service._initialized = True
        
        cache_key = citation_validator_service._generate_cache_key(
            sample_citations, sample_response_text, "standard"
        )
        
        assert cache_key.startswith("citation_validation:")
        assert isinstance(cache_key, str)
        
        # Same inputs should generate same key
        cache_key2 = citation_validator_service._generate_cache_key(
            sample_citations, sample_response_text, "standard"
        )
        assert cache_key == cache_key2
        
        # Different inputs should generate different keys
        cache_key3 = citation_validator_service._generate_cache_key(
            sample_citations, "different text", "standard"
        )
        assert cache_key != cache_key3
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, citation_validator_service, sample_citations, sample_response_text, sample_source_chunks):
        """Test caching of validation results."""
        citation_validator_service._initialized = True
        
        # Mock cache hit
        cached_result = {
            'total_citations': len(sample_citations),
            'valid_citations': len(sample_citations),
            'success': True
        }
        
        with patch.object(citation_validator_service.cache, 'get', return_value=cached_result):
            result = await citation_validator_service.validate_citations(
                sample_citations, sample_response_text, sample_source_chunks
            )
            
            assert result == cached_result
            assert citation_validator_service.validation_stats['cache_hits'] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, citation_validator_service):
        """Test error handling in validation."""
        citation_validator_service._initialized = True
        
        # Test with invalid input that might cause errors
        result = await citation_validator_service.validate_citations(
            citations=[],  # Empty citations
            response_text="",  # Empty response
            source_chunks=[],  # Empty chunks
            validation_level="invalid_level"  # Invalid level
        )
        
        # Should handle errors gracefully
        assert 'success' in result
        assert 'total_citations' in result
        assert 'validation_time' in result
    
    @pytest.mark.asyncio
    async def test_get_validation_stats(self, citation_validator_service):
        """Test validation statistics retrieval."""
        citation_validator_service._initialized = True
        
        # Populate some stats
        citation_validator_service.validation_stats['total_validations'] = 5
        citation_validator_service.validation_stats['validation_levels']['standard'] = 3
        
        stats = await citation_validator_service.get_validation_stats()
        
        assert 'performance_stats' in stats
        assert 'configuration' in stats
        assert 'validation_levels' in stats
        assert 'issue_types' in stats
        assert 'system_status' in stats
        
        assert stats['performance_stats']['total_validations'] == 5
        assert stats['performance_stats']['validation_levels']['standard'] == 3
    
    def test_validation_enums(self):
        """Test validation enums."""
        # Test ValidationLevel enum
        assert ValidationLevel.BASIC.value == "basic"
        assert ValidationLevel.STANDARD.value == "standard"
        assert ValidationLevel.COMPREHENSIVE.value == "comprehensive"
        
        # Test ValidationIssueType enum
        assert ValidationIssueType.MISSING_SOURCE.value == "missing_source"
        assert ValidationIssueType.INACCURATE_QUOTE.value == "inaccurate_quote"
        assert ValidationIssueType.LOW_RELEVANCE.value == "low_relevance"
        assert ValidationIssueType.WEAK_CONFIDENCE.value == "weak_confidence"
    
    def test_validation_dataclasses(self):
        """Test validation dataclasses."""
        # Test ValidationIssue
        issue = ValidationIssue(
            citation_id="test_cite",
            issue_type=ValidationIssueType.INACCURATE_QUOTE,
            severity="high",
            description="Test issue",
            suggested_fix="Fix the issue",
            confidence_impact=-0.2
        )
        
        assert issue.citation_id == "test_cite"
        assert issue.issue_type == ValidationIssueType.INACCURATE_QUOTE
        assert issue.severity == "high"
        assert issue.confidence_impact == -0.2
        
        # Test ValidationResult
        result = ValidationResult(
            citation_id="test_cite",
            is_valid=True,
            accuracy_score=0.9,
            relevance_score=0.8,
            confidence_score=0.85,
            overall_quality_score=0.85,
            issues=[issue],
            validation_details={"test": "data"}
        )
        
        assert result.citation_id == "test_cite"
        assert result.is_valid == True
        assert result.accuracy_score == 0.9
        assert len(result.issues) == 1
        assert result.validation_details == {"test": "data"}


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])