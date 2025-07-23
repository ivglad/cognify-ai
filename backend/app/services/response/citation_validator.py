"""
Citation quality validation system for ensuring accuracy and relevance of citations.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import time
import re
from dataclasses import dataclass
from enum import Enum
import difflib

import trio

from app.services.response.citation_generator import Citation, CitationType
from app.services.database.document_store import document_store
from app.services.embeddings.embedding_service import embedding_service
from app.core.config import settings
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Citation validation levels."""
    BASIC = "basic"          # Basic checks (existence, format)
    STANDARD = "standard"    # Standard checks (accuracy, relevance)
    COMPREHENSIVE = "comprehensive"  # Full validation (semantic, context)


class ValidationIssueType(Enum):
    """Types of validation issues."""
    MISSING_SOURCE = "missing_source"
    INACCURATE_QUOTE = "inaccurate_quote"
    LOW_RELEVANCE = "low_relevance"
    WEAK_CONFIDENCE = "weak_confidence"
    CONTEXT_MISMATCH = "context_mismatch"
    DUPLICATE_CITATION = "duplicate_citation"
    FORMATTING_ERROR = "formatting_error"


@dataclass
class ValidationIssue:
    """Validation issue data structure."""
    citation_id: str
    issue_type: ValidationIssueType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    suggested_fix: Optional[str] = None
    confidence_impact: float = 0.0


@dataclass
class ValidationResult:
    """Citation validation result."""
    citation_id: str
    is_valid: bool
    accuracy_score: float
    relevance_score: float
    confidence_score: float
    overall_quality_score: float
    issues: List[ValidationIssue]
    validation_details: Dict[str, Any]


class CitationValidator:
    """
    Citation quality validation system for ensuring accuracy and relevance.
    """
    
    def __init__(self,
                 min_accuracy_threshold: float = 0.7,
                 min_relevance_threshold: float = 0.6,
                 min_confidence_threshold: float = 0.5,
                 enable_caching: bool = True,
                 cache_ttl: int = 3600):
        """
        Initialize citation validator.
        
        Args:
            min_accuracy_threshold: Minimum accuracy score for valid citations
            min_relevance_threshold: Minimum relevance score for valid citations
            min_confidence_threshold: Minimum confidence score for valid citations
            enable_caching: Whether to cache validation results
            cache_ttl: Cache time-to-live in seconds
        """
        self.min_accuracy_threshold = min_accuracy_threshold
        self.min_relevance_threshold = min_relevance_threshold
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        self.document_store = document_store
        self.embedding_service = embedding_service
        self.cache = cache_manager
        
        self._initialized = False
        
        # Performance tracking
        self.validation_stats = {
            'total_validations': 0,
            'avg_validation_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_levels': {
                'basic': 0,
                'standard': 0,
                'comprehensive': 0
            },
            'issue_types': {issue_type.value: 0 for issue_type in ValidationIssueType},
            'avg_quality_scores': {
                'accuracy': 0.0,
                'relevance': 0.0,
                'confidence': 0.0,
                'overall': 0.0
            }
        }
    
    async def initialize(self):
        """Initialize citation validator."""
        if self._initialized:
            return
        
        try:
            # Initialize document store
            if not self.document_store._initialized:
                await self.document_store.initialize()
            
            # Initialize embedding service
            if not self.embedding_service._initialized:
                await self.embedding_service.initialize()
            
            self._initialized = True
            logger.info("Citation validator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize citation validator: {e}")
            self._initialized = False
            raise
    
    async def validate_citations(self,
                               citations: List[Citation],
                               response_text: str,
                               source_chunks: List[Dict[str, Any]],
                               validation_level: str = "standard") -> Dict[str, Any]:
        """
        Validate a list of citations for quality and accuracy.
        
        Args:
            citations: List of citations to validate
            response_text: Original response text
            source_chunks: Source chunks used for citation
            validation_level: Level of validation to perform
            
        Returns:
            Comprehensive validation results
        """
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Validate validation level
            try:
                validation_level_enum = ValidationLevel(validation_level.lower())
            except ValueError:
                validation_level_enum = ValidationLevel.STANDARD
                logger.warning(f"Invalid validation level '{validation_level}', using standard")
            
            # Check cache
            if self.enable_caching:
                cache_key = self._generate_cache_key(citations, response_text, validation_level)
                cached_result = await self.cache.get(cache_key)
                
                if cached_result:
                    self.validation_stats['cache_hits'] += 1
                    logger.debug("Using cached validation result")
                    return cached_result
                
                self.validation_stats['cache_misses'] += 1
            
            # Create chunk lookup for efficient access
            chunk_lookup = {chunk.get('chunk_id', ''): chunk for chunk in source_chunks}
            
            # Validate each citation
            validation_results = []
            
            for citation in citations:
                validation_result = await self._validate_single_citation(
                    citation, chunk_lookup, response_text, validation_level_enum
                )
                validation_results.append(validation_result)
            
            # Perform cross-citation validation
            cross_validation_issues = await self._perform_cross_citation_validation(
                citations, validation_results
            )
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(validation_results)
            
            # Create comprehensive result
            result = {
                'total_citations': len(citations),
                'valid_citations': sum(1 for vr in validation_results if vr.is_valid),
                'invalid_citations': sum(1 for vr in validation_results if not vr.is_valid),
                'validation_level': validation_level_enum.value,
                'overall_metrics': overall_metrics,
                'citation_results': [
                    {
                        'citation_id': vr.citation_id,
                        'is_valid': vr.is_valid,
                        'accuracy_score': vr.accuracy_score,
                        'relevance_score': vr.relevance_score,
                        'confidence_score': vr.confidence_score,
                        'overall_quality_score': vr.overall_quality_score,
                        'issues': [
                            {
                                'issue_type': issue.issue_type.value,
                                'severity': issue.severity,
                                'description': issue.description,
                                'suggested_fix': issue.suggested_fix,
                                'confidence_impact': issue.confidence_impact
                            }
                            for issue in vr.issues
                        ],
                        'validation_details': vr.validation_details
                    }
                    for vr in validation_results
                ],
                'cross_validation_issues': [
                    {
                        'issue_type': issue.issue_type.value,
                        'severity': issue.severity,
                        'description': issue.description,
                        'suggested_fix': issue.suggested_fix
                    }
                    for issue in cross_validation_issues
                ],
                'validation_time': time.time() - start_time,
                'success': True
            }
            
            # Cache result
            if self.enable_caching:
                await self.cache.set(cache_key, result, ttl=self.cache_ttl)
            
            # Update stats
            self._update_validation_stats(result, validation_level_enum.value)
            
            logger.info(f"Citation validation completed: {result['valid_citations']}/{result['total_citations']} valid in {result['validation_time']:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Citation validation failed: {e}")
            return self._create_error_response(str(e), start_time)
    
    async def _validate_single_citation(self,
                                      citation: Citation,
                                      chunk_lookup: Dict[str, Dict[str, Any]],
                                      response_text: str,
                                      validation_level: ValidationLevel) -> ValidationResult:
        """Validate a single citation."""
        try:
            issues = []
            validation_details = {}
            
            # Get source chunk
            chunk = chunk_lookup.get(citation.chunk_id)
            if not chunk:
                issues.append(ValidationIssue(
                    citation_id=citation.id,
                    issue_type=ValidationIssueType.MISSING_SOURCE,
                    severity="critical",
                    description=f"Source chunk {citation.chunk_id} not found",
                    suggested_fix="Verify chunk ID and ensure source is available",
                    confidence_impact=-0.5
                ))
                
                return ValidationResult(
                    citation_id=citation.id,
                    is_valid=False,
                    accuracy_score=0.0,
                    relevance_score=0.0,
                    confidence_score=0.0,
                    overall_quality_score=0.0,
                    issues=issues,
                    validation_details=validation_details
                )
            
            chunk_content = chunk.get('content', '')
            
            # Basic validation
            accuracy_score = await self._validate_accuracy(citation, chunk_content, issues)
            validation_details['accuracy_checks'] = {
                'source_text_found': citation.source_text in chunk_content,
                'text_similarity': accuracy_score
            }
            
            # Standard validation
            relevance_score = 0.0
            if validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                relevance_score = await self._validate_relevance(citation, response_text, issues)
                validation_details['relevance_checks'] = {
                    'semantic_similarity': relevance_score,
                    'context_match': relevance_score > 0.5
                }
            
            # Comprehensive validation
            if validation_level == ValidationLevel.COMPREHENSIVE:
                await self._validate_context(citation, chunk, response_text, issues)
                await self._validate_formatting(citation, issues)
                validation_details['comprehensive_checks'] = {
                    'context_validation': True,
                    'formatting_validation': True
                }
            
            # Confidence validation
            confidence_score = self._validate_confidence(citation, issues)
            validation_details['confidence_checks'] = {
                'original_confidence': citation.confidence,
                'adjusted_confidence': confidence_score
            }
            
            # Calculate overall quality score
            overall_quality_score = self._calculate_quality_score(
                accuracy_score, relevance_score, confidence_score, issues
            )
            
            # Determine if citation is valid
            is_valid = (
                accuracy_score >= self.min_accuracy_threshold and
                relevance_score >= self.min_relevance_threshold and
                confidence_score >= self.min_confidence_threshold and
                not any(issue.severity == "critical" for issue in issues)
            )
            
            return ValidationResult(
                citation_id=citation.id,
                is_valid=is_valid,
                accuracy_score=accuracy_score,
                relevance_score=relevance_score,
                confidence_score=confidence_score,
                overall_quality_score=overall_quality_score,
                issues=issues,
                validation_details=validation_details
            )
            
        except Exception as e:
            logger.error(f"Single citation validation failed: {e}")
            return ValidationResult(
                citation_id=citation.id,
                is_valid=False,
                accuracy_score=0.0,
                relevance_score=0.0,
                confidence_score=0.0,
                overall_quality_score=0.0,
                issues=[ValidationIssue(
                    citation_id=citation.id,
                    issue_type=ValidationIssueType.FORMATTING_ERROR,
                    severity="critical",
                    description=f"Validation error: {str(e)}",
                    confidence_impact=-1.0
                )],
                validation_details={'error': str(e)}
            )
    
    async def _validate_accuracy(self,
                               citation: Citation,
                               chunk_content: str,
                               issues: List[ValidationIssue]) -> float:
        """Validate citation accuracy."""
        try:
            accuracy_score = 0.0
            
            # Check if source text exists in chunk
            if citation.source_text not in chunk_content:
                issues.append(ValidationIssue(
                    citation_id=citation.id,
                    issue_type=ValidationIssueType.MISSING_SOURCE,
                    severity="high",
                    description="Source text not found in chunk content",
                    suggested_fix="Verify source text extraction",
                    confidence_impact=-0.3
                ))
                return 0.0
            
            # Validate based on citation type
            if citation.citation_type == CitationType.DIRECT_QUOTE:
                # For direct quotes, check exact match
                if citation.cited_text.lower().strip() in citation.source_text.lower().strip():
                    accuracy_score = 0.95
                else:
                    # Check for close match using difflib
                    similarity = difflib.SequenceMatcher(
                        None, 
                        citation.cited_text.lower().strip(),
                        citation.source_text.lower().strip()
                    ).ratio()
                    
                    accuracy_score = similarity
                    
                    if similarity < 0.8:
                        issues.append(ValidationIssue(
                            citation_id=citation.id,
                            issue_type=ValidationIssueType.INACCURATE_QUOTE,
                            severity="medium",
                            description=f"Direct quote accuracy: {similarity:.2f}",
                            suggested_fix="Verify exact text match for direct quotes",
                            confidence_impact=-0.2
                        ))
            
            elif citation.citation_type == CitationType.PARAPHRASE:
                # For paraphrases, use semantic similarity
                accuracy_score = await self._calculate_semantic_similarity(
                    citation.cited_text, citation.source_text
                )
                
                if accuracy_score < 0.6:
                    issues.append(ValidationIssue(
                        citation_id=citation.id,
                        issue_type=ValidationIssueType.LOW_RELEVANCE,
                        severity="medium",
                        description=f"Paraphrase similarity: {accuracy_score:.2f}",
                        suggested_fix="Improve semantic alignment between cited and source text",
                        confidence_impact=-0.15
                    ))
            
            else:
                # For other types, use moderate similarity check
                accuracy_score = await self._calculate_semantic_similarity(
                    citation.cited_text, citation.source_text
                )
            
            return min(1.0, accuracy_score)
            
        except Exception as e:
            logger.error(f"Accuracy validation failed: {e}")
            return 0.0
    
    async def _validate_relevance(self,
                                citation: Citation,
                                response_text: str,
                                issues: List[ValidationIssue]) -> float:
        """Validate citation relevance to response."""
        try:
            # Extract sentence from response that contains the citation
            response_sentences = self._split_into_sentences(response_text)
            
            if citation.sentence_start >= len(response_sentences):
                issues.append(ValidationIssue(
                    citation_id=citation.id,
                    issue_type=ValidationIssueType.CONTEXT_MISMATCH,
                    severity="medium",
                    description="Citation sentence index out of range",
                    suggested_fix="Verify sentence indexing",
                    confidence_impact=-0.1
                ))
                return 0.0
            
            target_sentence = response_sentences[citation.sentence_start]
            
            # Calculate relevance using semantic similarity
            relevance_score = await self._calculate_semantic_similarity(
                target_sentence, citation.source_text
            )
            
            # Also check relevance to cited text
            cited_relevance = await self._calculate_semantic_similarity(
                citation.cited_text, citation.source_text
            )
            
            # Combine scores
            combined_relevance = (relevance_score + cited_relevance) / 2
            
            if combined_relevance < 0.5:
                issues.append(ValidationIssue(
                    citation_id=citation.id,
                    issue_type=ValidationIssueType.LOW_RELEVANCE,
                    severity="medium",
                    description=f"Low relevance score: {combined_relevance:.2f}",
                    suggested_fix="Ensure citation supports the specific claim",
                    confidence_impact=-0.1
                ))
            
            return combined_relevance
            
        except Exception as e:
            logger.error(f"Relevance validation failed: {e}")
            return 0.0
    
    async def _validate_context(self,
                              citation: Citation,
                              chunk: Dict[str, Any],
                              response_text: str,
                              issues: List[ValidationIssue]):
        """Validate citation context."""
        try:
            # Check document context
            document_metadata = chunk.get('metadata', {})
            
            # Validate page number if available
            if citation.page_number and document_metadata.get('page_number'):
                if citation.page_number != document_metadata.get('page_number'):
                    issues.append(ValidationIssue(
                        citation_id=citation.id,
                        issue_type=ValidationIssueType.CONTEXT_MISMATCH,
                        severity="low",
                        description="Page number mismatch",
                        suggested_fix="Verify page number accuracy",
                        confidence_impact=-0.05
                    ))
            
            # Check document name consistency
            if citation.document_name != document_metadata.get('document_name', ''):
                issues.append(ValidationIssue(
                    citation_id=citation.id,
                    issue_type=ValidationIssueType.CONTEXT_MISMATCH,
                    severity="low",
                    description="Document name mismatch",
                    suggested_fix="Verify document name consistency",
                    confidence_impact=-0.05
                ))
            
        except Exception as e:
            logger.error(f"Context validation failed: {e}")
    
    async def _validate_formatting(self,
                                 citation: Citation,
                                 issues: List[ValidationIssue]):
        """Validate citation formatting."""
        try:
            # Check required fields
            if not citation.id:
                issues.append(ValidationIssue(
                    citation_id=citation.id or "unknown",
                    issue_type=ValidationIssueType.FORMATTING_ERROR,
                    severity="medium",
                    description="Missing citation ID",
                    suggested_fix="Ensure all citations have unique IDs"
                ))
            
            if not citation.chunk_id:
                issues.append(ValidationIssue(
                    citation_id=citation.id,
                    issue_type=ValidationIssueType.FORMATTING_ERROR,
                    severity="high",
                    description="Missing chunk ID",
                    suggested_fix="Ensure citation links to valid chunk"
                ))
            
            if not citation.cited_text.strip():
                issues.append(ValidationIssue(
                    citation_id=citation.id,
                    issue_type=ValidationIssueType.FORMATTING_ERROR,
                    severity="medium",
                    description="Empty cited text",
                    suggested_fix="Ensure cited text is not empty"
                ))
            
            if not citation.source_text.strip():
                issues.append(ValidationIssue(
                    citation_id=citation.id,
                    issue_type=ValidationIssueType.FORMATTING_ERROR,
                    severity="medium",
                    description="Empty source text",
                    suggested_fix="Ensure source text is not empty"
                ))
            
        except Exception as e:
            logger.error(f"Formatting validation failed: {e}")
    
    def _validate_confidence(self,
                           citation: Citation,
                           issues: List[ValidationIssue]) -> float:
        """Validate and adjust citation confidence."""
        try:
            confidence_score = citation.confidence
            
            # Check confidence threshold
            if confidence_score < self.min_confidence_threshold:
                issues.append(ValidationIssue(
                    citation_id=citation.id,
                    issue_type=ValidationIssueType.WEAK_CONFIDENCE,
                    severity="medium",
                    description=f"Confidence {confidence_score:.2f} below threshold {self.min_confidence_threshold}",
                    suggested_fix="Improve citation quality or lower threshold",
                    confidence_impact=-0.1
                ))
            
            # Adjust confidence based on issues
            for issue in issues:
                if issue.citation_id == citation.id:
                    confidence_score += issue.confidence_impact
            
            return max(0.0, min(1.0, confidence_score))
            
        except Exception as e:
            logger.error(f"Confidence validation failed: {e}")
            return citation.confidence
    
    async def _perform_cross_citation_validation(self,
                                               citations: List[Citation],
                                               validation_results: List[ValidationResult]) -> List[ValidationIssue]:
        """Perform validation across multiple citations."""
        try:
            cross_issues = []
            
            # Check for duplicate citations
            seen_sources = set()
            for citation in citations:
                source_key = f"{citation.chunk_id}:{citation.source_text[:50]}"
                if source_key in seen_sources:
                    cross_issues.append(ValidationIssue(
                        citation_id=citation.id,
                        issue_type=ValidationIssueType.DUPLICATE_CITATION,
                        severity="low",
                        description="Duplicate or very similar citation found",
                        suggested_fix="Remove duplicate citations or ensure they support different claims"
                    ))
                else:
                    seen_sources.add(source_key)
            
            # Check citation distribution
            sentence_citation_count = {}
            for citation in citations:
                sentence_citation_count[citation.sentence_start] = sentence_citation_count.get(citation.sentence_start, 0) + 1
            
            # Flag sentences with too many citations
            for sentence_idx, count in sentence_citation_count.items():
                if count > 5:  # Threshold for too many citations
                    cross_issues.append(ValidationIssue(
                        citation_id="multiple",
                        issue_type=ValidationIssueType.FORMATTING_ERROR,
                        severity="low",
                        description=f"Sentence {sentence_idx} has {count} citations (may be excessive)",
                        suggested_fix="Consider consolidating citations for better readability"
                    ))
            
            return cross_issues
            
        except Exception as e:
            logger.error(f"Cross-citation validation failed: {e}")
            return []
    
    def _calculate_quality_score(self,
                               accuracy_score: float,
                               relevance_score: float,
                               confidence_score: float,
                               issues: List[ValidationIssue]) -> float:
        """Calculate overall quality score for citation."""
        try:
            # Base score from individual metrics
            base_score = (accuracy_score * 0.4 + relevance_score * 0.3 + confidence_score * 0.3)
            
            # Apply penalties for issues
            penalty = 0.0
            for issue in issues:
                if issue.severity == "critical":
                    penalty += 0.3
                elif issue.severity == "high":
                    penalty += 0.2
                elif issue.severity == "medium":
                    penalty += 0.1
                elif issue.severity == "low":
                    penalty += 0.05
            
            final_score = max(0.0, base_score - penalty)
            return min(1.0, final_score)
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.0
    
    def _calculate_overall_metrics(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Calculate overall validation metrics."""
        try:
            if not validation_results:
                return {
                    'avg_accuracy': 0.0,
                    'avg_relevance': 0.0,
                    'avg_confidence': 0.0,
                    'avg_quality': 0.0,
                    'validation_rate': 0.0,
                    'issue_distribution': {}
                }
            
            total_accuracy = sum(vr.accuracy_score for vr in validation_results)
            total_relevance = sum(vr.relevance_score for vr in validation_results)
            total_confidence = sum(vr.confidence_score for vr in validation_results)
            total_quality = sum(vr.overall_quality_score for vr in validation_results)
            
            valid_count = sum(1 for vr in validation_results if vr.is_valid)
            
            # Count issue types
            issue_distribution = {}
            for vr in validation_results:
                for issue in vr.issues:
                    issue_type = issue.issue_type.value
                    issue_distribution[issue_type] = issue_distribution.get(issue_type, 0) + 1
            
            return {
                'avg_accuracy': total_accuracy / len(validation_results),
                'avg_relevance': total_relevance / len(validation_results),
                'avg_confidence': total_confidence / len(validation_results),
                'avg_quality': total_quality / len(validation_results),
                'validation_rate': valid_count / len(validation_results),
                'issue_distribution': issue_distribution
            }
            
        except Exception as e:
            logger.error(f"Overall metrics calculation failed: {e}")
            return {}
    
    async def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            # Simple similarity based on word overlap (can be enhanced with embeddings)
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            # Jaccard similarity
            jaccard = intersection / union if union > 0 else 0.0
            
            # Boost similarity for similar sentence lengths
            length_similarity = 1.0 - abs(len(words1) - len(words2)) / max(len(words1), len(words2))
            
            return (jaccard + length_similarity) / 2
            
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        except Exception as e:
            logger.error(f"Sentence splitting failed: {e}")
            return [text]
    
    def _generate_cache_key(self,
                          citations: List[Citation],
                          response_text: str,
                          validation_level: str) -> str:
        """Generate cache key for validation results."""
        try:
            citation_ids = [c.id for c in citations]
            content_hash = hash(f"{':'.join(citation_ids)}:{response_text}:{validation_level}")
            return f"citation_validation:{content_hash}"
        except Exception:
            return f"citation_validation:{hash(response_text)}"
    
    def _create_error_response(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create error response for validation."""
        return {
            'total_citations': 0,
            'valid_citations': 0,
            'invalid_citations': 0,
            'validation_level': 'error',
            'overall_metrics': {},
            'citation_results': [],
            'cross_validation_issues': [],
            'validation_time': time.time() - start_time,
            'success': False,
            'error': error_message
        }
    
    def _update_validation_stats(self, result: Dict[str, Any], validation_level: str):
        """Update validation statistics."""
        try:
            self.validation_stats['total_validations'] += 1
            self.validation_stats['validation_levels'][validation_level] += 1
            
            # Update average validation time
            validation_time = result.get('validation_time', 0)
            current_avg = self.validation_stats['avg_validation_time']
            total_validations = self.validation_stats['total_validations']
            
            new_avg = ((current_avg * (total_validations - 1)) + validation_time) / total_validations
            self.validation_stats['avg_validation_time'] = new_avg
            
            # Update issue type counts
            for citation_result in result.get('citation_results', []):
                for issue in citation_result.get('issues', []):
                    issue_type = issue['issue_type']
                    self.validation_stats['issue_types'][issue_type] += 1
            
            # Update quality scores
            overall_metrics = result.get('overall_metrics', {})
            for metric_name in ['accuracy', 'relevance', 'confidence', 'overall']:
                metric_key = f'avg_{metric_name}'
                if metric_key in overall_metrics:
                    current_avg = self.validation_stats['avg_quality_scores'][metric_name]
                    new_value = overall_metrics[metric_key]
                    new_avg = ((current_avg * (total_validations - 1)) + new_value) / total_validations
                    self.validation_stats['avg_quality_scores'][metric_name] = new_avg
                    
        except Exception as e:
            logger.error(f"Validation stats update failed: {e}")
    
    async def get_validation_stats(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        try:
            return {
                'performance_stats': self.validation_stats.copy(),
                'configuration': {
                    'min_accuracy_threshold': self.min_accuracy_threshold,
                    'min_relevance_threshold': self.min_relevance_threshold,
                    'min_confidence_threshold': self.min_confidence_threshold,
                    'enable_caching': self.enable_caching,
                    'cache_ttl': self.cache_ttl
                },
                'validation_levels': [level.value for level in ValidationLevel],
                'issue_types': [issue_type.value for issue_type in ValidationIssueType],
                'system_status': {
                    'initialized': self._initialized,
                    'document_store_available': self.document_store._initialized if hasattr(self.document_store, '_initialized') else False,
                    'embedding_service_available': self.embedding_service._initialized if hasattr(self.embedding_service, '_initialized') else False,
                    'cache_available': self.cache is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Validation stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
citation_validator = CitationValidator()