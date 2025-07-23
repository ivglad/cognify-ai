"""
Citation generation system for linking responses to source chunks.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import trio

from app.core.config import settings
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class CitationType(Enum):
    """Types of citations."""
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    REFERENCE = "reference"
    SUPPORTING_EVIDENCE = "supporting_evidence"


@dataclass
class Citation:
    """Citation data structure."""
    id: str
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int]
    sentence_start: int
    sentence_end: int
    cited_text: str
    source_text: str
    confidence: float
    citation_type: CitationType
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CitationGenerator:
    """
    System for generating precise citations linking responses to source chunks.
    """
    
    def __init__(self,
                 min_confidence_threshold: float = 0.7,
                 max_citations_per_sentence: int = 3,
                 enable_caching: bool = True):
        """
        Initialize citation generator.
        
        Args:
            min_confidence_threshold: Minimum confidence for citations
            max_citations_per_sentence: Maximum citations per sentence
            enable_caching: Whether to cache citation results
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.max_citations_per_sentence = max_citations_per_sentence
        self.enable_caching = enable_caching
        
        self.cache = cache_manager
        
        # Performance tracking
        self.citation_stats = {
            'total_citations_generated': 0,
            'avg_confidence': 0.0,
            'citation_types': defaultdict(int),
            'avg_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def generate_citations(self,
                               response_text: str,
                               source_chunks: List[Dict[str, Any]],
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate citations for a response text based on source chunks.
        
        Args:
            response_text: Generated response text
            source_chunks: Source chunks used for generation
            context: Optional context for citation generation
            
        Returns:
            Citations with mapping information
        """
        start_time = time.time()
        
        try:
            if not response_text.strip() or not source_chunks:
                return self._create_empty_citation_result(start_time)
            
            # Check cache
            if self.enable_caching:
                cache_key = self._generate_cache_key(response_text, source_chunks)
                cached_result = await self.cache.get(cache_key)
                
                if cached_result:
                    self.citation_stats['cache_hits'] += 1
                    logger.debug("Using cached citation result")
                    return cached_result
                
                self.citation_stats['cache_misses'] += 1
            
            # Split response into sentences
            sentences = self._split_into_sentences(response_text)
            
            # Generate citations for each sentence
            all_citations = []
            sentence_mappings = []
            
            for i, sentence in enumerate(sentences):
                sentence_citations = await self._generate_sentence_citations(
                    sentence, source_chunks, i, context
                )
                
                all_citations.extend(sentence_citations['citations'])
                sentence_mappings.append({
                    'sentence_index': i,
                    'sentence_text': sentence,
                    'citations': sentence_citations['citations'],
                    'confidence': sentence_citations['avg_confidence']
                })
            
            # Create citation result
            result = {
                'response_text': response_text,
                'total_citations': len(all_citations),
                'citations': all_citations,
                'sentence_mappings': sentence_mappings,
                'avg_confidence': self._calculate_average_confidence(all_citations),
                'citation_coverage': self._calculate_citation_coverage(sentences, all_citations),
                'processing_time': time.time() - start_time,
                'success': True
            }
            
            # Cache result
            if self.enable_caching:
                await self.cache.set(cache_key, result, ttl=1800)  # 30 minutes
            
            # Update stats
            self._update_citation_stats(result)
            
            logger.info(f"Generated {len(all_citations)} citations in {result['processing_time']:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Citation generation failed: {e}")
            return {
                'response_text': response_text,
                'total_citations': 0,
                'citations': [],
                'sentence_mappings': [],
                'avg_confidence': 0.0,
                'citation_coverage': 0.0,
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    async def _generate_sentence_citations(self,
                                         sentence: str,
                                         source_chunks: List[Dict[str, Any]],
                                         sentence_index: int,
                                         context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate citations for a single sentence."""
        try:
            citations = []
            
            # Find potential citations by matching content
            for chunk in source_chunks:
                chunk_citations = await self._find_chunk_citations(
                    sentence, chunk, sentence_index
                )
                citations.extend(chunk_citations)
            
            # Sort by confidence and limit
            citations.sort(key=lambda x: x.confidence, reverse=True)
            citations = citations[:self.max_citations_per_sentence]
            
            # Filter by confidence threshold
            filtered_citations = [
                c for c in citations if c.confidence >= self.min_confidence_threshold
            ]
            
            avg_confidence = (
                sum(c.confidence for c in filtered_citations) / len(filtered_citations)
                if filtered_citations else 0.0
            )
            
            return {
                'citations': filtered_citations,
                'avg_confidence': avg_confidence,
                'total_candidates': len(citations)
            }
            
        except Exception as e:
            logger.error(f"Sentence citation generation failed: {e}")
            return {
                'citations': [],
                'avg_confidence': 0.0,
                'total_candidates': 0
            }
    
    async def _find_chunk_citations(self,
                                  sentence: str,
                                  chunk: Dict[str, Any],
                                  sentence_index: int) -> List[Citation]:
        """Find citations within a specific chunk."""
        try:
            citations = []
            chunk_content = chunk.get('content', '')
            chunk_id = chunk.get('chunk_id', '')
            document_id = chunk.get('document_id', '')
            
            if not chunk_content or not chunk_id:
                return citations
            
            # Find different types of citations
            
            # 1. Direct quotes (exact matches)
            direct_quotes = self._find_direct_quotes(sentence, chunk_content)
            for quote_info in direct_quotes:
                citation = Citation(
                    id=f"cite_{sentence_index}_{chunk_id}_{len(citations)}",
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name=chunk.get('metadata', {}).get('document_name', 'Unknown'),
                    page_number=chunk.get('metadata', {}).get('page_number'),
                    sentence_start=sentence_index,
                    sentence_end=sentence_index,
                    cited_text=quote_info['cited_text'],
                    source_text=quote_info['source_text'],
                    confidence=quote_info['confidence'],
                    citation_type=CitationType.DIRECT_QUOTE,
                    url=chunk.get('url'),
                    metadata=chunk.get('metadata', {})
                )
                citations.append(citation)
            
            # 2. Paraphrases (semantic similarity)
            paraphrases = await self._find_paraphrases(sentence, chunk_content)
            for para_info in paraphrases:
                citation = Citation(
                    id=f"cite_{sentence_index}_{chunk_id}_{len(citations)}",
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name=chunk.get('metadata', {}).get('document_name', 'Unknown'),
                    page_number=chunk.get('metadata', {}).get('page_number'),
                    sentence_start=sentence_index,
                    sentence_end=sentence_index,
                    cited_text=para_info['cited_text'],
                    source_text=para_info['source_text'],
                    confidence=para_info['confidence'],
                    citation_type=CitationType.PARAPHRASE,
                    url=chunk.get('url'),
                    metadata=chunk.get('metadata', {})
                )
                citations.append(citation)
            
            # 3. Supporting evidence (keyword/concept matches)
            evidence = self._find_supporting_evidence(sentence, chunk_content)
            for evidence_info in evidence:
                citation = Citation(
                    id=f"cite_{sentence_index}_{chunk_id}_{len(citations)}",
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name=chunk.get('metadata', {}).get('document_name', 'Unknown'),
                    page_number=chunk.get('metadata', {}).get('page_number'),
                    sentence_start=sentence_index,
                    sentence_end=sentence_index,
                    cited_text=evidence_info['cited_text'],
                    source_text=evidence_info['source_text'],
                    confidence=evidence_info['confidence'],
                    citation_type=CitationType.SUPPORTING_EVIDENCE,
                    url=chunk.get('url'),
                    metadata=chunk.get('metadata', {})
                )
                citations.append(citation)
            
            return citations
            
        except Exception as e:
            logger.error(f"Chunk citation finding failed: {e}")
            return []
    
    def _find_direct_quotes(self, sentence: str, chunk_content: str) -> List[Dict[str, Any]]:
        """Find direct quotes (exact text matches)."""
        try:
            quotes = []
            
            # Split sentence into phrases
            sentence_words = sentence.split()
            
            # Look for exact phrase matches (3+ words)
            for i in range(len(sentence_words) - 2):
                for j in range(i + 3, min(i + 10, len(sentence_words) + 1)):  # Max 10 words
                    phrase = ' '.join(sentence_words[i:j])
                    
                    if phrase.lower() in chunk_content.lower():
                        # Find the exact match in chunk
                        start_idx = chunk_content.lower().find(phrase.lower())
                        if start_idx != -1:
                            end_idx = start_idx + len(phrase)
                            source_text = chunk_content[start_idx:end_idx]
                            
                            # Calculate confidence based on phrase length and exactness
                            confidence = min(0.95, 0.7 + (len(phrase.split()) * 0.05))
                            
                            quotes.append({
                                'cited_text': phrase,
                                'source_text': source_text,
                                'confidence': confidence
                            })
            
            # Remove duplicates and sort by confidence
            unique_quotes = []
            seen_phrases = set()
            
            for quote in sorted(quotes, key=lambda x: x['confidence'], reverse=True):
                if quote['cited_text'] not in seen_phrases:
                    seen_phrases.add(quote['cited_text'])
                    unique_quotes.append(quote)
            
            return unique_quotes[:3]  # Limit to top 3
            
        except Exception as e:
            logger.error(f"Direct quote finding failed: {e}")
            return []
    
    async def _find_paraphrases(self, sentence: str, chunk_content: str) -> List[Dict[str, Any]]:
        """Find paraphrases using semantic similarity."""
        try:
            paraphrases = []
            
            # Split chunk into sentences
            chunk_sentences = self._split_into_sentences(chunk_content)
            
            # Compare sentence with each chunk sentence
            for chunk_sentence in chunk_sentences:
                if len(chunk_sentence.split()) < 3:  # Skip very short sentences
                    continue
                
                # Calculate semantic similarity (simplified)
                similarity = self._calculate_semantic_similarity(sentence, chunk_sentence)
                
                if similarity > 0.6:  # Threshold for paraphrase
                    paraphrases.append({
                        'cited_text': sentence,
                        'source_text': chunk_sentence,
                        'confidence': similarity
                    })
            
            # Sort by similarity and return top matches
            paraphrases.sort(key=lambda x: x['confidence'], reverse=True)
            return paraphrases[:2]  # Limit to top 2
            
        except Exception as e:
            logger.error(f"Paraphrase finding failed: {e}")
            return []
    
    def _find_supporting_evidence(self, sentence: str, chunk_content: str) -> List[Dict[str, Any]]:
        """Find supporting evidence based on keyword/concept matches."""
        try:
            evidence = []
            
            # Extract key terms from sentence
            sentence_terms = self._extract_key_terms(sentence)
            
            if not sentence_terms:
                return evidence
            
            # Split chunk into sentences
            chunk_sentences = self._split_into_sentences(chunk_content)
            
            for chunk_sentence in chunk_sentences:
                if len(chunk_sentence.split()) < 4:  # Skip very short sentences
                    continue
                
                # Count matching terms
                chunk_terms = self._extract_key_terms(chunk_sentence)
                matching_terms = set(sentence_terms) & set(chunk_terms)
                
                if matching_terms:
                    # Calculate confidence based on term overlap
                    confidence = len(matching_terms) / max(len(sentence_terms), 1)
                    confidence = min(0.8, confidence * 0.6)  # Cap at 0.8, scale down
                    
                    if confidence > 0.3:  # Minimum threshold
                        evidence.append({
                            'cited_text': sentence,
                            'source_text': chunk_sentence,
                            'confidence': confidence
                        })
            
            # Sort by confidence and return top matches
            evidence.sort(key=lambda x: x['confidence'], reverse=True)
            return evidence[:2]  # Limit to top 2
            
        except Exception as e:
            logger.error(f"Supporting evidence finding failed: {e}")
            return []    
    d
ef _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            # Simple sentence splitting (can be enhanced with NLTK)
            sentences = re.split(r'[.!?]+', text)
            
            # Clean and filter sentences
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Minimum sentence length
                    cleaned_sentences.append(sentence)
            
            return cleaned_sentences
            
        except Exception as e:
            logger.error(f"Sentence splitting failed: {e}")
            return [text]  # Return original text as single sentence
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        try:
            # Simple key term extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Filter out common stop words
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our',
                'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
                'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'this', 'that', 'with',
                'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when',
                'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them',
                'well', 'were', 'what', 'your'
            }
            
            key_terms = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term in key_terms:
                if term not in seen:
                    seen.add(term)
                    unique_terms.append(term)
            
            return unique_terms[:10]  # Limit to top 10 terms
            
        except Exception as e:
            logger.error(f"Key term extraction failed: {e}")
            return []
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts (simplified)."""
        try:
            # Simple similarity based on word overlap
            words1 = set(self._extract_key_terms(text1))
            words2 = set(self._extract_key_terms(text2))
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            # Jaccard similarity
            jaccard = intersection / union if union > 0 else 0.0
            
            # Boost similarity if sentences have similar structure
            structure_bonus = 0.0
            if abs(len(text1.split()) - len(text2.split())) <= 3:
                structure_bonus = 0.1
            
            return min(1.0, jaccard + structure_bonus)
            
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_average_confidence(self, citations: List[Citation]) -> float:
        """Calculate average confidence of citations."""
        if not citations:
            return 0.0
        
        return sum(c.confidence for c in citations) / len(citations)
    
    def _calculate_citation_coverage(self, sentences: List[str], citations: List[Citation]) -> float:
        """Calculate what percentage of sentences have citations."""
        if not sentences:
            return 0.0
        
        cited_sentences = set()
        for citation in citations:
            cited_sentences.add(citation.sentence_start)
        
        return len(cited_sentences) / len(sentences)
    
    def _generate_cache_key(self, response_text: str, source_chunks: List[Dict[str, Any]]) -> str:
        """Generate cache key for citation results."""
        try:
            chunk_ids = [chunk.get('chunk_id', '') for chunk in source_chunks]
            content_hash = hash(f"{response_text}:{':'.join(chunk_ids)}")
            return f"citations:{content_hash}"
        except Exception:
            return f"citations:{hash(response_text)}"
    
    def _create_empty_citation_result(self, start_time: float) -> Dict[str, Any]:
        """Create empty citation result."""
        return {
            'response_text': '',
            'total_citations': 0,
            'citations': [],
            'sentence_mappings': [],
            'avg_confidence': 0.0,
            'citation_coverage': 0.0,
            'processing_time': time.time() - start_time,
            'success': True
        }
    
    def _update_citation_stats(self, result: Dict[str, Any]):
        """Update citation generation statistics."""
        try:
            citations = result.get('citations', [])
            
            self.citation_stats['total_citations_generated'] += len(citations)
            
            # Update average confidence
            if citations:
                total_confidence = sum(c.confidence for c in citations)
                current_avg = self.citation_stats['avg_confidence']
                current_total = self.citation_stats['total_citations_generated'] - len(citations)
                
                if current_total > 0:
                    new_avg = ((current_avg * current_total) + total_confidence) / self.citation_stats['total_citations_generated']
                else:
                    new_avg = total_confidence / len(citations)
                
                self.citation_stats['avg_confidence'] = new_avg
            
            # Update citation type counts
            for citation in citations:
                self.citation_stats['citation_types'][citation.citation_type.value] += 1
            
            # Update average processing time
            processing_time = result.get('processing_time', 0)
            current_avg_time = self.citation_stats['avg_processing_time']
            
            if self.citation_stats['total_citations_generated'] > len(citations):
                # Not the first batch
                total_batches = (self.citation_stats['total_citations_generated'] // max(len(citations), 1))
                new_avg_time = ((current_avg_time * (total_batches - 1)) + processing_time) / total_batches
                self.citation_stats['avg_processing_time'] = new_avg_time
            else:
                self.citation_stats['avg_processing_time'] = processing_time
                
        except Exception as e:
            logger.error(f"Citation stats update failed: {e}")
    
    async def validate_citations(self,
                               citations: List[Citation],
                               response_text: str,
                               source_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the accuracy and quality of generated citations.
        
        Args:
            citations: List of citations to validate
            response_text: Original response text
            source_chunks: Source chunks used for citation
            
        Returns:
            Validation results with metrics
        """
        try:
            validation_results = {
                'total_citations': len(citations),
                'valid_citations': 0,
                'invalid_citations': 0,
                'accuracy_score': 0.0,
                'completeness_score': 0.0,
                'relevance_score': 0.0,
                'citation_issues': [],
                'validation_details': []
            }
            
            if not citations:
                return validation_results
            
            # Create chunk lookup
            chunk_lookup = {chunk.get('chunk_id', ''): chunk for chunk in source_chunks}
            
            valid_count = 0
            total_accuracy = 0.0
            total_relevance = 0.0
            
            for citation in citations:
                validation_detail = await self._validate_single_citation(
                    citation, chunk_lookup, response_text
                )
                
                validation_results['validation_details'].append(validation_detail)
                
                if validation_detail['is_valid']:
                    valid_count += 1
                    total_accuracy += validation_detail['accuracy_score']
                    total_relevance += validation_detail['relevance_score']
                else:
                    validation_results['citation_issues'].extend(validation_detail['issues'])
            
            validation_results['valid_citations'] = valid_count
            validation_results['invalid_citations'] = len(citations) - valid_count
            
            if valid_count > 0:
                validation_results['accuracy_score'] = total_accuracy / valid_count
                validation_results['relevance_score'] = total_relevance / valid_count
            
            # Calculate completeness (how well citations cover the response)
            validation_results['completeness_score'] = self._calculate_completeness_score(
                citations, response_text
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Citation validation failed: {e}")
            return {
                'total_citations': len(citations),
                'valid_citations': 0,
                'invalid_citations': len(citations),
                'accuracy_score': 0.0,
                'completeness_score': 0.0,
                'relevance_score': 0.0,
                'citation_issues': [f"Validation error: {str(e)}"],
                'validation_details': []
            }
    
    async def _validate_single_citation(self,
                                      citation: Citation,
                                      chunk_lookup: Dict[str, Dict[str, Any]],
                                      response_text: str) -> Dict[str, Any]:
        """Validate a single citation."""
        try:
            validation = {
                'citation_id': citation.id,
                'is_valid': True,
                'accuracy_score': 0.0,
                'relevance_score': 0.0,
                'issues': []
            }
            
            # Check if chunk exists
            chunk = chunk_lookup.get(citation.chunk_id)
            if not chunk:
                validation['is_valid'] = False
                validation['issues'].append(f"Chunk {citation.chunk_id} not found")
                return validation
            
            chunk_content = chunk.get('content', '')
            
            # Validate source text exists in chunk
            if citation.source_text not in chunk_content:
                validation['is_valid'] = False
                validation['issues'].append("Source text not found in chunk")
                return validation
            
            # Calculate accuracy score
            if citation.citation_type == CitationType.DIRECT_QUOTE:
                # For direct quotes, check exact match
                if citation.cited_text.lower() in citation.source_text.lower():
                    validation['accuracy_score'] = 0.95
                else:
                    validation['accuracy_score'] = 0.5
                    validation['issues'].append("Direct quote not exactly matched")
            else:
                # For paraphrases and evidence, use semantic similarity
                similarity = self._calculate_semantic_similarity(
                    citation.cited_text, citation.source_text
                )
                validation['accuracy_score'] = similarity
                
                if similarity < 0.5:
                    validation['issues'].append("Low semantic similarity")
            
            # Calculate relevance score
            validation['relevance_score'] = min(citation.confidence, validation['accuracy_score'])
            
            # Check confidence threshold
            if citation.confidence < self.min_confidence_threshold:
                validation['issues'].append(f"Confidence {citation.confidence:.2f} below threshold")
            
            return validation
            
        except Exception as e:
            logger.error(f"Single citation validation failed: {e}")
            return {
                'citation_id': citation.id,
                'is_valid': False,
                'accuracy_score': 0.0,
                'relevance_score': 0.0,
                'issues': [f"Validation error: {str(e)}"]
            }
    
    def _calculate_completeness_score(self, citations: List[Citation], response_text: str) -> float:
        """Calculate how completely citations cover the response."""
        try:
            sentences = self._split_into_sentences(response_text)
            if not sentences:
                return 0.0
            
            # Count sentences with citations
            cited_sentences = set()
            for citation in citations:
                cited_sentences.add(citation.sentence_start)
            
            return len(cited_sentences) / len(sentences)
            
        except Exception:
            return 0.0
    
    async def format_citations(self,
                             citations: List[Citation],
                             format_style: str = "apa") -> Dict[str, Any]:
        """
        Format citations according to specified style.
        
        Args:
            citations: List of citations to format
            format_style: Citation format style (apa, mla, chicago)
            
        Returns:
            Formatted citations
        """
        try:
            formatted_citations = []
            
            for citation in citations:
                if format_style.lower() == "apa":
                    formatted = self._format_apa_citation(citation)
                elif format_style.lower() == "mla":
                    formatted = self._format_mla_citation(citation)
                elif format_style.lower() == "chicago":
                    formatted = self._format_chicago_citation(citation)
                else:
                    formatted = self._format_simple_citation(citation)
                
                formatted_citations.append({
                    'citation_id': citation.id,
                    'formatted_text': formatted,
                    'citation_type': citation.citation_type.value,
                    'confidence': citation.confidence
                })
            
            return {
                'formatted_citations': formatted_citations,
                'format_style': format_style,
                'total_citations': len(formatted_citations)
            }
            
        except Exception as e:
            logger.error(f"Citation formatting failed: {e}")
            return {
                'formatted_citations': [],
                'format_style': format_style,
                'total_citations': 0,
                'error': str(e)
            }
    
    def _format_apa_citation(self, citation: Citation) -> str:
        """Format citation in APA style."""
        try:
            doc_name = citation.document_name or "Unknown Document"
            page_info = f", p. {citation.page_number}" if citation.page_number else ""
            
            if citation.url:
                return f"{doc_name}{page_info}. Retrieved from {citation.url}"
            else:
                return f"{doc_name}{page_info}"
                
        except Exception:
            return f"Source: {citation.document_name or 'Unknown'}"
    
    def _format_mla_citation(self, citation: Citation) -> str:
        """Format citation in MLA style."""
        try:
            doc_name = citation.document_name or "Unknown Document"
            page_info = f" {citation.page_number}" if citation.page_number else ""
            
            return f"({doc_name}{page_info})"
            
        except Exception:
            return f"({citation.document_name or 'Unknown'})"
    
    def _format_chicago_citation(self, citation: Citation) -> str:
        """Format citation in Chicago style."""
        try:
            doc_name = citation.document_name or "Unknown Document"
            page_info = f", {citation.page_number}" if citation.page_number else ""
            
            return f"{doc_name}{page_info}"
            
        except Exception:
            return f"{citation.document_name or 'Unknown'}"
    
    def _format_simple_citation(self, citation: Citation) -> str:
        """Format citation in simple style."""
        try:
            doc_name = citation.document_name or "Unknown Document"
            page_info = f" (p. {citation.page_number})" if citation.page_number else ""
            
            return f"[{doc_name}{page_info}]"
            
        except Exception:
            return f"[{citation.document_name or 'Unknown'}]"
    
    async def get_citation_stats(self) -> Dict[str, Any]:
        """Get comprehensive citation generation statistics."""
        try:
            return {
                'configuration': {
                    'min_confidence_threshold': self.min_confidence_threshold,
                    'max_citations_per_sentence': self.max_citations_per_sentence,
                    'enable_caching': self.enable_caching
                },
                'performance_stats': dict(self.citation_stats),
                'available_citation_types': [ct.value for ct in CitationType],
                'available_formats': ['apa', 'mla', 'chicago', 'simple']
            }
            
        except Exception as e:
            logger.error(f"Citation stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
citation_generator = CitationGenerator()      cit
ed_sentences = set()
            for citation in citations:
                cited_sentences.add(citation.sentence_start)
            
            return len(cited_sentences) / len(sentences)
            
        except Exception as e:
            logger.error(f"Completeness score calculation failed: {e}")
            return 0.0
    
    async def get_citation_stats(self) -> Dict[str, Any]:
        """Get comprehensive citation generation statistics."""
        try:
            return {
                'performance_stats': self.citation_stats.copy(),
                'configuration': {
                    'min_confidence_threshold': self.min_confidence_threshold,
                    'max_citations_per_sentence': self.max_citations_per_sentence,
                    'enable_caching': self.enable_caching
                },
                'citation_types': [ct.value for ct in CitationType],
                'system_status': {
                    'cache_available': self.cache is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Citation stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
citation_generator = CitationGenerator()