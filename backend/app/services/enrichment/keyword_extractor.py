"""
Keyword extraction service using multiple algorithms and LLM integration.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter
import re
import math

import trio
import numpy as np

from app.services.nlp.rag_tokenizer import rag_tokenizer
from app.services.nlp.term_weighting import term_weighting_system
from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """
    Multi-algorithm keyword extraction service with LLM integration.
    """
    
    def __init__(self,
                 max_keywords: int = 20,
                 min_keyword_length: int = 3,
                 max_keyword_length: int = 50,
                 min_frequency: int = 2,
                 tfidf_weight: float = 0.4,
                 position_weight: float = 0.3,
                 frequency_weight: float = 0.3):
        """
        Initialize keyword extractor.
        
        Args:
            max_keywords: Maximum number of keywords to extract
            min_keyword_length: Minimum keyword length in characters
            max_keyword_length: Maximum keyword length in characters
            min_frequency: Minimum frequency for keyword consideration
            tfidf_weight: Weight for TF-IDF score
            position_weight: Weight for position score
            frequency_weight: Weight for frequency score
        """
        self.max_keywords = max_keywords
        self.min_keyword_length = min_keyword_length
        self.max_keyword_length = max_keyword_length
        self.min_frequency = min_frequency
        self.tfidf_weight = tfidf_weight
        self.position_weight = position_weight
        self.frequency_weight = frequency_weight
        
        self.tokenizer = rag_tokenizer
        self.term_weighting = term_weighting_system
        self.cache = cache_manager
        
        # Keyword patterns for different languages
        self.keyword_patterns = {
            'russian': re.compile(r'\b[а-яё]{3,}\b', re.IGNORECASE),
            'english': re.compile(r'\b[a-z]{3,}\b', re.IGNORECASE),
            'mixed': re.compile(r'\b[a-zа-яё]{3,}\b', re.IGNORECASE)
        }
        
        # Stop words for keyword filtering
        self.stop_words = set()
        self._initialize_stop_words()
    
    def _initialize_stop_words(self):
        """Initialize stop words for keyword filtering."""
        try:
            # Common stop words that shouldn't be keywords
            russian_stops = {
                'это', 'что', 'как', 'где', 'когда', 'почему', 'который', 'которая', 'которые',
                'также', 'более', 'менее', 'очень', 'может', 'должен', 'можно', 'нужно',
                'есть', 'был', 'была', 'были', 'будет', 'будут', 'имеет', 'имеют'
            }
            
            english_stops = {
                'this', 'that', 'what', 'how', 'where', 'when', 'why', 'which', 'who',
                'also', 'more', 'less', 'very', 'can', 'could', 'should', 'would',
                'have', 'has', 'had', 'will', 'would', 'been', 'being', 'are', 'is'
            }
            
            self.stop_words = russian_stops | english_stops
            
        except Exception as e:
            logger.error(f"Stop words initialization failed: {e}")
    
    async def extract_keywords(self, 
                             text: str,
                             document_id: Optional[str] = None,
                             use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Extract keywords from text using multiple algorithms.
        
        Args:
            text: Input text
            document_id: Optional document ID for caching
            use_cache: Whether to use caching
            
        Returns:
            List of keyword dictionaries with scores and metadata
        """
        if not text or not text.strip():
            return []
        
        try:
            # Initialize services
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            if not self.term_weighting._initialized:
                await self.term_weighting.initialize()
            
            # Check cache
            if use_cache and document_id:
                cache_key = f"keywords:{document_id}"
                cached_keywords = await self.cache.get(cache_key)
                if cached_keywords:
                    logger.debug(f"Using cached keywords for document {document_id}")
                    return cached_keywords
            
            logger.debug(f"Extracting keywords from text ({len(text)} chars)")
            
            # Extract keywords using multiple methods
            keywords = await self._extract_multi_algorithm(text)
            
            # Cache results
            if use_cache and document_id and keywords:
                await self.cache.set(f"keywords:{document_id}", keywords, ttl=86400)  # 24 hours
            
            logger.debug(f"Extracted {len(keywords)} keywords")
            
            return keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    async def _extract_multi_algorithm(self, text: str) -> List[Dict[str, Any]]:
        """Extract keywords using multiple algorithms and combine results."""
        try:
            # Method 1: TF-IDF based extraction
            tfidf_keywords = await self._extract_tfidf_keywords(text)
            
            # Method 2: Statistical keyword extraction
            statistical_keywords = await self._extract_statistical_keywords(text)
            
            # Method 3: N-gram based extraction
            ngram_keywords = await self._extract_ngram_keywords(text)
            
            # Method 4: Position-based extraction
            position_keywords = await self._extract_position_keywords(text)
            
            # Combine and rank all keywords
            combined_keywords = await self._combine_keyword_results([
                tfidf_keywords,
                statistical_keywords,
                ngram_keywords,
                position_keywords
            ])
            
            return combined_keywords
            
        except Exception as e:
            logger.error(f"Multi-algorithm extraction failed: {e}")
            return []
    
    async def _extract_tfidf_keywords(self, text: str) -> List[Dict[str, Any]]:
        """Extract keywords using TF-IDF scores."""
        try:
            # Calculate TF-IDF scores
            tfidf_scores = await self.term_weighting.calculate_tf_idf(text)
            
            if not tfidf_scores:
                return []
            
            keywords = []
            for term, score in tfidf_scores.items():
                if self._is_valid_keyword(term):
                    keywords.append({
                        'keyword': term,
                        'score': score,
                        'method': 'tfidf',
                        'frequency': 1,  # Will be updated later
                        'positions': []
                    })
            
            # Sort by score and take top keywords
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:self.max_keywords]
            
        except Exception as e:
            logger.error(f"TF-IDF keyword extraction failed: {e}")
            return []
    
    async def _extract_statistical_keywords(self, text: str) -> List[Dict[str, Any]]:
        """Extract keywords using statistical analysis."""
        try:
            # Tokenize text
            tokens = await self.tokenizer.tokenize_text(text, remove_stopwords=True, stem_words=False)
            
            if not tokens:
                return []
            
            # Count token frequencies
            token_counts = Counter(tokens)
            
            # Calculate statistical scores
            keywords = []
            total_tokens = len(tokens)
            
            for token, count in token_counts.items():
                if count >= self.min_frequency and self._is_valid_keyword(token):
                    # Calculate statistical score
                    frequency_score = count / total_tokens
                    length_bonus = min(1.0, len(token) / 10)  # Bonus for longer terms
                    
                    score = frequency_score * (1 + length_bonus)
                    
                    keywords.append({
                        'keyword': token,
                        'score': score,
                        'method': 'statistical',
                        'frequency': count,
                        'positions': []
                    })
            
            # Sort by score
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:self.max_keywords]
            
        except Exception as e:
            logger.error(f"Statistical keyword extraction failed: {e}")
            return []
    
    async def _extract_ngram_keywords(self, text: str) -> List[Dict[str, Any]]:
        """Extract keywords using n-gram analysis."""
        try:
            keywords = []
            
            # Extract 2-grams and 3-grams
            for n in [2, 3]:
                ngrams = await self.tokenizer.extract_ngrams(text, n=n, remove_stopwords=True)
                
                if ngrams:
                    ngram_counts = Counter(ngrams)
                    
                    for ngram, count in ngram_counts.items():
                        if count >= self.min_frequency:
                            # Convert back to readable format
                            readable_ngram = ngram.replace('_', ' ')
                            
                            if self._is_valid_keyword(readable_ngram):
                                # Score based on frequency and n-gram length
                                score = count * (n / 2)  # Bonus for longer n-grams
                                
                                keywords.append({
                                    'keyword': readable_ngram,
                                    'score': score,
                                    'method': f'{n}gram',
                                    'frequency': count,
                                    'positions': []
                                })
            
            # Sort by score
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:self.max_keywords]
            
        except Exception as e:
            logger.error(f"N-gram keyword extraction failed: {e}")
            return []
    
    async def _extract_position_keywords(self, text: str) -> List[Dict[str, Any]]:
        """Extract keywords based on position in text (titles, beginnings, etc.)."""
        try:
            keywords = []
            
            # Split text into sentences
            sentences = await self.tokenizer.split_into_sentences(text)
            
            if not sentences:
                return []
            
            # Analyze first few sentences (likely to contain important terms)
            important_sentences = sentences[:min(3, len(sentences))]
            
            for sentence_idx, sentence in enumerate(important_sentences):
                tokens = await self.tokenizer.tokenize_text(sentence, remove_stopwords=True, stem_words=False)
                
                for token_idx, token in enumerate(tokens):
                    if self._is_valid_keyword(token):
                        # Calculate position score
                        sentence_score = 1.0 / (sentence_idx + 1)  # Earlier sentences get higher score
                        token_score = 1.0 / (token_idx + 1)  # Earlier tokens get higher score
                        
                        position_score = (sentence_score + token_score) / 2
                        
                        keywords.append({
                            'keyword': token,
                            'score': position_score,
                            'method': 'position',
                            'frequency': 1,
                            'positions': [{'sentence': sentence_idx, 'token': token_idx}]
                        })
            
            # Sort by score
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:self.max_keywords]
            
        except Exception as e:
            logger.error(f"Position-based keyword extraction failed: {e}")
            return []
    
    async def _combine_keyword_results(self, keyword_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Combine results from multiple keyword extraction methods."""
        try:
            # Collect all keywords
            all_keywords = {}
            
            for keyword_list in keyword_lists:
                for kw in keyword_list:
                    keyword = kw['keyword'].lower()
                    
                    if keyword not in all_keywords:
                        all_keywords[keyword] = {
                            'keyword': kw['keyword'],  # Keep original case
                            'scores': [],
                            'methods': [],
                            'frequencies': [],
                            'positions': []
                        }
                    
                    all_keywords[keyword]['scores'].append(kw['score'])
                    all_keywords[keyword]['methods'].append(kw['method'])
                    all_keywords[keyword]['frequencies'].append(kw.get('frequency', 1))
                    all_keywords[keyword]['positions'].extend(kw.get('positions', []))
            
            # Calculate combined scores
            final_keywords = []
            
            for keyword, data in all_keywords.items():
                # Combine scores from different methods
                combined_score = self._calculate_combined_score(data['scores'], data['methods'])
                
                # Calculate additional metrics
                total_frequency = sum(data['frequencies'])
                method_count = len(set(data['methods']))
                
                final_keywords.append({
                    'keyword': data['keyword'],
                    'score': combined_score,
                    'frequency': total_frequency,
                    'methods': list(set(data['methods'])),
                    'method_count': method_count,
                    'positions': data['positions'],
                    'confidence': min(1.0, method_count / 4)  # Confidence based on method agreement
                })
            
            # Sort by combined score
            final_keywords.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top keywords
            return final_keywords[:self.max_keywords]
            
        except Exception as e:
            logger.error(f"Keyword combination failed: {e}")
            return []
    
    def _calculate_combined_score(self, scores: List[float], methods: List[str]) -> float:
        """Calculate combined score from multiple methods."""
        try:
            if not scores:
                return 0.0
            
            # Weight scores by method type
            method_weights = {
                'tfidf': self.tfidf_weight,
                'statistical': self.frequency_weight,
                'position': self.position_weight,
                '2gram': 0.2,
                '3gram': 0.3
            }
            
            weighted_scores = []
            for score, method in zip(scores, methods):
                weight = method_weights.get(method, 0.1)
                weighted_scores.append(score * weight)
            
            # Calculate weighted average
            if weighted_scores:
                combined_score = sum(weighted_scores) / len(weighted_scores)
                
                # Bonus for multiple method agreement
                method_bonus = 1 + (len(set(methods)) - 1) * 0.1
                
                return combined_score * method_bonus
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Combined score calculation failed: {e}")
            return 0.0
    
    def _is_valid_keyword(self, term: str) -> bool:
        """Check if term is a valid keyword."""
        try:
            if not term or not term.strip():
                return False
            
            term = term.strip().lower()
            
            # Check length
            if len(term) < self.min_keyword_length or len(term) > self.max_keyword_length:
                return False
            
            # Check if it's a stop word
            if term in self.stop_words:
                return False
            
            # Check if it's purely numeric
            if term.isdigit():
                return False
            
            # Check if it contains only special characters
            if not re.search(r'[a-zа-яё]', term, re.IGNORECASE):
                return False
            
            # Check for too many special characters
            special_char_ratio = len(re.findall(r'[^a-zа-яё\s]', term, re.IGNORECASE)) / len(term)
            if special_char_ratio > 0.3:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def extract_keywords_batch(self, 
                                   texts: List[str],
                                   document_ids: Optional[List[str]] = None) -> List[List[Dict[str, Any]]]:
        """Extract keywords from multiple texts in batch."""
        try:
            if not texts:
                return []
            
            results = []
            
            # Process in parallel with limited concurrency
            async with trio.open_nursery() as nursery:
                semaphore = trio.Semaphore(5)  # Limit concurrent extractions
                
                async def extract_single(text: str, doc_id: Optional[str]):
                    async with semaphore:
                        keywords = await self.extract_keywords(text, doc_id)
                        results.append(keywords)
                
                for i, text in enumerate(texts):
                    doc_id = document_ids[i] if document_ids and i < len(document_ids) else None
                    nursery.start_soon(extract_single, text, doc_id)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch keyword extraction failed: {e}")
            return [[] for _ in texts]
    
    async def validate_keywords(self, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate extracted keywords quality."""
        try:
            if not keywords:
                return {
                    'valid': True,
                    'issues': [],
                    'stats': {'total_keywords': 0}
                }
            
            issues = []
            stats = {
                'total_keywords': len(keywords),
                'avg_score': 0.0,
                'avg_frequency': 0.0,
                'method_distribution': Counter(),
                'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
            }
            
            scores = []
            frequencies = []
            
            for kw in keywords:
                scores.append(kw.get('score', 0))
                frequencies.append(kw.get('frequency', 0))
                
                # Count methods
                for method in kw.get('methods', []):
                    stats['method_distribution'][method] += 1
                
                # Count confidence levels
                confidence = kw.get('confidence', 0)
                if confidence > 0.7:
                    stats['confidence_distribution']['high'] += 1
                elif confidence > 0.4:
                    stats['confidence_distribution']['medium'] += 1
                else:
                    stats['confidence_distribution']['low'] += 1
            
            # Calculate averages
            if scores:
                stats['avg_score'] = sum(scores) / len(scores)
            if frequencies:
                stats['avg_frequency'] = sum(frequencies) / len(frequencies)
            
            # Check for issues
            if stats['avg_score'] < 0.1:
                issues.append("Average keyword score is very low")
            
            if stats['confidence_distribution']['low'] > len(keywords) * 0.5:
                issues.append("More than 50% of keywords have low confidence")
            
            if len(stats['method_distribution']) < 2:
                issues.append("Keywords extracted using only one method")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Keyword validation failed: {e}")
            return {'error': str(e)}
    
    async def get_extractor_stats(self) -> Dict[str, Any]:
        """Get keyword extractor statistics."""
        try:
            tokenizer_stats = await self.tokenizer.get_tokenizer_stats()
            term_weighting_stats = await self.term_weighting.get_term_weighting_stats()
            
            return {
                'max_keywords': self.max_keywords,
                'min_keyword_length': self.min_keyword_length,
                'max_keyword_length': self.max_keyword_length,
                'min_frequency': self.min_frequency,
                'weights': {
                    'tfidf': self.tfidf_weight,
                    'position': self.position_weight,
                    'frequency': self.frequency_weight
                },
                'stop_words_count': len(self.stop_words),
                'tokenizer': tokenizer_stats,
                'term_weighting': term_weighting_stats
            }
            
        except Exception as e:
            logger.error(f"Extractor stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
keyword_extractor = KeywordExtractor()