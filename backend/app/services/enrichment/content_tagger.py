"""
Content tagging system with taxonomy management.
"""
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
import re
import json

import trio

from app.services.nlp.rag_tokenizer import rag_tokenizer
from app.services.enrichment.keyword_extractor import keyword_extractor
from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class ContentTagger:
    """
    Content tagging system with hierarchical taxonomy management.
    """
    
    def __init__(self,
                 max_tags: int = 15,
                 min_tag_confidence: float = 0.6,
                 taxonomy_file: Optional[str] = None):
        """
        Initialize content tagger.
        
        Args:
            max_tags: Maximum number of tags to assign
            min_tag_confidence: Minimum confidence threshold for tags
            taxonomy_file: Path to taxonomy configuration file
        """
        self.max_tags = max_tags
        self.min_tag_confidence = min_tag_confidence
        self.taxonomy_file = taxonomy_file
        
        self.tokenizer = rag_tokenizer
        self.keyword_extractor = keyword_extractor
        self.cache = cache_manager
        
        # Taxonomy structure
        self.taxonomy = {}
        self.tag_patterns = {}
        self.tag_synonyms = {}
        self.tag_hierarchy = {}
        
        # Initialize default taxonomy
        self._initialize_default_taxonomy()
    
    def _initialize_default_taxonomy(self):
        """Initialize default taxonomy structure."""
        try:
            # Default taxonomy categories
            self.taxonomy = {
                'technology': {
                    'programming': ['python', 'javascript', 'java', 'c++', 'programming', 'code', 'development'],
                    'ai_ml': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'ai', 'ml'],
                    'web': ['web development', 'html', 'css', 'frontend', 'backend', 'api', 'rest'],
                    'database': ['database', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql'],
                    'cloud': ['cloud computing', 'aws', 'azure', 'docker', 'kubernetes', 'devops']
                },
                'business': {
                    'management': ['management', 'leadership', 'strategy', 'planning', 'organization'],
                    'finance': ['finance', 'accounting', 'budget', 'investment', 'economics'],
                    'marketing': ['marketing', 'advertising', 'branding', 'promotion', 'sales'],
                    'operations': ['operations', 'logistics', 'supply chain', 'process', 'efficiency']
                },
                'science': {
                    'mathematics': ['mathematics', 'statistics', 'algebra', 'calculus', 'geometry'],
                    'physics': ['physics', 'mechanics', 'thermodynamics', 'quantum', 'relativity'],
                    'chemistry': ['chemistry', 'organic', 'inorganic', 'biochemistry', 'molecular'],
                    'biology': ['biology', 'genetics', 'evolution', 'ecology', 'microbiology']
                },
                'education': {
                    'learning': ['education', 'learning', 'teaching', 'training', 'instruction'],
                    'research': ['research', 'study', 'analysis', 'investigation', 'methodology'],
                    'academic': ['academic', 'university', 'school', 'curriculum', 'degree']
                }
            }
            
            # Russian translations
            self.taxonomy.update({
                'технологии': {
                    'программирование': ['программирование', 'код', 'разработка', 'python', 'javascript'],
                    'ии_мо': ['искусственный интеллект', 'машинное обучение', 'нейронная сеть', 'ии', 'мо'],
                    'веб': ['веб разработка', 'html', 'css', 'фронтенд', 'бэкенд', 'api'],
                    'базы_данных': ['база данных', 'sql', 'nosql', 'mongodb', 'postgresql'],
                    'облако': ['облачные вычисления', 'aws', 'azure', 'docker', 'kubernetes']
                },
                'бизнес': {
                    'управление': ['управление', 'менеджмент', 'стратегия', 'планирование'],
                    'финансы': ['финансы', 'бухгалтерия', 'бюджет', 'инвестиции', 'экономика'],
                    'маркетинг': ['маркетинг', 'реклама', 'брендинг', 'продвижение', 'продажи'],
                    'операции': ['операции', 'логистика', 'цепочка поставок', 'процесс']
                }
            })
            
            # Build tag patterns and hierarchy
            self._build_tag_patterns()
            self._build_tag_hierarchy()
            
            logger.info(f"Initialized taxonomy with {len(self.taxonomy)} main categories")
            
        except Exception as e:
            logger.error(f"Default taxonomy initialization failed: {e}")
    
    def _build_tag_patterns(self):
        """Build regex patterns for tag matching."""
        try:
            self.tag_patterns = {}
            
            for main_category, subcategories in self.taxonomy.items():
                for subcategory, keywords in subcategories.items():
                    tag_name = f"{main_category}:{subcategory}"
                    
                    # Create pattern from keywords
                    pattern_parts = []
                    for keyword in keywords:
                        # Escape special regex characters and create word boundary pattern
                        escaped = re.escape(keyword.lower())
                        pattern_parts.append(f"\\b{escaped}\\b")
                    
                    if pattern_parts:
                        pattern = '|'.join(pattern_parts)
                        self.tag_patterns[tag_name] = re.compile(pattern, re.IGNORECASE)
            
            logger.debug(f"Built {len(self.tag_patterns)} tag patterns")
            
        except Exception as e:
            logger.error(f"Tag pattern building failed: {e}")
    
    def _build_tag_hierarchy(self):
        """Build tag hierarchy for parent-child relationships."""
        try:
            self.tag_hierarchy = {}
            
            for main_category, subcategories in self.taxonomy.items():
                # Main category as parent
                self.tag_hierarchy[main_category] = {
                    'parent': None,
                    'children': list(subcategories.keys()),
                    'level': 0
                }
                
                # Subcategories as children
                for subcategory in subcategories.keys():
                    full_tag = f"{main_category}:{subcategory}"
                    self.tag_hierarchy[full_tag] = {
                        'parent': main_category,
                        'children': [],
                        'level': 1
                    }
            
            logger.debug(f"Built hierarchy for {len(self.tag_hierarchy)} tags")
            
        except Exception as e:
            logger.error(f"Tag hierarchy building failed: {e}")
    
    async def tag_content(self, 
                         text: str,
                         document_id: Optional[str] = None,
                         use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Tag content with relevant categories and topics.
        
        Args:
            text: Input text to tag
            document_id: Optional document ID for caching
            use_cache: Whether to use caching
            
        Returns:
            List of tags with confidence scores and metadata
        """
        if not text or not text.strip():
            return []
        
        try:
            # Initialize services
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            # Check cache
            if use_cache and document_id:
                cache_key = f"tags:{document_id}"
                cached_tags = await self.cache.get(cache_key)
                if cached_tags:
                    logger.debug(f"Using cached tags for document {document_id}")
                    return cached_tags
            
            logger.debug(f"Tagging content ({len(text)} chars)")
            
            # Extract tags using multiple methods
            tags = await self._extract_multi_method_tags(text)
            
            # Validate and rank tags
            validated_tags = await self._validate_and_rank_tags(tags, text)
            
            # Cache results
            if use_cache and document_id and validated_tags:
                await self.cache.set(f"tags:{document_id}", validated_tags, ttl=86400)  # 24 hours
            
            logger.debug(f"Generated {len(validated_tags)} validated tags")
            
            return validated_tags
            
        except Exception as e:
            logger.error(f"Content tagging failed: {e}")
            return []
    
    async def _extract_multi_method_tags(self, text: str) -> List[Dict[str, Any]]:
        """Extract tags using multiple methods."""
        try:
            # Method 1: Pattern-based tagging
            pattern_tags = await self._extract_pattern_tags(text)
            
            # Method 2: Keyword-based tagging
            keyword_tags = await self._extract_keyword_tags(text)
            
            # Method 3: Statistical tagging
            statistical_tags = await self._extract_statistical_tags(text)
            
            # Method 4: Context-based tagging
            context_tags = await self._extract_context_tags(text)
            
            # Combine all tags
            all_tags = []
            all_tags.extend(pattern_tags)
            all_tags.extend(keyword_tags)
            all_tags.extend(statistical_tags)
            all_tags.extend(context_tags)
            
            # Merge duplicate tags
            merged_tags = await self._merge_duplicate_tags(all_tags)
            
            return merged_tags
            
        except Exception as e:
            logger.error(f"Multi-method tag extraction failed: {e}")
            return []
    
    async def _extract_pattern_tags(self, text: str) -> List[Dict[str, Any]]:
        """Extract tags using predefined patterns."""
        try:
            tags = []
            text_lower = text.lower()
            
            for tag_name, pattern in self.tag_patterns.items():
                matches = pattern.findall(text_lower)
                
                if matches:
                    # Calculate confidence based on match frequency and context
                    match_count = len(matches)
                    confidence = min(1.0, match_count / 10)  # Normalize by frequency
                    
                    # Boost confidence for multiple matches
                    if match_count > 1:
                        confidence += 0.1
                    
                    tags.append({
                        'tag': tag_name,
                        'confidence': confidence,
                        'method': 'pattern',
                        'matches': matches,
                        'match_count': match_count,
                        'category': tag_name.split(':')[0],
                        'subcategory': tag_name.split(':')[1] if ':' in tag_name else None
                    })
            
            return tags
            
        except Exception as e:
            logger.error(f"Pattern-based tag extraction failed: {e}")
            return []
    
    async def _extract_keyword_tags(self, text: str) -> List[Dict[str, Any]]:
        """Extract tags based on extracted keywords."""
        try:
            # Get keywords from text
            keywords = await self.keyword_extractor.extract_keywords(text)
            
            if not keywords:
                return []
            
            tags = []
            
            # Match keywords to taxonomy
            for kw in keywords:
                keyword_text = kw['keyword'].lower()
                
                # Find matching tags
                for tag_name, pattern in self.tag_patterns.items():
                    if pattern.search(keyword_text):
                        # Calculate confidence based on keyword score
                        base_confidence = kw.get('score', 0)
                        keyword_confidence = min(1.0, base_confidence * 2)  # Boost keyword-based confidence
                        
                        tags.append({
                            'tag': tag_name,
                            'confidence': keyword_confidence,
                            'method': 'keyword',
                            'source_keyword': kw['keyword'],
                            'keyword_score': base_confidence,
                            'category': tag_name.split(':')[0],
                            'subcategory': tag_name.split(':')[1] if ':' in tag_name else None
                        })
            
            return tags
            
        except Exception as e:
            logger.error(f"Keyword-based tag extraction failed: {e}")
            return []
    
    async def _extract_statistical_tags(self, text: str) -> List[Dict[str, Any]]:
        """Extract tags using statistical analysis."""
        try:
            tags = []
            
            # Tokenize text
            tokens = await self.tokenizer.tokenize_text(text, remove_stopwords=True, stem_words=False)
            
            if not tokens:
                return []
            
            # Count token frequencies
            token_counts = Counter(tokens)
            total_tokens = len(tokens)
            
            # Check each token against taxonomy
            for token, count in token_counts.items():
                if count >= 2:  # Minimum frequency
                    # Find matching tags
                    for tag_name, pattern in self.tag_patterns.items():
                        if pattern.search(token.lower()):
                            # Calculate statistical confidence
                            frequency_score = count / total_tokens
                            statistical_confidence = min(1.0, frequency_score * 20)  # Scale frequency
                            
                            tags.append({
                                'tag': tag_name,
                                'confidence': statistical_confidence,
                                'method': 'statistical',
                                'source_token': token,
                                'frequency': count,
                                'frequency_score': frequency_score,
                                'category': tag_name.split(':')[0],
                                'subcategory': tag_name.split(':')[1] if ':' in tag_name else None
                            })
            
            return tags
            
        except Exception as e:
            logger.error(f"Statistical tag extraction failed: {e}")
            return []
    
    async def _extract_context_tags(self, text: str) -> List[Dict[str, Any]]:
        """Extract tags based on context and co-occurrence."""
        try:
            tags = []
            
            # Split text into sentences for context analysis
            sentences = await self.tokenizer.split_into_sentences(text)
            
            if not sentences:
                return []
            
            # Analyze each sentence for tag co-occurrence
            for sentence in sentences:
                sentence_lower = sentence.lower()
                sentence_tags = []
                
                # Find all tags in this sentence
                for tag_name, pattern in self.tag_patterns.items():
                    if pattern.search(sentence_lower):
                        sentence_tags.append(tag_name)
                
                # If multiple tags in same sentence, boost their confidence
                if len(sentence_tags) > 1:
                    for tag_name in sentence_tags:
                        context_confidence = 0.3 + (len(sentence_tags) - 1) * 0.1
                        
                        tags.append({
                            'tag': tag_name,
                            'confidence': context_confidence,
                            'method': 'context',
                            'context_sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                            'co_occurring_tags': [t for t in sentence_tags if t != tag_name],
                            'category': tag_name.split(':')[0],
                            'subcategory': tag_name.split(':')[1] if ':' in tag_name else None
                        })
            
            return tags
            
        except Exception as e:
            logger.error(f"Context-based tag extraction failed: {e}")
            return []
    
    async def _merge_duplicate_tags(self, tags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge duplicate tags and combine their confidence scores."""
        try:
            if not tags:
                return []
            
            # Group tags by tag name
            tag_groups = defaultdict(list)
            for tag in tags:
                tag_groups[tag['tag']].append(tag)
            
            merged_tags = []
            
            for tag_name, tag_list in tag_groups.items():
                # Combine confidence scores
                confidences = [t['confidence'] for t in tag_list]
                methods = [t['method'] for t in tag_list]
                
                # Calculate combined confidence
                combined_confidence = self._combine_confidences(confidences, methods)
                
                # Collect all metadata
                all_metadata = {}
                for tag in tag_list:
                    for key, value in tag.items():
                        if key not in ['tag', 'confidence', 'method']:
                            if key not in all_metadata:
                                all_metadata[key] = []
                            if isinstance(value, list):
                                all_metadata[key].extend(value)
                            else:
                                all_metadata[key].append(value)
                
                # Create merged tag
                merged_tag = {
                    'tag': tag_name,
                    'confidence': combined_confidence,
                    'methods': list(set(methods)),
                    'method_count': len(set(methods)),
                    'category': tag_name.split(':')[0],
                    'subcategory': tag_name.split(':')[1] if ':' in tag_name else None
                }
                
                # Add metadata
                merged_tag.update(all_metadata)
                
                merged_tags.append(merged_tag)
            
            return merged_tags
            
        except Exception as e:
            logger.error(f"Tag merging failed: {e}")
            return tags
    
    def _combine_confidences(self, confidences: List[float], methods: List[str]) -> float:
        """Combine confidence scores from multiple methods."""
        try:
            if not confidences:
                return 0.0
            
            # Weight different methods
            method_weights = {
                'pattern': 0.4,
                'keyword': 0.3,
                'statistical': 0.2,
                'context': 0.1
            }
            
            # Calculate weighted average
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for confidence, method in zip(confidences, methods):
                weight = method_weights.get(method, 0.1)
                weighted_sum += confidence * weight
                weight_sum += weight
            
            if weight_sum > 0:
                base_confidence = weighted_sum / weight_sum
            else:
                base_confidence = sum(confidences) / len(confidences)
            
            # Bonus for multiple method agreement
            method_bonus = 1 + (len(set(methods)) - 1) * 0.1
            
            return min(1.0, base_confidence * method_bonus)
            
        except Exception as e:
            logger.error(f"Confidence combination failed: {e}")
            return sum(confidences) / len(confidences) if confidences else 0.0
    
    async def _validate_and_rank_tags(self, 
                                    tags: List[Dict[str, Any]], 
                                    text: str) -> List[Dict[str, Any]]:
        """Validate and rank tags by relevance and confidence."""
        try:
            if not tags:
                return []
            
            validated_tags = []
            
            for tag in tags:
                # Calculate final confidence score
                final_confidence = await self._calculate_final_confidence(tag, text)
                tag['final_confidence'] = final_confidence
                
                # Filter by minimum confidence
                if final_confidence >= self.min_tag_confidence:
                    validated_tags.append(tag)
            
            # Sort by confidence and return top tags
            validated_tags.sort(key=lambda x: x['final_confidence'], reverse=True)
            
            # Apply hierarchy rules (prefer more specific tags)
            hierarchical_tags = self._apply_hierarchy_rules(validated_tags)
            
            return hierarchical_tags[:self.max_tags]
            
        except Exception as e:
            logger.error(f"Tag validation failed: {e}")
            return tags[:self.max_tags]
    
    async def _calculate_final_confidence(self, 
                                        tag: Dict[str, Any], 
                                        text: str) -> float:
        """Calculate final confidence score for a tag."""
        try:
            base_confidence = tag.get('confidence', 0.5)
            
            # Method diversity bonus
            method_count = tag.get('method_count', 1)
            method_bonus = min(0.2, (method_count - 1) * 0.05)
            
            # Hierarchy bonus (more specific tags get bonus)
            hierarchy_bonus = 0.0
            if tag.get('subcategory'):
                hierarchy_bonus = 0.1
            
            # Text relevance
            relevance_score = await self._calculate_tag_relevance(tag, text)
            
            # Combine scores
            final_confidence = (base_confidence * 0.6 + 
                              relevance_score * 0.3 + 
                              method_bonus + 
                              hierarchy_bonus)
            
            return min(1.0, final_confidence)
            
        except Exception as e:
            logger.error(f"Final confidence calculation failed: {e}")
            return tag.get('confidence', 0.5)
    
    async def _calculate_tag_relevance(self, 
                                     tag: Dict[str, Any], 
                                     text: str) -> float:
        """Calculate relevance of tag to text content."""
        try:
            tag_name = tag['tag']
            text_lower = text.lower()
            
            # Get keywords for this tag from taxonomy
            category = tag.get('category', '')
            subcategory = tag.get('subcategory', '')
            
            tag_keywords = []
            if category in self.taxonomy and subcategory in self.taxonomy[category]:
                tag_keywords = self.taxonomy[category][subcategory]
            
            if not tag_keywords:
                return 0.5
            
            # Count occurrences of tag keywords in text
            total_occurrences = 0
            for keyword in tag_keywords:
                occurrences = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))
                total_occurrences += occurrences
            
            # Normalize by text length
            text_words = len(text.split())
            relevance = min(1.0, total_occurrences / max(1, text_words / 100))
            
            return relevance
            
        except Exception as e:
            logger.error(f"Tag relevance calculation failed: {e}")
            return 0.5
    
    def _apply_hierarchy_rules(self, tags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply hierarchy rules to prefer more specific tags."""
        try:
            # Group tags by category
            category_groups = defaultdict(list)
            for tag in tags:
                category = tag.get('category', '')
                category_groups[category].append(tag)
            
            final_tags = []
            
            for category, category_tags in category_groups.items():
                # Sort by specificity (subcategory tags are more specific)
                category_tags.sort(key=lambda x: (
                    x.get('subcategory') is not None,  # Subcategory tags first
                    x['final_confidence']
                ), reverse=True)
                
                # Take best tags from each category
                max_per_category = max(1, self.max_tags // len(category_groups))
                final_tags.extend(category_tags[:max_per_category])
            
            # Sort final list by confidence
            final_tags.sort(key=lambda x: x['final_confidence'], reverse=True)
            
            return final_tags
            
        except Exception as e:
            logger.error(f"Hierarchy rules application failed: {e}")
            return tags
    
    async def get_tag_suggestions(self, 
                                partial_tag: str, 
                                limit: int = 10) -> List[Dict[str, Any]]:
        """Get tag suggestions based on partial input."""
        try:
            suggestions = []
            partial_lower = partial_tag.lower()
            
            # Search in taxonomy
            for main_category, subcategories in self.taxonomy.items():
                # Check main category
                if partial_lower in main_category.lower():
                    suggestions.append({
                        'tag': main_category,
                        'type': 'category',
                        'full_path': main_category,
                        'match_type': 'category_name'
                    })
                
                # Check subcategories
                for subcategory, keywords in subcategories.items():
                    full_tag = f"{main_category}:{subcategory}"
                    
                    if partial_lower in subcategory.lower():
                        suggestions.append({
                            'tag': full_tag,
                            'type': 'subcategory',
                            'full_path': full_tag,
                            'match_type': 'subcategory_name'
                        })
                    
                    # Check keywords
                    for keyword in keywords:
                        if partial_lower in keyword.lower():
                            suggestions.append({
                                'tag': full_tag,
                                'type': 'keyword_match',
                                'full_path': full_tag,
                                'match_type': 'keyword',
                                'matched_keyword': keyword
                            })
            
            # Remove duplicates and sort
            unique_suggestions = []
            seen_tags = set()
            
            for suggestion in suggestions:
                if suggestion['tag'] not in seen_tags:
                    seen_tags.add(suggestion['tag'])
                    unique_suggestions.append(suggestion)
            
            return unique_suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Tag suggestions failed: {e}")
            return []
    
    async def get_tag_hierarchy(self, tag: str) -> Dict[str, Any]:
        """Get hierarchy information for a tag."""
        try:
            if tag not in self.tag_hierarchy:
                return {'error': f'Tag {tag} not found in hierarchy'}
            
            hierarchy_info = self.tag_hierarchy[tag].copy()
            
            # Add children details
            if hierarchy_info['children']:
                children_details = []
                for child in hierarchy_info['children']:
                    child_tag = f"{tag}:{child}" if ':' not in child else child
                    if child_tag in self.tag_hierarchy:
                        children_details.append({
                            'tag': child_tag,
                            'name': child,
                            'level': self.tag_hierarchy[child_tag]['level']
                        })
                hierarchy_info['children_details'] = children_details
            
            # Add parent details
            if hierarchy_info['parent']:
                parent_tag = hierarchy_info['parent']
                if parent_tag in self.tag_hierarchy:
                    hierarchy_info['parent_details'] = {
                        'tag': parent_tag,
                        'name': parent_tag,
                        'level': self.tag_hierarchy[parent_tag]['level']
                    }
            
            return hierarchy_info
            
        except Exception as e:
            logger.error(f"Tag hierarchy retrieval failed: {e}")
            return {'error': str(e)}
    
    async def get_tagger_stats(self) -> Dict[str, Any]:
        """Get content tagger statistics."""
        try:
            return {
                'max_tags': self.max_tags,
                'min_tag_confidence': self.min_tag_confidence,
                'taxonomy_categories': len(self.taxonomy),
                'total_patterns': len(self.tag_patterns),
                'hierarchy_nodes': len(self.tag_hierarchy),
                'tokenizer_initialized': self.tokenizer._initialized if hasattr(self.tokenizer, '_initialized') else False
            }
            
        except Exception as e:
            logger.error(f"Tagger stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
content_tagger = ContentTagger()