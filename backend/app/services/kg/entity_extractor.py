"""
Entity extraction system for knowledge graph construction.
"""
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import re
from collections import Counter, defaultdict

import trio
import spacy
from spacy import displacy

from app.services.nlp.rag_tokenizer import rag_tokenizer
from app.services.enrichment.keyword_extractor import keyword_extractor
from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Entity extraction system using spaCy NLP and custom patterns.
    """
    
    def __init__(self,
                 min_entity_confidence: float = 0.6,
                 max_entities_per_text: int = 50,
                 enable_custom_patterns: bool = True):
        """
        Initialize entity extractor.
        
        Args:
            min_entity_confidence: Minimum confidence threshold for entities
            max_entities_per_text: Maximum entities to extract per text
            enable_custom_patterns: Whether to use custom entity patterns
        """
        self.min_entity_confidence = min_entity_confidence
        self.max_entities_per_text = max_entities_per_text
        self.enable_custom_patterns = enable_custom_patterns
        
        self.tokenizer = rag_tokenizer
        self.keyword_extractor = keyword_extractor
        self.cache = cache_manager
        
        # spaCy models
        self.nlp_en = None
        self.nlp_ru = None
        
        # Entity type mappings
        self.entity_type_mapping = {
            # spaCy English types
            'PERSON': 'Person',
            'ORG': 'Organization', 
            'GPE': 'Location',
            'LOC': 'Location',
            'PRODUCT': 'Product',
            'EVENT': 'Event',
            'WORK_OF_ART': 'CreativeWork',
            'LAW': 'Law',
            'LANGUAGE': 'Language',
            'DATE': 'Date',
            'TIME': 'Time',
            'PERCENT': 'Percentage',
            'MONEY': 'Money',
            'QUANTITY': 'Quantity',
            'ORDINAL': 'Number',
            'CARDINAL': 'Number',
            
            # Custom types
            'TECHNOLOGY': 'Technology',
            'CONCEPT': 'Concept',
            'METHOD': 'Method',
            'TOOL': 'Tool',
            'FRAMEWORK': 'Framework',
            'ALGORITHM': 'Algorithm'
        }
        
        # Custom entity patterns
        self.custom_patterns = {
            'TECHNOLOGY': [
                re.compile(r'\b(?:Python|JavaScript|Java|C\+\+|React|Vue|Angular|Django|Flask)\b', re.IGNORECASE),
                re.compile(r'\b(?:машинное обучение|искусственный интеллект|нейронная сеть)\b', re.IGNORECASE),
                re.compile(r'\b(?:machine learning|artificial intelligence|neural network|deep learning)\b', re.IGNORECASE)
            ],
            'FRAMEWORK': [
                re.compile(r'\b(?:TensorFlow|PyTorch|Keras|Scikit-learn|OpenCV)\b', re.IGNORECASE),
                re.compile(r'\b(?:Spring|Hibernate|Express|Laravel|Rails)\b', re.IGNORECASE)
            ],
            'CONCEPT': [
                re.compile(r'\b(?:алгоритм|структура данных|база данных|архитектура)\b', re.IGNORECASE),
                re.compile(r'\b(?:algorithm|data structure|database|architecture|design pattern)\b', re.IGNORECASE)
            ]
        }
        
        # Initialize spaCy models
        self._initialize_spacy_models()
    
    def _initialize_spacy_models(self):
        """Initialize spaCy NLP models."""
        try:
            # Try to load English model
            try:
                self.nlp_en = spacy.load("en_core_web_sm")
                logger.info("Loaded English spaCy model")
            except OSError:
                logger.warning("English spaCy model not found, using basic tokenizer")
                self.nlp_en = None
            
            # Try to load Russian model
            try:
                self.nlp_ru = spacy.load("ru_core_news_sm")
                logger.info("Loaded Russian spaCy model")
            except OSError:
                logger.warning("Russian spaCy model not found, using basic patterns")
                self.nlp_ru = None
            
            # If no models available, log warning
            if not self.nlp_en and not self.nlp_ru:
                logger.warning("No spaCy models available, using pattern-based extraction only")
                
        except Exception as e:
            logger.error(f"spaCy model initialization failed: {e}")
    
    async def extract_entities(self, 
                             text: str,
                             document_id: Optional[str] = None,
                             use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            document_id: Optional document ID for caching
            use_cache: Whether to use caching
            
        Returns:
            List of extracted entities with metadata
        """
        if not text or not text.strip():
            return []
        
        try:
            # Initialize services
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            # Check cache
            if use_cache and document_id:
                cache_key = f"entities:{document_id}"
                cached_entities = await self.cache.get(cache_key)
                if cached_entities:
                    logger.debug(f"Using cached entities for document {document_id}")
                    return cached_entities
            
            logger.debug(f"Extracting entities from text ({len(text)} chars)")
            
            # Extract entities using multiple methods
            entities = await self._extract_multi_method_entities(text)
            
            # Validate and rank entities
            validated_entities = await self._validate_and_rank_entities(entities, text)
            
            # Cache results
            if use_cache and document_id and validated_entities:
                await self.cache.set(f"entities:{document_id}", validated_entities, ttl=86400)  # 24 hours
            
            logger.debug(f"Extracted {len(validated_entities)} validated entities")
            
            return validated_entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def _extract_multi_method_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using multiple methods."""
        try:
            # Method 1: spaCy NER
            spacy_entities = await self._extract_spacy_entities(text)
            
            # Method 2: Custom pattern matching
            pattern_entities = await self._extract_pattern_entities(text)
            
            # Method 3: Keyword-based entities
            keyword_entities = await self._extract_keyword_entities(text)
            
            # Method 4: Statistical entities
            statistical_entities = await self._extract_statistical_entities(text)
            
            # Combine all entities
            all_entities = []
            all_entities.extend(spacy_entities)
            all_entities.extend(pattern_entities)
            all_entities.extend(keyword_entities)
            all_entities.extend(statistical_entities)
            
            # Merge duplicate entities
            merged_entities = await self._merge_duplicate_entities(all_entities)
            
            return merged_entities
            
        except Exception as e:
            logger.error(f"Multi-method entity extraction failed: {e}")
            return []
    
    async def _extract_spacy_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NER."""
        try:
            entities = []
            
            # Detect language
            language = await self._detect_language(text)
            
            # Choose appropriate model
            nlp_model = None
            if language == 'russian' and self.nlp_ru:
                nlp_model = self.nlp_ru
            elif language == 'english' and self.nlp_en:
                nlp_model = self.nlp_en
            elif self.nlp_en:  # Fallback to English
                nlp_model = self.nlp_en
            
            if not nlp_model:
                logger.debug("No spaCy model available for entity extraction")
                return []
            
            # Process text with spaCy
            doc = await trio.to_thread.run_sync(nlp_model, text)
            
            # Extract entities
            for ent in doc.ents:
                entity_type = self.entity_type_mapping.get(ent.label_, ent.label_)
                
                entities.append({
                    'text': ent.text.strip(),
                    'type': entity_type,
                    'spacy_label': ent.label_,
                    'start_pos': ent.start_char,
                    'end_pos': ent.end_char,
                    'confidence': self._calculate_spacy_confidence(ent),
                    'method': 'spacy',
                    'language': language,
                    'context': self._get_entity_context(text, ent.start_char, ent.end_char)
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"spaCy entity extraction failed: {e}")
            return []
    
    async def _extract_pattern_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using custom patterns."""
        try:
            if not self.enable_custom_patterns:
                return []
            
            entities = []
            
            for entity_type, patterns in self.custom_patterns.items():
                for pattern in patterns:
                    matches = pattern.finditer(text)
                    
                    for match in matches:
                        entity_text = match.group().strip()
                        
                        if len(entity_text) >= 2:  # Minimum entity length
                            entities.append({
                                'text': entity_text,
                                'type': self.entity_type_mapping.get(entity_type, entity_type),
                                'pattern_type': entity_type,
                                'start_pos': match.start(),
                                'end_pos': match.end(),
                                'confidence': 0.8,  # High confidence for pattern matches
                                'method': 'pattern',
                                'context': self._get_entity_context(text, match.start(), match.end())
                            })
            
            return entities
            
        except Exception as e:
            logger.error(f"Pattern entity extraction failed: {e}")
            return []
    
    async def _extract_keyword_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities based on extracted keywords."""
        try:
            # Get keywords
            keywords = await self.keyword_extractor.extract_keywords(text)
            
            if not keywords:
                return []
            
            entities = []
            
            for kw in keywords:
                keyword_text = kw['keyword']
                
                # Classify keyword as potential entity
                entity_type = await self._classify_keyword_as_entity(keyword_text, text)
                
                if entity_type:
                    # Find positions of keyword in text
                    positions = self._find_keyword_positions(keyword_text, text)
                    
                    for start_pos, end_pos in positions:
                        entities.append({
                            'text': keyword_text,
                            'type': entity_type,
                            'start_pos': start_pos,
                            'end_pos': end_pos,
                            'confidence': min(0.9, kw.get('score', 0.5) + 0.3),
                            'method': 'keyword',
                            'keyword_score': kw.get('score', 0),
                            'context': self._get_entity_context(text, start_pos, end_pos)
                        })
            
            return entities
            
        except Exception as e:
            logger.error(f"Keyword entity extraction failed: {e}")
            return []
    
    async def _extract_statistical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using statistical analysis."""
        try:
            entities = []
            
            # Find capitalized words (potential proper nouns)
            capitalized_pattern = re.compile(r'\b[A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)*\b')
            matches = capitalized_pattern.finditer(text)
            
            # Count frequencies
            capitalized_words = {}
            for match in matches:
                word = match.group().strip()
                if len(word) >= 3:  # Minimum length
                    if word not in capitalized_words:
                        capitalized_words[word] = []
                    capitalized_words[word].append((match.start(), match.end()))
            
            # Filter by frequency and context
            for word, positions in capitalized_words.items():
                if len(positions) >= 2 or self._looks_like_entity(word):
                    # Classify entity type
                    entity_type = await self._classify_statistical_entity(word, text)
                    
                    for start_pos, end_pos in positions:
                        entities.append({
                            'text': word,
                            'type': entity_type,
                            'start_pos': start_pos,
                            'end_pos': end_pos,
                            'confidence': min(0.7, 0.4 + len(positions) * 0.1),
                            'method': 'statistical',
                            'frequency': len(positions),
                            'context': self._get_entity_context(text, start_pos, end_pos)
                        })
            
            return entities
            
        except Exception as e:
            logger.error(f"Statistical entity extraction failed: {e}")
            return []
    
    def _calculate_spacy_confidence(self, ent) -> float:
        """Calculate confidence score for spaCy entity."""
        try:
            # Base confidence based on entity label
            label_confidences = {
                'PERSON': 0.9,
                'ORG': 0.8,
                'GPE': 0.8,
                'LOC': 0.7,
                'PRODUCT': 0.6,
                'EVENT': 0.6,
                'DATE': 0.9,
                'TIME': 0.9,
                'MONEY': 0.9,
                'PERCENT': 0.9
            }
            
            base_confidence = label_confidences.get(ent.label_, 0.5)
            
            # Length bonus
            length_bonus = min(0.2, len(ent.text) / 50)
            
            return min(1.0, base_confidence + length_bonus)
            
        except Exception:
            return 0.5
    
    def _get_entity_context(self, text: str, start_pos: int, end_pos: int, context_size: int = 50) -> str:
        """Get context around entity."""
        try:
            context_start = max(0, start_pos - context_size)
            context_end = min(len(text), end_pos + context_size)
            
            context = text[context_start:context_end]
            
            # Mark the entity in context
            entity_in_context = text[start_pos:end_pos]
            context = context.replace(entity_in_context, f"**{entity_in_context}**", 1)
            
            return context.strip()
            
        except Exception:
            return ""
    
    async def _classify_keyword_as_entity(self, keyword: str, text: str) -> Optional[str]:
        """Classify keyword as potential entity type."""
        try:
            keyword_lower = keyword.lower()
            
            # Technology patterns
            tech_indicators = ['api', 'framework', 'library', 'algorithm', 'method', 'protocol', 'standard']
            if any(indicator in keyword_lower for indicator in tech_indicators):
                return 'Technology'
            
            # Check if it's a proper noun (capitalized)
            if keyword[0].isupper() and len(keyword) > 3:
                # Could be organization, person, or location
                if any(word in keyword_lower for word in ['inc', 'corp', 'ltd', 'company', 'group']):
                    return 'Organization'
                elif any(word in keyword_lower for word in ['university', 'institute', 'school', 'college']):
                    return 'Organization'
                else:
                    return 'Concept'  # Generic concept
            
            # Concept indicators
            concept_indicators = ['principle', 'theory', 'model', 'approach', 'technique', 'strategy']
            if any(indicator in keyword_lower for indicator in concept_indicators):
                return 'Concept'
            
            return None
            
        except Exception:
            return None
    
    def _find_keyword_positions(self, keyword: str, text: str) -> List[Tuple[int, int]]:
        """Find all positions of keyword in text."""
        try:
            positions = []
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            
            for match in pattern.finditer(text):
                positions.append((match.start(), match.end()))
            
            return positions
            
        except Exception:
            return []
    
    def _looks_like_entity(self, word: str) -> bool:
        """Check if word looks like an entity."""
        try:
            # Check if it's a proper noun pattern
            if not word[0].isupper():
                return False
            
            # Skip common words
            common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But'}
            if word in common_words:
                return False
            
            # Check length
            if len(word) < 3:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _classify_statistical_entity(self, word: str, text: str) -> str:
        """Classify statistically found entity."""
        try:
            word_lower = word.lower()
            
            # Check context for clues
            context_patterns = {
                'Person': [r'\b(?:Mr|Mrs|Dr|Prof)\s+' + re.escape(word), 
                          re.escape(word) + r'\s+(?:said|wrote|developed|created)'],
                'Organization': [re.escape(word) + r'\s+(?:Inc|Corp|Ltd|Company|Group)',
                               r'(?:at|from|with)\s+' + re.escape(word)],
                'Location': [r'(?:in|at|from|to)\s+' + re.escape(word),
                           re.escape(word) + r'\s+(?:city|country|region)']
            }
            
            for entity_type, patterns in context_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        return entity_type
            
            # Default classification
            return 'Concept'
            
        except Exception:
            return 'Concept'
    
    async def _detect_language(self, text: str) -> str:
        """Detect language of text."""
        try:
            # Simple language detection based on character sets
            cyrillic_count = len(re.findall(r'[а-яёА-ЯЁ]', text))
            latin_count = len(re.findall(r'[a-zA-Z]', text))
            
            if cyrillic_count > latin_count:
                return 'russian'
            else:
                return 'english'
                
        except Exception:
            return 'english'
    
    async def _merge_duplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge duplicate entities."""
        try:
            if not entities:
                return []
            
            # Group entities by text (case-insensitive)
            entity_groups = defaultdict(list)
            for entity in entities:
                key = entity['text'].lower().strip()
                entity_groups[key].append(entity)
            
            merged_entities = []
            
            for entity_text, entity_list in entity_groups.items():
                if not entity_list:
                    continue
                
                # Sort by confidence
                entity_list.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                best_entity = entity_list[0]
                
                # Combine information from all instances
                all_methods = set()
                all_positions = []
                total_confidence = 0
                
                for entity in entity_list:
                    all_methods.add(entity.get('method', 'unknown'))
                    if 'start_pos' in entity and 'end_pos' in entity:
                        all_positions.append((entity['start_pos'], entity['end_pos']))
                    total_confidence += entity.get('confidence', 0)
                
                # Calculate combined confidence
                combined_confidence = min(1.0, total_confidence / len(entity_list) + len(all_methods) * 0.1)
                
                # Create merged entity
                merged_entity = best_entity.copy()
                merged_entity.update({
                    'confidence': combined_confidence,
                    'methods': list(all_methods),
                    'method_count': len(all_methods),
                    'occurrence_count': len(entity_list),
                    'positions': all_positions
                })
                
                merged_entities.append(merged_entity)
            
            return merged_entities
            
        except Exception as e:
            logger.error(f"Entity merging failed: {e}")
            return entities
    
    async def _validate_and_rank_entities(self, 
                                        entities: List[Dict[str, Any]], 
                                        text: str) -> List[Dict[str, Any]]:
        """Validate and rank entities."""
        try:
            if not entities:
                return []
            
            validated_entities = []
            
            for entity in entities:
                # Calculate final confidence
                final_confidence = await self._calculate_final_confidence(entity, text)
                entity['final_confidence'] = final_confidence
                
                # Filter by minimum confidence
                if final_confidence >= self.min_entity_confidence:
                    validated_entities.append(entity)
            
            # Sort by confidence
            validated_entities.sort(key=lambda x: x['final_confidence'], reverse=True)
            
            # Apply deduplication and ranking rules
            final_entities = self._apply_ranking_rules(validated_entities)
            
            return final_entities[:self.max_entities_per_text]
            
        except Exception as e:
            logger.error(f"Entity validation failed: {e}")
            return entities[:self.max_entities_per_text]
    
    async def _calculate_final_confidence(self, 
                                        entity: Dict[str, Any], 
                                        text: str) -> float:
        """Calculate final confidence score for entity."""
        try:
            base_confidence = entity.get('confidence', 0.5)
            
            # Method diversity bonus
            method_count = entity.get('method_count', 1)
            method_bonus = min(0.2, (method_count - 1) * 0.1)
            
            # Frequency bonus
            occurrence_count = entity.get('occurrence_count', 1)
            frequency_bonus = min(0.2, (occurrence_count - 1) * 0.05)
            
            # Length bonus (longer entities are often more specific)
            entity_text = entity.get('text', '')
            length_bonus = min(0.1, len(entity_text.split()) * 0.02)
            
            # Type confidence (some types are more reliable)
            type_confidences = {
                'Person': 0.1,
                'Organization': 0.1,
                'Location': 0.1,
                'Technology': 0.05,
                'Date': 0.15,
                'Money': 0.15
            }
            type_bonus = type_confidences.get(entity.get('type', ''), 0)
            
            # Combine all factors
            final_confidence = (base_confidence + 
                              method_bonus + 
                              frequency_bonus + 
                              length_bonus + 
                              type_bonus)
            
            return min(1.0, final_confidence)
            
        except Exception as e:
            logger.error(f"Final confidence calculation failed: {e}")
            return entity.get('confidence', 0.5)
    
    def _apply_ranking_rules(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply ranking rules to prioritize better entities."""
        try:
            # Group by entity type
            type_groups = defaultdict(list)
            for entity in entities:
                entity_type = entity.get('type', 'Unknown')
                type_groups[entity_type].append(entity)
            
            final_entities = []
            
            # Prioritize certain types
            type_priorities = {
                'Person': 1.0,
                'Organization': 0.9,
                'Location': 0.8,
                'Technology': 0.7,
                'Concept': 0.6,
                'Date': 0.5,
                'Money': 0.4
            }
            
            # Sort types by priority
            sorted_types = sorted(type_groups.keys(), 
                                key=lambda t: type_priorities.get(t, 0.3), 
                                reverse=True)
            
            # Take entities from each type proportionally
            max_per_type = max(1, self.max_entities_per_text // len(type_groups))
            
            for entity_type in sorted_types:
                type_entities = type_groups[entity_type]
                type_entities.sort(key=lambda x: x['final_confidence'], reverse=True)
                final_entities.extend(type_entities[:max_per_type])
            
            # Sort final list by confidence
            final_entities.sort(key=lambda x: x['final_confidence'], reverse=True)
            
            return final_entities
            
        except Exception as e:
            logger.error(f"Ranking rules application failed: {e}")
            return entities
    
    async def get_entity_types(self) -> List[str]:
        """Get list of supported entity types."""
        return list(set(self.entity_type_mapping.values()))
    
    async def get_extractor_stats(self) -> Dict[str, Any]:
        """Get entity extractor statistics."""
        try:
            return {
                'min_entity_confidence': self.min_entity_confidence,
                'max_entities_per_text': self.max_entities_per_text,
                'enable_custom_patterns': self.enable_custom_patterns,
                'spacy_models': {
                    'english_available': self.nlp_en is not None,
                    'russian_available': self.nlp_ru is not None
                },
                'entity_types': len(set(self.entity_type_mapping.values())),
                'custom_patterns': len(self.custom_patterns),
                'tokenizer_initialized': self.tokenizer._initialized if hasattr(self.tokenizer, '_initialized') else False
            }
            
        except Exception as e:
            logger.error(f"Extractor stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
entity_extractor = EntityExtractor()