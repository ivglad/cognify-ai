"""
Relation extraction system for knowledge graph construction.
"""
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import re
from collections import defaultdict, Counter

import trio
import spacy

from app.services.kg.entity_extractor import entity_extractor
from app.services.nlp.rag_tokenizer import rag_tokenizer
from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class RelationExtractor:
    """
    Relation extraction system for identifying relationships between entities.
    """
    
    def __init__(self,
                 min_relation_confidence: float = 0.5,
                 max_relations_per_text: int = 100,
                 max_entity_distance: int = 50):
        """
        Initialize relation extractor.
        
        Args:
            min_relation_confidence: Minimum confidence threshold for relations
            max_relations_per_text: Maximum relations to extract per text
            max_entity_distance: Maximum word distance between related entities
        """
        self.min_relation_confidence = min_relation_confidence
        self.max_relations_per_text = max_relations_per_text
        self.max_entity_distance = max_entity_distance
        
        self.entity_extractor = entity_extractor
        self.tokenizer = rag_tokenizer
        self.cache = cache_manager
        
        # Relation patterns for different types
        self.relation_patterns = self._initialize_relation_patterns()
        
        # Dependency patterns (for spaCy)
        self.dependency_patterns = self._initialize_dependency_patterns()
    
    def _initialize_relation_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize regex patterns for relation extraction."""
        return {
            'is_a': [
                re.compile(r'(.+?)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(.+?)(?:\.|,|;|$)', re.IGNORECASE),
                re.compile(r'(.+?)\s+(?:является|представляет собой|это)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE)
            ],
            'has_property': [
                re.compile(r'(.+?)\s+(?:has|have|had|contains|includes)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE),
                re.compile(r'(.+?)\s+(?:имеет|содержит|включает)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE)
            ],
            'located_in': [
                re.compile(r'(.+?)\s+(?:in|at|located in|situated in)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE),
                re.compile(r'(.+?)\s+(?:в|на|находится в|расположен в)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE)
            ],
            'works_for': [
                re.compile(r'(.+?)\s+(?:works for|employed by|at)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE),
                re.compile(r'(.+?)\s+(?:работает в|сотрудник)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE)
            ],
            'created_by': [
                re.compile(r'(.+?)\s+(?:created by|developed by|made by|invented by)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE),
                re.compile(r'(.+?)\s+(?:создан|разработан|изобретен)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE)
            ],
            'part_of': [
                re.compile(r'(.+?)\s+(?:part of|component of|element of)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE),
                re.compile(r'(.+?)\s+(?:часть|компонент|элемент)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE)
            ],
            'uses': [
                re.compile(r'(.+?)\s+(?:uses|utilizes|employs|applies)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE),
                re.compile(r'(.+?)\s+(?:использует|применяет|задействует)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE)
            ],
            'similar_to': [
                re.compile(r'(.+?)\s+(?:similar to|like|resembles|comparable to)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE),
                re.compile(r'(.+?)\s+(?:похож на|подобен|аналогичен)\s+(.+?)(?:\.|,|;|$)', re.IGNORECASE)
            ]
        }
    
    def _initialize_dependency_patterns(self) -> Dict[str, List[str]]:
        """Initialize dependency parsing patterns for spaCy."""
        return {
            'subject_object': ['nsubj', 'dobj'],
            'subject_complement': ['nsubj', 'attr'],
            'possessive': ['poss'],
            'compound': ['compound'],
            'apposition': ['appos'],
            'prepositional': ['prep', 'pobj']
        }
    
    async def extract_relations(self, 
                              text: str,
                              entities: Optional[List[Dict[str, Any]]] = None,
                              document_id: Optional[str] = None,
                              use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Extract relations from text.
        
        Args:
            text: Input text
            entities: Pre-extracted entities (optional)
            document_id: Optional document ID for caching
            use_cache: Whether to use caching
            
        Returns:
            List of extracted relations with metadata
        """
        if not text or not text.strip():
            return []
        
        try:
            # Initialize services
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            # Check cache
            if use_cache and document_id:
                cache_key = f"relations:{document_id}"
                cached_relations = await self.cache.get(cache_key)
                if cached_relations:
                    logger.debug(f"Using cached relations for document {document_id}")
                    return cached_relations
            
            # Extract entities if not provided
            if entities is None:
                entities = await self.entity_extractor.extract_entities(text, document_id, use_cache)
            
            if not entities:
                logger.debug("No entities found, cannot extract relations")
                return []
            
            logger.debug(f"Extracting relations from text with {len(entities)} entities")
            
            # Extract relations using multiple methods
            relations = await self._extract_multi_method_relations(text, entities)
            
            # Validate and rank relations
            validated_relations = await self._validate_and_rank_relations(relations, text, entities)
            
            # Cache results
            if use_cache and document_id and validated_relations:
                await self.cache.set(f"relations:{document_id}", validated_relations, ttl=86400)  # 24 hours
            
            logger.debug(f"Extracted {len(validated_relations)} validated relations")
            
            return validated_relations
            
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return [] 
   
    async def _extract_multi_method_relations(self, 
                                            text: str, 
                                            entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations using multiple methods."""
        try:
            # Method 1: Pattern-based extraction
            pattern_relations = await self._extract_pattern_relations(text, entities)
            
            # Method 2: Dependency parsing (if spaCy available)
            dependency_relations = await self._extract_dependency_relations(text, entities)
            
            # Method 3: Co-occurrence based relations
            cooccurrence_relations = await self._extract_cooccurrence_relations(text, entities)
            
            # Method 4: Statistical relations
            statistical_relations = await self._extract_statistical_relations(text, entities)
            
            # Combine all relations
            all_relations = []
            all_relations.extend(pattern_relations)
            all_relations.extend(dependency_relations)
            all_relations.extend(cooccurrence_relations)
            all_relations.extend(statistical_relations)
            
            # Merge duplicate relations
            merged_relations = await self._merge_duplicate_relations(all_relations)
            
            return merged_relations
            
        except Exception as e:
            logger.error(f"Multi-method relation extraction failed: {e}")
            return []
    
    async def _extract_pattern_relations(self, 
                                       text: str, 
                                       entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations using predefined patterns."""
        try:
            relations = []
            
            # Create entity lookup for quick matching
            entity_texts = {entity['text'].lower(): entity for entity in entities}
            
            for relation_type, patterns in self.relation_patterns.items():
                for pattern in patterns:
                    matches = pattern.finditer(text)
                    
                    for match in matches:
                        if len(match.groups()) >= 2:
                            subject_text = match.group(1).strip()
                            object_text = match.group(2).strip()
                            
                            # Find matching entities
                            subject_entity = self._find_matching_entity(subject_text, entity_texts)
                            object_entity = self._find_matching_entity(object_text, entity_texts)
                            
                            if subject_entity and object_entity and subject_entity != object_entity:
                                relations.append({
                                    'subject': subject_entity['text'],
                                    'subject_type': subject_entity['type'],
                                    'predicate': relation_type,
                                    'object': object_entity['text'],
                                    'object_type': object_entity['type'],
                                    'confidence': 0.8,  # High confidence for pattern matches
                                    'method': 'pattern',
                                    'pattern_match': match.group(0),
                                    'start_pos': match.start(),
                                    'end_pos': match.end(),
                                    'context': self._get_relation_context(text, match.start(), match.end())
                                })
            
            return relations
            
        except Exception as e:
            logger.error(f"Pattern relation extraction failed: {e}")
            return []
    
    async def _extract_dependency_relations(self, 
                                          text: str, 
                                          entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations using dependency parsing."""
        try:
            relations = []
            
            # Check if spaCy models are available
            if not (self.entity_extractor.nlp_en or self.entity_extractor.nlp_ru):
                return []
            
            # Detect language and choose model
            language = await self._detect_language(text)
            nlp_model = None
            
            if language == 'russian' and self.entity_extractor.nlp_ru:
                nlp_model = self.entity_extractor.nlp_ru
            elif self.entity_extractor.nlp_en:
                nlp_model = self.entity_extractor.nlp_en
            
            if not nlp_model:
                return []
            
            # Process text with spaCy
            doc = await trio.to_thread.run_sync(nlp_model, text)
            
            # Create entity position mapping
            entity_positions = {}
            for entity in entities:
                for pos in entity.get('positions', []):
                    if isinstance(pos, tuple) and len(pos) == 2:
                        start, end = pos
                        entity_positions[(start, end)] = entity
            
            # Extract relations from dependency tree
            for sent in doc.sents:
                sent_relations = await self._extract_sentence_relations(sent, entity_positions)
                relations.extend(sent_relations)
            
            return relations
            
        except Exception as e:
            logger.error(f"Dependency relation extraction failed: {e}")
            return []
    
    async def _extract_sentence_relations(self, 
                                        sent, 
                                        entity_positions: Dict[Tuple[int, int], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations from a single sentence using dependency parsing."""
        try:
            relations = []
            
            # Find entities in this sentence
            sent_entities = []
            for token in sent:
                for (start, end), entity in entity_positions.items():
                    if start <= token.idx < end:
                        sent_entities.append((token, entity))
                        break
            
            if len(sent_entities) < 2:
                return []
            
            # Extract relations based on dependency patterns
            for i, (token1, entity1) in enumerate(sent_entities):
                for j, (token2, entity2) in enumerate(sent_entities[i+1:], i+1):
                    relation = await self._analyze_token_relation(token1, token2, entity1, entity2, sent)
                    if relation:
                        relations.append(relation)
            
            return relations
            
        except Exception as e:
            logger.error(f"Sentence relation extraction failed: {e}")
            return []
    
    async def _analyze_token_relation(self, 
                                    token1, token2, 
                                    entity1: Dict[str, Any], 
                                    entity2: Dict[str, Any], 
                                    sent) -> Optional[Dict[str, Any]]:
        """Analyze relation between two tokens using dependency parsing."""
        try:
            # Find dependency path between tokens
            path = self._find_dependency_path(token1, token2)
            
            if not path:
                return None
            
            # Classify relation based on dependency path
            relation_type = self._classify_dependency_relation(path, token1, token2)
            
            if not relation_type:
                return None
            
            # Calculate confidence based on path length and types
            confidence = self._calculate_dependency_confidence(path, relation_type)
            
            return {
                'subject': entity1['text'],
                'subject_type': entity1['type'],
                'predicate': relation_type,
                'object': entity2['text'],
                'object_type': entity2['type'],
                'confidence': confidence,
                'method': 'dependency',
                'dependency_path': [token.dep_ for token in path],
                'context': sent.text
            }
            
        except Exception as e:
            logger.error(f"Token relation analysis failed: {e}")
            return None
    
    def _find_dependency_path(self, token1, token2):
        """Find dependency path between two tokens."""
        try:
            # Simple path finding - can be improved with more sophisticated algorithms
            if token1.head == token2 or token2.head == token1:
                return [token1, token2]
            
            # Check if they share a common head
            if token1.head == token2.head:
                return [token1, token1.head, token2]
            
            # More complex path finding would go here
            return None
            
        except Exception:
            return None
    
    def _classify_dependency_relation(self, path, token1, token2) -> Optional[str]:
        """Classify relation type based on dependency path."""
        try:
            if len(path) == 2:
                # Direct dependency
                if token1.dep_ == 'nsubj' and token2.dep_ == 'dobj':
                    return 'acts_on'
                elif token1.dep_ == 'nsubj' and token2.dep_ == 'attr':
                    return 'is_a'
                elif token1.dep_ == 'poss':
                    return 'has_property'
                elif token1.dep_ == 'compound':
                    return 'part_of'
                elif token1.dep_ == 'appos':
                    return 'is_a'
            
            elif len(path) == 3:
                # Indirect dependency through common head
                head = path[1]
                if head.pos_ in ['VERB', 'AUX']:
                    return 'related_to'
                elif head.pos_ in ['NOUN', 'PROPN']:
                    return 'associated_with'
            
            return 'related_to'  # Default relation
            
        except Exception:
            return None
    
    def _calculate_dependency_confidence(self, path, relation_type: str) -> float:
        """Calculate confidence for dependency-based relation."""
        try:
            base_confidence = 0.6
            
            # Shorter paths are more reliable
            if len(path) == 2:
                base_confidence += 0.2
            elif len(path) == 3:
                base_confidence += 0.1
            
            # Some relation types are more reliable
            reliable_types = {'is_a', 'has_property', 'part_of'}
            if relation_type in reliable_types:
                base_confidence += 0.1
            
            return min(1.0, base_confidence)
            
        except Exception:
            return 0.5
    
    async def _extract_cooccurrence_relations(self, 
                                            text: str, 
                                            entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations based on entity co-occurrence."""
        try:
            relations = []
            
            # Split text into sentences
            sentences = await self.tokenizer.split_into_sentences(text)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Find entities in this sentence
                sentence_entities = []
                for entity in entities:
                    if entity['text'].lower() in sentence_lower:
                        sentence_entities.append(entity)
                
                # Create relations between co-occurring entities
                if len(sentence_entities) >= 2:
                    for i, entity1 in enumerate(sentence_entities):
                        for entity2 in sentence_entities[i+1:]:
                            if entity1 != entity2:
                                # Calculate distance between entities
                                distance = self._calculate_entity_distance(entity1['text'], entity2['text'], sentence)
                                
                                if distance <= self.max_entity_distance:
                                    # Infer relation type based on entity types
                                    relation_type = self._infer_relation_type(entity1, entity2, sentence)
                                    
                                    # Calculate confidence based on distance and context
                                    confidence = self._calculate_cooccurrence_confidence(distance, sentence, entity1, entity2)
                                    
                                    relations.append({
                                        'subject': entity1['text'],
                                        'subject_type': entity1['type'],
                                        'predicate': relation_type,
                                        'object': entity2['text'],
                                        'object_type': entity2['type'],
                                        'confidence': confidence,
                                        'method': 'cooccurrence',
                                        'distance': distance,
                                        'context': sentence
                                    })
            
            return relations
            
        except Exception as e:
            logger.error(f"Co-occurrence relation extraction failed: {e}")
            return [] 
   
    def _calculate_entity_distance(self, entity1_text: str, entity2_text: str, sentence: str) -> int:
        """Calculate word distance between two entities in a sentence."""
        try:
            words = sentence.split()
            entity1_pos = -1
            entity2_pos = -1
            
            # Find positions of entities
            for i, word in enumerate(words):
                if entity1_text.lower() in word.lower() and entity1_pos == -1:
                    entity1_pos = i
                if entity2_text.lower() in word.lower() and entity2_pos == -1:
                    entity2_pos = i
            
            if entity1_pos != -1 and entity2_pos != -1:
                return abs(entity1_pos - entity2_pos)
            
            return self.max_entity_distance + 1  # Max distance if not found
            
        except Exception:
            return self.max_entity_distance + 1
    
    def _infer_relation_type(self, 
                           entity1: Dict[str, Any], 
                           entity2: Dict[str, Any], 
                           context: str) -> str:
        """Infer relation type based on entity types and context."""
        try:
            type1 = entity1.get('type', '')
            type2 = entity2.get('type', '')
            context_lower = context.lower()
            
            # Type-based relation inference
            if type1 == 'Person' and type2 == 'Organization':
                if any(word in context_lower for word in ['works', 'employed', 'ceo', 'director']):
                    return 'works_for'
                else:
                    return 'associated_with'
            
            elif type1 == 'Person' and type2 == 'Location':
                if any(word in context_lower for word in ['born', 'from', 'lives']):
                    return 'located_in'
                else:
                    return 'associated_with'
            
            elif type1 == 'Technology' and type2 == 'Organization':
                if any(word in context_lower for word in ['developed', 'created', 'made']):
                    return 'created_by'
                else:
                    return 'used_by'
            
            elif type1 == 'Product' and type2 == 'Organization':
                return 'created_by'
            
            elif 'Concept' in [type1, type2]:
                return 'related_to'
            
            # Default relation
            return 'associated_with'
            
        except Exception:
            return 'related_to'
    
    def _calculate_cooccurrence_confidence(self, 
                                         distance: int, 
                                         sentence: str, 
                                         entity1: Dict[str, Any], 
                                         entity2: Dict[str, Any]) -> float:
        """Calculate confidence for co-occurrence based relation."""
        try:
            base_confidence = 0.3
            
            # Distance bonus (closer entities are more likely to be related)
            distance_bonus = max(0, (self.max_entity_distance - distance) / self.max_entity_distance * 0.3)
            
            # Context clues bonus
            context_bonus = 0
            context_lower = sentence.lower()
            
            # Look for relation indicators
            relation_indicators = [
                'is', 'are', 'was', 'were', 'has', 'have', 'contains', 'includes',
                'works', 'created', 'developed', 'uses', 'applies', 'located'
            ]
            
            for indicator in relation_indicators:
                if indicator in context_lower:
                    context_bonus += 0.1
                    break
            
            # Entity type compatibility bonus
            type_bonus = self._calculate_type_compatibility(entity1.get('type', ''), entity2.get('type', ''))
            
            total_confidence = base_confidence + distance_bonus + context_bonus + type_bonus
            
            return min(1.0, total_confidence)
            
        except Exception:
            return 0.3
    
    def _calculate_type_compatibility(self, type1: str, type2: str) -> float:
        """Calculate compatibility bonus based on entity types."""
        try:
            # Compatible type pairs
            compatible_pairs = {
                ('Person', 'Organization'): 0.2,
                ('Person', 'Location'): 0.15,
                ('Technology', 'Organization'): 0.2,
                ('Product', 'Organization'): 0.2,
                ('Concept', 'Technology'): 0.15,
                ('Method', 'Technology'): 0.15
            }
            
            # Check both directions
            pair1 = (type1, type2)
            pair2 = (type2, type1)
            
            return compatible_pairs.get(pair1, compatible_pairs.get(pair2, 0))
            
        except Exception:
            return 0
    
    async def _extract_statistical_relations(self, 
                                           text: str, 
                                           entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations using statistical analysis."""
        try:
            relations = []
            
            # Count entity co-occurrences across sentences
            cooccurrence_counts = defaultdict(int)
            
            sentences = await self.tokenizer.split_into_sentences(text)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                sentence_entities = []
                
                # Find entities in sentence
                for entity in entities:
                    if entity['text'].lower() in sentence_lower:
                        sentence_entities.append(entity)
                
                # Count co-occurrences
                for i, entity1 in enumerate(sentence_entities):
                    for entity2 in sentence_entities[i+1:]:
                        pair = tuple(sorted([entity1['text'], entity2['text']]))
                        cooccurrence_counts[pair] += 1
            
            # Create relations for frequently co-occurring entities
            for (entity1_text, entity2_text), count in cooccurrence_counts.items():
                if count >= 2:  # Minimum co-occurrence threshold
                    # Find entity objects
                    entity1 = next((e for e in entities if e['text'] == entity1_text), None)
                    entity2 = next((e for e in entities if e['text'] == entity2_text), None)
                    
                    if entity1 and entity2:
                        # Calculate statistical confidence
                        confidence = min(0.8, 0.3 + count * 0.1)
                        
                        # Infer relation type
                        relation_type = self._infer_statistical_relation_type(entity1, entity2, text)
                        
                        relations.append({
                            'subject': entity1['text'],
                            'subject_type': entity1['type'],
                            'predicate': relation_type,
                            'object': entity2['text'],
                            'object_type': entity2['type'],
                            'confidence': confidence,
                            'method': 'statistical',
                            'cooccurrence_count': count,
                            'context': f"Co-occurs {count} times in text"
                        })
            
            return relations
            
        except Exception as e:
            logger.error(f"Statistical relation extraction failed: {e}")
            return []
    
    def _infer_statistical_relation_type(self, 
                                       entity1: Dict[str, Any], 
                                       entity2: Dict[str, Any], 
                                       text: str) -> str:
        """Infer relation type for statistical relations."""
        try:
            # Use same logic as co-occurrence but with text-wide context
            return self._infer_relation_type(entity1, entity2, text[:200])  # Use first 200 chars as context
            
        except Exception:
            return 'frequently_mentioned_with'
    
    def _find_matching_entity(self, 
                            text: str, 
                            entity_lookup: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find entity that matches the given text."""
        try:
            text_lower = text.lower().strip()
            
            # Exact match
            if text_lower in entity_lookup:
                return entity_lookup[text_lower]
            
            # Partial match
            for entity_text, entity in entity_lookup.items():
                if text_lower in entity_text or entity_text in text_lower:
                    return entity
            
            return None
            
        except Exception:
            return None
    
    def _get_relation_context(self, text: str, start_pos: int, end_pos: int, context_size: int = 100) -> str:
        """Get context around relation."""
        try:
            context_start = max(0, start_pos - context_size)
            context_end = min(len(text), end_pos + context_size)
            
            return text[context_start:context_end].strip()
            
        except Exception:
            return ""
    
    async def _detect_language(self, text: str) -> str:
        """Detect language of text."""
        try:
            # Simple language detection
            cyrillic_count = len(re.findall(r'[а-яёА-ЯЁ]', text))
            latin_count = len(re.findall(r'[a-zA-Z]', text))
            
            if cyrillic_count > latin_count:
                return 'russian'
            else:
                return 'english'
                
        except Exception:
            return 'english'
    
    async def _merge_duplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge duplicate relations."""
        try:
            if not relations:
                return []
            
            # Group relations by subject-predicate-object triple
            relation_groups = defaultdict(list)
            
            for relation in relations:
                key = (
                    relation.get('subject', '').lower(),
                    relation.get('predicate', ''),
                    relation.get('object', '').lower()
                )
                relation_groups[key].append(relation)
            
            merged_relations = []
            
            for triple, relation_list in relation_groups.items():
                if not relation_list:
                    continue
                
                # Sort by confidence
                relation_list.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                best_relation = relation_list[0]
                
                # Combine information from all instances
                all_methods = set()
                all_confidences = []
                all_contexts = []
                
                for rel in relation_list:
                    all_methods.add(rel.get('method', 'unknown'))
                    all_confidences.append(rel.get('confidence', 0))
                    if rel.get('context'):
                        all_contexts.append(rel['context'])
                
                # Calculate combined confidence
                combined_confidence = self._combine_relation_confidences(all_confidences, list(all_methods))
                
                # Create merged relation
                merged_relation = best_relation.copy()
                merged_relation.update({
                    'confidence': combined_confidence,
                    'methods': list(all_methods),
                    'method_count': len(all_methods),
                    'occurrence_count': len(relation_list),
                    'all_contexts': all_contexts[:3]  # Keep top 3 contexts
                })
                
                merged_relations.append(merged_relation)
            
            return merged_relations
            
        except Exception as e:
            logger.error(f"Relation merging failed: {e}")
            return relations
    
    def _combine_relation_confidences(self, confidences: List[float], methods: List[str]) -> float:
        """Combine confidence scores from multiple methods."""
        try:
            if not confidences:
                return 0.0
            
            # Weight different methods
            method_weights = {
                'pattern': 0.4,
                'dependency': 0.3,
                'cooccurrence': 0.2,
                'statistical': 0.1
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
            
        except Exception:
            return sum(confidences) / len(confidences) if confidences else 0.0
    
    async def _validate_and_rank_relations(self, 
                                         relations: List[Dict[str, Any]], 
                                         text: str, 
                                         entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and rank relations."""
        try:
            if not relations:
                return []
            
            validated_relations = []
            
            for relation in relations:
                # Calculate final confidence
                final_confidence = await self._calculate_final_relation_confidence(relation, text, entities)
                relation['final_confidence'] = final_confidence
                
                # Filter by minimum confidence
                if final_confidence >= self.min_relation_confidence:
                    validated_relations.append(relation)
            
            # Sort by confidence
            validated_relations.sort(key=lambda x: x['final_confidence'], reverse=True)
            
            # Apply ranking rules
            final_relations = self._apply_relation_ranking_rules(validated_relations)
            
            return final_relations[:self.max_relations_per_text]
            
        except Exception as e:
            logger.error(f"Relation validation failed: {e}")
            return relations[:self.max_relations_per_text]
    
    async def _calculate_final_relation_confidence(self, 
                                                 relation: Dict[str, Any], 
                                                 text: str, 
                                                 entities: List[Dict[str, Any]]) -> float:
        """Calculate final confidence score for relation."""
        try:
            base_confidence = relation.get('confidence', 0.5)
            
            # Method diversity bonus
            method_count = relation.get('method_count', 1)
            method_bonus = min(0.2, (method_count - 1) * 0.1)
            
            # Entity confidence bonus
            subject_entity = next((e for e in entities if e['text'] == relation.get('subject', '')), None)
            object_entity = next((e for e in entities if e['text'] == relation.get('object', '')), None)
            
            entity_bonus = 0
            if subject_entity and object_entity:
                avg_entity_confidence = (subject_entity.get('final_confidence', 0.5) + 
                                       object_entity.get('final_confidence', 0.5)) / 2
                entity_bonus = (avg_entity_confidence - 0.5) * 0.2
            
            # Relation type bonus
            type_bonuses = {
                'is_a': 0.1,
                'part_of': 0.1,
                'created_by': 0.1,
                'works_for': 0.1,
                'located_in': 0.1
            }
            type_bonus = type_bonuses.get(relation.get('predicate', ''), 0)
            
            # Combine all factors
            final_confidence = (base_confidence + 
                              method_bonus + 
                              entity_bonus + 
                              type_bonus)
            
            return min(1.0, final_confidence)
            
        except Exception as e:
            logger.error(f"Final relation confidence calculation failed: {e}")
            return relation.get('confidence', 0.5)
    
    def _apply_relation_ranking_rules(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply ranking rules to prioritize better relations."""
        try:
            # Group by relation type
            type_groups = defaultdict(list)
            for relation in relations:
                rel_type = relation.get('predicate', 'unknown')
                type_groups[rel_type].append(relation)
            
            final_relations = []
            
            # Prioritize certain relation types
            type_priorities = {
                'is_a': 1.0,
                'part_of': 0.9,
                'created_by': 0.8,
                'works_for': 0.8,
                'located_in': 0.7,
                'uses': 0.6,
                'has_property': 0.6,
                'similar_to': 0.5,
                'related_to': 0.3,
                'associated_with': 0.2
            }
            
            # Sort types by priority
            sorted_types = sorted(type_groups.keys(), 
                                key=lambda t: type_priorities.get(t, 0.1), 
                                reverse=True)
            
            # Take relations from each type proportionally
            max_per_type = max(1, self.max_relations_per_text // len(type_groups))
            
            for rel_type in sorted_types:
                type_relations = type_groups[rel_type]
                type_relations.sort(key=lambda x: x['final_confidence'], reverse=True)
                final_relations.extend(type_relations[:max_per_type])
            
            # Sort final list by confidence
            final_relations.sort(key=lambda x: x['final_confidence'], reverse=True)
            
            return final_relations
            
        except Exception as e:
            logger.error(f"Relation ranking rules application failed: {e}")
            return relations
    
    async def get_relation_types(self) -> List[str]:
        """Get list of supported relation types."""
        return list(self.relation_patterns.keys()) + [
            'acts_on', 'related_to', 'associated_with', 'frequently_mentioned_with'
        ]
    
    async def get_extractor_stats(self) -> Dict[str, Any]:
        """Get relation extractor statistics."""
        try:
            return {
                'min_relation_confidence': self.min_relation_confidence,
                'max_relations_per_text': self.max_relations_per_text,
                'max_entity_distance': self.max_entity_distance,
                'relation_patterns': len(self.relation_patterns),
                'dependency_patterns': len(self.dependency_patterns),
                'spacy_available': (self.entity_extractor.nlp_en is not None or 
                                  self.entity_extractor.nlp_ru is not None),
                'tokenizer_initialized': self.tokenizer._initialized if hasattr(self.tokenizer, '_initialized') else False
            }
            
        except Exception as e:
            logger.error(f"Extractor stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
relation_extractor = RelationExtractor()