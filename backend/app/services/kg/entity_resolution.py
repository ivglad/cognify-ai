"""
Entity resolution system for knowledge graph with similarity matching and merging.
"""
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from datetime import datetime

from app.core.logging_config import get_logger
from app.services.embeddings.embedding_service import embedding_service

logger = get_logger(__name__)


class SimilarityMethod(str, Enum):
    """Similarity calculation methods."""
    STRING_SIMILARITY = "string_similarity"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    HYBRID = "hybrid"
    PHONETIC = "phonetic"


class EntityType(str, Enum):
    """Entity types for resolution."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    PRODUCT = "product"
    OTHER = "other"


@dataclass
class EntityCandidate:
    """Entity candidate for resolution."""
    entity_id: str
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    attributes: Dict[str, Any] = None
    source_chunks: List[str] = None
    confidence: float = 0.0
    embedding: Optional[np.ndarray] = None


@dataclass
class SimilarityScore:
    """Similarity score between entities."""
    entity1_id: str
    entity2_id: str
    similarity_score: float
    method: SimilarityMethod
    confidence: float
    details: Dict[str, Any] = None


@dataclass
class EntityCluster:
    """Cluster of similar entities."""
    cluster_id: str
    entities: List[EntityCandidate]
    representative_entity: EntityCandidate
    similarity_scores: List[SimilarityScore]
    merge_confidence: float
    created_at: datetime

class En
tityResolver:
    """
    Entity resolution system with similarity matching and intelligent merging.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.8,
                 embedding_weight: float = 0.6,
                 string_weight: float = 0.4):
        self.similarity_threshold = similarity_threshold
        self.embedding_weight = embedding_weight
        self.string_weight = string_weight
        
        # Similarity calculation components
        self.embedding_service = embedding_service
        
        # Entity type specific thresholds
        self.type_thresholds = {
            EntityType.PERSON: 0.85,
            EntityType.ORGANIZATION: 0.80,
            EntityType.LOCATION: 0.75,
            EntityType.CONCEPT: 0.70,
            EntityType.EVENT: 0.80,
            EntityType.PRODUCT: 0.75,
            EntityType.OTHER: 0.70
        }
        
        # Caching for embeddings and similarity scores
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # Statistics
        self.resolution_stats = {
            'total_entities_processed': 0,
            'total_clusters_created': 0,
            'total_entities_merged': 0,
            'average_cluster_size': 0.0,
            'resolution_accuracy': 0.0
        }
    
    async def resolve_entities(self, entities: List[EntityCandidate]) -> List[EntityCluster]:
        """
        Resolve duplicate entities and create clusters.
        
        Args:
            entities: List of entity candidates to resolve
            
        Returns:
            List of entity clusters with resolved duplicates
        """
        try:
            if not entities:
                return []
            
            logger.info(f"Starting entity resolution for {len(entities)} entities")
            
            # Generate embeddings for all entities
            await self._generate_embeddings(entities)
            
            # Calculate similarity matrix
            similarity_matrix = await self._calculate_similarity_matrix(entities)
            
            # Create clusters based on similarity
            clusters = await self._create_clusters(entities, similarity_matrix)
            
            # Merge entities within clusters
            resolved_clusters = await self._merge_cluster_entities(clusters)
            
            # Update statistics
            self._update_resolution_stats(entities, resolved_clusters)
            
            logger.info(f"Entity resolution completed: {len(resolved_clusters)} clusters created")
            return resolved_clusters
            
        except Exception as e:
            logger.error(f"Entity resolution failed: {e}")
            return []
    
    async def _generate_embeddings(self, entities: List[EntityCandidate]):
        """Generate embeddings for entities."""
        try:
            for entity in entities:
                if entity.entity_id in self.embedding_cache:
                    entity.embedding = self.embedding_cache[entity.entity_id]
                    continue
                
                # Create text for embedding
                text_parts = [entity.name]
                if entity.description:
                    text_parts.append(entity.description)
                if entity.attributes:
                    for key, value in entity.attributes.items():
                        if isinstance(value, str):
                            text_parts.append(f"{key}: {value}")
                
                text = " ".join(text_parts)
                
                # Generate embedding
                embedding = await self.embedding_service.generate_embedding(text)
                if embedding is not None:
                    entity.embedding = embedding
                    self.embedding_cache[entity.entity_id] = embedding
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
    
    async def _calculate_similarity_matrix(self, entities: List[EntityCandidate]) -> np.ndarray:
        """Calculate similarity matrix between all entities."""
        try:
            n = len(entities)
            similarity_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    entity1, entity2 = entities[i], entities[j]
                    
                    # Check cache first
                    cache_key = (entity1.entity_id, entity2.entity_id)
                    if cache_key in self.similarity_cache:
                        similarity = self.similarity_cache[cache_key]
                    else:
                        similarity = await self._calculate_entity_similarity(entity1, entity2)
                        self.similarity_cache[cache_key] = similarity
                    
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
            
            # Set diagonal to 1.0 (entity similarity with itself)
            np.fill_diagonal(similarity_matrix, 1.0)
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Similarity matrix calculation failed: {e}")
            return np.eye(len(entities))
    
    async def _calculate_entity_similarity(self, 
                                         entity1: EntityCandidate, 
                                         entity2: EntityCandidate) -> float:
        """Calculate similarity between two entities."""
        try:
            # Different entity types get lower similarity
            if entity1.entity_type != entity2.entity_type:
                return 0.0
            
            # Calculate string similarity
            string_sim = self._calculate_string_similarity(entity1.name, entity2.name)
            
            # Calculate embedding similarity if available
            embedding_sim = 0.0
            if entity1.embedding is not None and entity2.embedding is not None:
                embedding_sim = self._calculate_embedding_similarity(
                    entity1.embedding, entity2.embedding
                )
            
            # Calculate attribute similarity
            attr_sim = self._calculate_attribute_similarity(
                entity1.attributes or {}, entity2.attributes or {}
            )
            
            # Weighted combination
            if entity1.embedding is not None and entity2.embedding is not None:
                # Use embedding + string + attributes
                similarity = (
                    self.embedding_weight * embedding_sim +
                    self.string_weight * string_sim +
                    0.2 * attr_sim
                )
            else:
                # Use string + attributes only
                similarity = 0.8 * string_sim + 0.2 * attr_sim
            
            return min(1.0, max(0.0, similarity))
            
        except Exception as e:
            logger.error(f"Entity similarity calculation failed: {e}")
            return 0.0    

    def _calculate_string_similarity(self, name1: str, name2: str) -> float:
        """Calculate string similarity between entity names."""
        try:
            if not name1 or not name2:
                return 0.0
            
            # Normalize names
            name1 = self._normalize_name(name1)
            name2 = self._normalize_name(name2)
            
            # Exact match
            if name1 == name2:
                return 1.0
            
            # Calculate edit distance similarity
            edit_sim = self._edit_distance_similarity(name1, name2)
            
            # Calculate token similarity
            token_sim = self._token_similarity(name1, name2)
            
            # Calculate phonetic similarity
            phonetic_sim = self._phonetic_similarity(name1, name2)
            
            # Weighted combination
            similarity = 0.5 * edit_sim + 0.3 * token_sim + 0.2 * phonetic_sim
            
            return similarity
            
        except Exception as e:
            logger.error(f"String similarity calculation failed: {e}")
            return 0.0
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        try:
            # Convert to lowercase
            name = name.lower().strip()
            
            # Remove extra whitespace
            name = re.sub(r'\s+', ' ', name)
            
            # Remove common prefixes/suffixes
            prefixes = ['mr.', 'mrs.', 'dr.', 'prof.', 'the ', 'a ']
            suffixes = [' inc.', ' ltd.', ' corp.', ' llc', ' co.']
            
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix):].strip()
            
            for suffix in suffixes:
                if name.endswith(suffix):
                    name = name[:-len(suffix)].strip()
            
            return name
            
        except Exception as e:
            logger.error(f"Name normalization failed: {e}")
            return name
    
    def _edit_distance_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity based on edit distance."""
        try:
            if not str1 or not str2:
                return 0.0
            
            # Calculate Levenshtein distance
            distance = self._levenshtein_distance(str1, str2)
            max_len = max(len(str1), len(str2))
            
            if max_len == 0:
                return 1.0
            
            similarity = 1.0 - (distance / max_len)
            return max(0.0, similarity)
            
        except Exception as e:
            logger.error(f"Edit distance similarity calculation failed: {e}")
            return 0.0
    
    def _levenshtein_distance(self, str1: str, str2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        try:
            if len(str1) < len(str2):
                return self._levenshtein_distance(str2, str1)
            
            if len(str2) == 0:
                return len(str1)
            
            previous_row = list(range(len(str2) + 1))
            for i, c1 in enumerate(str1):
                current_row = [i + 1]
                for j, c2 in enumerate(str2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
            
        except Exception as e:
            logger.error(f"Levenshtein distance calculation failed: {e}")
            return max(len(str1), len(str2))
    
    def _token_similarity(self, str1: str, str2: str) -> float:
        """Calculate token-based similarity."""
        try:
            tokens1 = set(str1.split())
            tokens2 = set(str2.split())
            
            if not tokens1 and not tokens2:
                return 1.0
            
            if not tokens1 or not tokens2:
                return 0.0
            
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            
            return len(intersection) / len(union)
            
        except Exception as e:
            logger.error(f"Token similarity calculation failed: {e}")
            return 0.0
    
    def _phonetic_similarity(self, str1: str, str2: str) -> float:
        """Calculate phonetic similarity using Soundex algorithm."""
        try:
            soundex1 = self._soundex(str1)
            soundex2 = self._soundex(str2)
            
            return 1.0 if soundex1 == soundex2 else 0.0
            
        except Exception as e:
            logger.error(f"Phonetic similarity calculation failed: {e}")
            return 0.0
    
    def _soundex(self, name: str) -> str:
        """Generate Soundex code for phonetic matching."""
        try:
            if not name:
                return "0000"
            
            name = name.upper()
            soundex = name[0]
            
            # Soundex mapping
            mapping = {
                'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3',
                'L': '4', 'MN': '5', 'R': '6'
            }
            
            for char in name[1:]:
                for key, value in mapping.items():
                    if char in key:
                        if value != soundex[-1]:
                            soundex += value
                        break
            
            # Pad or truncate to 4 characters
            soundex = soundex.ljust(4, '0')[:4]
            return soundex
            
        except Exception as e:
            logger.error(f"Soundex generation failed: {e}")
            return "0000"
    
    def _calculate_embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        try:
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Embedding similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_attribute_similarity(self, attrs1: Dict[str, Any], attrs2: Dict[str, Any]) -> float:
        """Calculate similarity between entity attributes."""
        try:
            if not attrs1 and not attrs2:
                return 1.0
            
            if not attrs1 or not attrs2:
                return 0.0
            
            # Find common attributes
            common_keys = set(attrs1.keys()).intersection(set(attrs2.keys()))
            
            if not common_keys:
                return 0.0
            
            matches = 0
            for key in common_keys:
                val1, val2 = attrs1[key], attrs2[key]
                
                if isinstance(val1, str) and isinstance(val2, str):
                    # String comparison
                    if self._normalize_name(val1) == self._normalize_name(val2):
                        matches += 1
                elif val1 == val2:
                    # Exact match
                    matches += 1
            
            return matches / len(common_keys)
            
        except Exception as e:
            logger.error(f"Attribute similarity calculation failed: {e}")
            return 0.0   
 
    async def _create_clusters(self, 
                             entities: List[EntityCandidate], 
                             similarity_matrix: np.ndarray) -> List[EntityCluster]:
        """Create entity clusters based on similarity matrix."""
        try:
            n = len(entities)
            visited = [False] * n
            clusters = []
            
            for i in range(n):
                if visited[i]:
                    continue
                
                # Start new cluster
                cluster_entities = [entities[i]]
                cluster_indices = [i]
                visited[i] = True
                
                # Find similar entities using threshold
                threshold = self.type_thresholds.get(
                    entities[i].entity_type, self.similarity_threshold
                )
                
                for j in range(i + 1, n):
                    if not visited[j] and similarity_matrix[i][j] >= threshold:
                        cluster_entities.append(entities[j])
                        cluster_indices.append(j)
                        visited[j] = True
                
                # Create cluster if it has entities
                if cluster_entities:
                    # Find representative entity (highest average similarity)
                    representative = self._find_representative_entity(
                        cluster_entities, cluster_indices, similarity_matrix
                    )
                    
                    # Calculate similarity scores within cluster
                    similarity_scores = []
                    for idx1 in cluster_indices:
                        for idx2 in cluster_indices:
                            if idx1 < idx2:
                                score = SimilarityScore(
                                    entity1_id=entities[idx1].entity_id,
                                    entity2_id=entities[idx2].entity_id,
                                    similarity_score=similarity_matrix[idx1][idx2],
                                    method=SimilarityMethod.HYBRID,
                                    confidence=0.8
                                )
                                similarity_scores.append(score)
                    
                    # Calculate merge confidence
                    merge_confidence = self._calculate_merge_confidence(
                        cluster_entities, similarity_scores
                    )
                    
                    cluster = EntityCluster(
                        cluster_id=f"cluster_{len(clusters)}",
                        entities=cluster_entities,
                        representative_entity=representative,
                        similarity_scores=similarity_scores,
                        merge_confidence=merge_confidence,
                        created_at=datetime.now()
                    )
                    
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Cluster creation failed: {e}")
            return []
    
    def _find_representative_entity(self, 
                                  entities: List[EntityCandidate],
                                  indices: List[int],
                                  similarity_matrix: np.ndarray) -> EntityCandidate:
        """Find the most representative entity in a cluster."""
        try:
            if len(entities) == 1:
                return entities[0]
            
            # Calculate average similarity for each entity
            best_entity = entities[0]
            best_score = 0.0
            
            for i, entity in enumerate(entities):
                idx = indices[i]
                
                # Calculate average similarity with other entities in cluster
                similarities = []
                for j, other_idx in enumerate(indices):
                    if i != j:
                        similarities.append(similarity_matrix[idx][other_idx])
                
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                
                # Prefer entities with more attributes and longer descriptions
                attribute_bonus = len(entity.attributes or {}) * 0.01
                description_bonus = len(entity.description or "") * 0.001
                
                total_score = avg_similarity + attribute_bonus + description_bonus
                
                if total_score > best_score:
                    best_score = total_score
                    best_entity = entity
            
            return best_entity
            
        except Exception as e:
            logger.error(f"Representative entity selection failed: {e}")
            return entities[0]
    
    def _calculate_merge_confidence(self, 
                                  entities: List[EntityCandidate],
                                  similarity_scores: List[SimilarityScore]) -> float:
        """Calculate confidence for merging entities in cluster."""
        try:
            if len(entities) <= 1:
                return 1.0
            
            if not similarity_scores:
                return 0.0
            
            # Average similarity score
            avg_similarity = sum(score.similarity_score for score in similarity_scores) / len(similarity_scores)
            
            # Penalty for large clusters (might be over-merging)
            size_penalty = max(0.0, (len(entities) - 2) * 0.1)
            
            # Bonus for consistent entity types
            types = [entity.entity_type for entity in entities]
            type_consistency = 1.0 if len(set(types)) == 1 else 0.5
            
            confidence = (avg_similarity * type_consistency) - size_penalty
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Merge confidence calculation failed: {e}")
            return 0.5
    
    async def _merge_cluster_entities(self, clusters: List[EntityCluster]) -> List[EntityCluster]:
        """Merge entities within each cluster."""
        try:
            merged_clusters = []
            
            for cluster in clusters:
                if len(cluster.entities) == 1:
                    # No merging needed
                    merged_clusters.append(cluster)
                    continue
                
                # Merge entities in cluster
                merged_entity = await self._merge_entities(cluster.entities)
                
                # Update cluster with merged entity
                cluster.representative_entity = merged_entity
                cluster.entities = [merged_entity]  # Replace with single merged entity
                
                merged_clusters.append(cluster)
            
            return merged_clusters
            
        except Exception as e:
            logger.error(f"Cluster entity merging failed: {e}")
            return clusters
    
    async def _merge_entities(self, entities: List[EntityCandidate]) -> EntityCandidate:
        """Merge multiple entities into a single representative entity."""
        try:
            if len(entities) == 1:
                return entities[0]
            
            # Use the representative entity as base
            base_entity = max(entities, key=lambda e: e.confidence)
            
            # Merge names (use the most common or longest)
            names = [entity.name for entity in entities]
            merged_name = max(names, key=len)  # Use longest name
            
            # Merge descriptions
            descriptions = [entity.description for entity in entities if entity.description]
            merged_description = " | ".join(descriptions) if descriptions else None
            
            # Merge attributes
            merged_attributes = {}
            for entity in entities:
                if entity.attributes:
                    for key, value in entity.attributes.items():
                        if key not in merged_attributes:
                            merged_attributes[key] = value
                        elif isinstance(value, list):
                            # Merge lists
                            if isinstance(merged_attributes[key], list):
                                merged_attributes[key].extend(value)
                            else:
                                merged_attributes[key] = [merged_attributes[key]] + value
            
            # Merge source chunks
            merged_source_chunks = []
            for entity in entities:
                if entity.source_chunks:
                    merged_source_chunks.extend(entity.source_chunks)
            
            # Calculate merged confidence (average)
            merged_confidence = sum(entity.confidence for entity in entities) / len(entities)
            
            # Create merged entity
            merged_entity = EntityCandidate(
                entity_id=f"merged_{base_entity.entity_id}",
                name=merged_name,
                entity_type=base_entity.entity_type,
                description=merged_description,
                attributes=merged_attributes,
                source_chunks=list(set(merged_source_chunks)),  # Remove duplicates
                confidence=merged_confidence,
                embedding=base_entity.embedding  # Use base entity's embedding
            )
            
            return merged_entity
            
        except Exception as e:
            logger.error(f"Entity merging failed: {e}")
            return entities[0]
    
    def _update_resolution_stats(self, 
                               original_entities: List[EntityCandidate],
                               clusters: List[EntityCluster]):
        """Update resolution statistics."""
        try:
            self.resolution_stats['total_entities_processed'] = len(original_entities)
            self.resolution_stats['total_clusters_created'] = len(clusters)
            
            total_merged = sum(len(cluster.entities) for cluster in clusters if len(cluster.entities) > 1)
            self.resolution_stats['total_entities_merged'] = total_merged
            
            if clusters:
                avg_cluster_size = sum(len(cluster.entities) for cluster in clusters) / len(clusters)
                self.resolution_stats['average_cluster_size'] = avg_cluster_size
            
        except Exception as e:
            logger.error(f"Stats update failed: {e}")
    
    async def get_resolution_stats(self) -> Dict[str, Any]:
        """Get entity resolution statistics."""
        try:
            return {
                **self.resolution_stats,
                'similarity_threshold': self.similarity_threshold,
                'type_thresholds': {k.value: v for k, v in self.type_thresholds.items()},
                'cache_sizes': {
                    'embedding_cache': len(self.embedding_cache),
                    'similarity_cache': len(self.similarity_cache)
                }
            }
            
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {'error': str(e)}
    
    def clear_caches(self):
        """Clear resolution caches."""
        try:
            self.embedding_cache.clear()
            self.similarity_cache.clear()
            logger.info("Entity resolution caches cleared")
            
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")


# Global instance
entity_resolver = EntityResolver()