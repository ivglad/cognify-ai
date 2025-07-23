"""
Knowledge graph integration with search system.
"""
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import networkx as nx

import trio

from app.services.kg.graph_builder import knowledge_graph_builder
from app.services.kg.entity_extractor import entity_extractor
from app.services.embeddings.embedding_service import embedding_service
from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class KGSearchIntegration:
    """
    Integration layer between knowledge graph and search system.
    """
    
    def __init__(self,
                 entity_expansion_limit: int = 5,
                 relation_hop_limit: int = 2,
                 kg_boost_factor: float = 0.3):
        """
        Initialize KG search integration.
        
        Args:
            entity_expansion_limit: Maximum entities to expand per query entity
            relation_hop_limit: Maximum hops to traverse in graph
            kg_boost_factor: Boost factor for KG-enhanced results
        """
        self.entity_expansion_limit = entity_expansion_limit
        self.relation_hop_limit = relation_hop_limit
        self.kg_boost_factor = kg_boost_factor
        
        self.graph_builder = knowledge_graph_builder
        self.entity_extractor = entity_extractor
        self.embedding_service = embedding_service
        self.cache = cache_manager
    
    async def enhance_search_query(self, 
                                 query: str,
                                 document_graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance search query using knowledge graph information.
        
        Args:
            query: Original search query
            document_graphs: List of knowledge graphs from documents
            
        Returns:
            Enhanced query information with entity expansions and context
        """
        try:
            # Extract entities from query
            query_entities = await self.entity_extractor.extract_entities(query)
            
            if not query_entities:
                return {
                    'original_query': query,
                    'enhanced_query': query,
                    'query_entities': [],
                    'expanded_entities': [],
                    'related_concepts': [],
                    'boost_terms': []
                }
            
            logger.debug(f"Found {len(query_entities)} entities in query")
            
            # Find related entities and concepts from graphs
            expanded_entities = await self._expand_query_entities(query_entities, document_graphs)
            
            # Generate related concepts
            related_concepts = await self._find_related_concepts(query_entities, document_graphs)
            
            # Create boost terms for search
            boost_terms = await self._generate_boost_terms(query_entities, expanded_entities, related_concepts)
            
            # Construct enhanced query
            enhanced_query = await self._construct_enhanced_query(query, boost_terms)
            
            return {
                'original_query': query,
                'enhanced_query': enhanced_query,
                'query_entities': [entity['text'] for entity in query_entities],
                'expanded_entities': expanded_entities,
                'related_concepts': related_concepts,
                'boost_terms': boost_terms
            }
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return {
                'original_query': query,
                'enhanced_query': query,
                'error': str(e)
            }
    
    async def _expand_query_entities(self, 
                                   query_entities: List[Dict[str, Any]], 
                                   document_graphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand query entities using knowledge graph relationships."""
        try:
            expanded_entities = []
            
            for query_entity in query_entities:
                entity_text = query_entity['text']
                
                # Find this entity in document graphs
                for graph_data in document_graphs:
                    # Find related entities through graph traversal
                    related = await self._find_related_entities(entity_text, graph_data)
                    
                    for related_entity in related:
                        if related_entity not in expanded_entities:
                            expanded_entities.append(related_entity)
                    
                    # Limit expansion per entity
                    if len(expanded_entities) >= self.entity_expansion_limit:
                        break
            
            return expanded_entities[:self.entity_expansion_limit]
            
        except Exception as e:
            logger.error(f"Entity expansion failed: {e}")
            return []
    
    async def _find_related_entities(self, 
                                   entity_text: str, 
                                   graph_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find entities related to given entity in a graph."""
        try:
            related_entities = []
            
            # Build NetworkX graph from data
            graph = await self._build_graph_from_data(graph_data)
            
            if entity_text not in graph.nodes():
                return []
            
            # Find entities within hop limit
            visited = set()
            queue = [(entity_text, 0)]  # (node, hop_count)
            
            while queue:
                current_node, hop_count = queue.pop(0)
                
                if current_node in visited or hop_count > self.relation_hop_limit:
                    continue
                
                visited.add(current_node)
                
                # Add current node if it's not the original entity
                if current_node != entity_text:
                    node_data = graph.nodes[current_node]
                    related_entities.append({
                        'text': current_node,
                        'type': node_data.get('type', 'Unknown'),
                        'confidence': node_data.get('confidence', 0.5),
                        'hop_distance': hop_count,
                        'relation_path': []  # Could be enhanced to track path
                    })
                
                # Add neighbors to queue
                if hop_count < self.relation_hop_limit:
                    for neighbor in graph.neighbors(current_node):
                        if neighbor not in visited:
                            queue.append((neighbor, hop_count + 1))
            
            # Sort by confidence and hop distance
            related_entities.sort(key=lambda x: (x['hop_distance'], -x['confidence']))
            
            return related_entities
            
        except Exception as e:
            logger.error(f"Related entity finding failed: {e}")
            return []
    
    async def _build_graph_from_data(self, graph_data: Dict[str, Any]) -> nx.Graph:
        """Build NetworkX graph from serialized graph data."""
        try:
            graph = nx.DiGraph()
            
            # Add nodes
            for node in graph_data.get('nodes', []):
                graph.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
            
            # Add edges
            for edge in graph_data.get('edges', []):
                graph.add_edge(edge['source'], edge['target'], 
                             **{k: v for k, v in edge.items() if k not in ['source', 'target']})
            
            return graph
            
        except Exception as e:
            logger.error(f"Graph building from data failed: {e}")
            return nx.DiGraph()
    
    async def _find_related_concepts(self, 
                                   query_entities: List[Dict[str, Any]], 
                                   document_graphs: List[Dict[str, Any]]) -> List[str]:
        """Find concepts related to query entities."""
        try:
            related_concepts = set()
            
            for graph_data in document_graphs:
                # Look for concept-type entities in communities with query entities
                for community in graph_data.get('communities', []):
                    community_nodes = set(community.get('nodes', []))
                    
                    # Check if any query entity is in this community
                    query_entity_texts = {entity['text'] for entity in query_entities}
                    if community_nodes & query_entity_texts:
                        # Add concept entities from this community
                        for node_data in graph_data.get('nodes', []):
                            if (node_data['id'] in community_nodes and 
                                node_data.get('type') in ['Concept', 'Technology', 'Method'] and
                                node_data['id'] not in query_entity_texts):
                                related_concepts.add(node_data['id'])
            
            return list(related_concepts)
            
        except Exception as e:
            logger.error(f"Related concept finding failed: {e}")
            return []
    
    async def _generate_boost_terms(self, 
                                  query_entities: List[Dict[str, Any]], 
                                  expanded_entities: List[Dict[str, Any]], 
                                  related_concepts: List[str]) -> List[Dict[str, Any]]:
        """Generate terms to boost in search results."""
        try:
            boost_terms = []
            
            # Add query entities with high boost
            for entity in query_entities:
                boost_terms.append({
                    'term': entity['text'],
                    'boost': 2.0,
                    'type': 'query_entity',
                    'entity_type': entity.get('type', 'Unknown')
                })
            
            # Add expanded entities with medium boost
            for entity in expanded_entities:
                boost_factor = 1.5 - (entity.get('hop_distance', 1) * 0.2)  # Decrease boost with distance
                boost_terms.append({
                    'term': entity['text'],
                    'boost': max(1.1, boost_factor),
                    'type': 'expanded_entity',
                    'entity_type': entity.get('type', 'Unknown'),
                    'hop_distance': entity.get('hop_distance', 1)
                })
            
            # Add related concepts with lower boost
            for concept in related_concepts:
                boost_terms.append({
                    'term': concept,
                    'boost': 1.2,
                    'type': 'related_concept',
                    'entity_type': 'Concept'
                })
            
            return boost_terms
            
        except Exception as e:
            logger.error(f"Boost term generation failed: {e}")
            return []
    
    async def _construct_enhanced_query(self, 
                                      original_query: str, 
                                      boost_terms: List[Dict[str, Any]]) -> str:
        """Construct enhanced query string."""
        try:
            # Start with original query
            enhanced_parts = [original_query]
            
            # Add boost terms
            for term_data in boost_terms:
                term = term_data['term']
                boost = term_data['boost']
                
                # Add term with boost notation (Elasticsearch style)
                if boost > 1.0:
                    enhanced_parts.append(f'"{term}"^{boost:.1f}')
            
            return ' '.join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Enhanced query construction failed: {e}")
            return original_query
    
    async def enhance_search_results(self, 
                                   search_results: List[Dict[str, Any]], 
                                   query_enhancement: Dict[str, Any],
                                   document_graphs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance search results using knowledge graph information.
        
        Args:
            search_results: Original search results
            query_enhancement: Query enhancement data
            document_graphs: Mapping of document_id to knowledge graph
            
        Returns:
            Enhanced search results with KG-based scoring
        """
        try:
            enhanced_results = []
            
            for result in search_results:
                document_id = result.get('document_id', '')
                
                # Get knowledge graph for this document
                graph_data = document_graphs.get(document_id, {})
                
                # Calculate KG-based relevance boost
                kg_boost = await self._calculate_kg_relevance_boost(result, query_enhancement, graph_data)
                
                # Apply boost to original score
                original_score = result.get('score', 0.0)
                enhanced_score = original_score * (1 + kg_boost * self.kg_boost_factor)
                
                # Add KG information to result
                enhanced_result = result.copy()
                enhanced_result.update({
                    'enhanced_score': enhanced_score,
                    'kg_boost': kg_boost,
                    'kg_entities': await self._extract_result_entities(result, graph_data),
                    'kg_relations': await self._extract_result_relations(result, graph_data, query_enhancement)
                })
                
                enhanced_results.append(enhanced_result)
            
            # Re-sort by enhanced score
            enhanced_results.sort(key=lambda x: x.get('enhanced_score', 0), reverse=True)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Search result enhancement failed: {e}")
            return search_results
    
    async def _calculate_kg_relevance_boost(self, 
                                         result: Dict[str, Any], 
                                         query_enhancement: Dict[str, Any], 
                                         graph_data: Dict[str, Any]) -> float:
        """Calculate relevance boost based on knowledge graph."""
        try:
            boost = 0.0
            
            # Get query entities and expanded entities
            query_entities = set(query_enhancement.get('query_entities', []))
            expanded_entities = {e['text'] for e in query_enhancement.get('expanded_entities', [])}
            related_concepts = set(query_enhancement.get('related_concepts', []))
            
            # Get entities in this document
            document_entities = {node['id'] for node in graph_data.get('nodes', [])}
            
            # Boost for query entity matches
            query_matches = query_entities & document_entities
            boost += len(query_matches) * 0.3
            
            # Boost for expanded entity matches
            expanded_matches = expanded_entities & document_entities
            boost += len(expanded_matches) * 0.2
            
            # Boost for related concept matches
            concept_matches = related_concepts & document_entities
            boost += len(concept_matches) * 0.1
            
            # Community relevance boost
            community_boost = await self._calculate_community_relevance(
                query_entities | expanded_entities, graph_data
            )
            boost += community_boost
            
            return min(1.0, boost)  # Cap boost at 1.0
            
        except Exception as e:
            logger.error(f"KG relevance boost calculation failed: {e}")
            return 0.0
    
    async def _calculate_community_relevance(self, 
                                           query_entities: Set[str], 
                                           graph_data: Dict[str, Any]) -> float:
        """Calculate relevance boost based on community structure."""
        try:
            boost = 0.0
            
            for community in graph_data.get('communities', []):
                community_nodes = set(community.get('nodes', []))
                
                # Check overlap with query entities
                overlap = query_entities & community_nodes
                if overlap:
                    # Boost based on community density and size
                    community_size = len(community_nodes)
                    density = community.get('density', 0)
                    
                    # Higher boost for smaller, denser communities
                    community_boost = (len(overlap) / len(query_entities)) * density * (1 / max(1, community_size / 10))
                    boost += community_boost
            
            return min(0.3, boost)  # Cap community boost
            
        except Exception as e:
            logger.error(f"Community relevance calculation failed: {e}")
            return 0.0
    
    async def _extract_result_entities(self, 
                                     result: Dict[str, Any], 
                                     graph_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relevant entities from result using KG."""
        try:
            # Get top entities from the document's knowledge graph
            nodes = graph_data.get('nodes', [])
            
            # Sort by confidence and degree
            sorted_nodes = sorted(nodes, 
                                key=lambda x: (x.get('confidence', 0), x.get('degree', 0)), 
                                reverse=True)
            
            # Return top entities
            return sorted_nodes[:5]
            
        except Exception as e:
            logger.error(f"Result entity extraction failed: {e}")
            return []
    
    async def _extract_result_relations(self, 
                                      result: Dict[str, Any], 
                                      graph_data: Dict[str, Any], 
                                      query_enhancement: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relevant relations from result using KG."""
        try:
            query_entities = set(query_enhancement.get('query_entities', []))
            edges = graph_data.get('edges', [])
            
            # Find relations involving query entities
            relevant_relations = []
            
            for edge in edges:
                source = edge.get('source', '')
                target = edge.get('target', '')
                
                if source in query_entities or target in query_entities:
                    relevant_relations.append({
                        'source': source,
                        'target': target,
                        'relation': edge.get('primary_relation', 'related_to'),
                        'weight': edge.get('weight', 1.0)
                    })
            
            # Sort by weight and return top relations
            relevant_relations.sort(key=lambda x: x['weight'], reverse=True)
            return relevant_relations[:3]
            
        except Exception as e:
            logger.error(f"Result relation extraction failed: {e}")
            return []
    
    async def get_entity_recommendations(self, 
                                       entity: str, 
                                       document_graphs: List[Dict[str, Any]], 
                                       limit: int = 10) -> List[Dict[str, Any]]:
        """Get entity recommendations based on knowledge graph."""
        try:
            recommendations = []
            
            for graph_data in document_graphs:
                # Find related entities
                related = await self._find_related_entities(entity, graph_data)
                recommendations.extend(related)
            
            # Remove duplicates and sort
            unique_recommendations = {}
            for rec in recommendations:
                key = rec['text']
                if key not in unique_recommendations or rec['confidence'] > unique_recommendations[key]['confidence']:
                    unique_recommendations[key] = rec
            
            # Sort by confidence and hop distance
            sorted_recommendations = sorted(
                unique_recommendations.values(),
                key=lambda x: (-x['confidence'], x['hop_distance'])
            )
            
            return sorted_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Entity recommendation failed: {e}")
            return []
    
    async def get_integration_stats(self) -> Dict[str, Any]:
        """Get KG search integration statistics."""
        try:
            return {
                'entity_expansion_limit': self.entity_expansion_limit,
                'relation_hop_limit': self.relation_hop_limit,
                'kg_boost_factor': self.kg_boost_factor,
                'graph_builder_available': self.graph_builder is not None,
                'entity_extractor_available': self.entity_extractor is not None,
                'embedding_service_available': self.embedding_service is not None
            }
            
        except Exception as e:
            logger.error(f"Integration stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
kg_search_integration = KGSearchIntegration()