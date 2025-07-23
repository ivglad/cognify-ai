"""
Knowledge graph builder with community detection using NetworkX and Leiden algorithm.
"""
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import json

import trio
import networkx as nx
import numpy as np
from graspologic.partition import leiden

from app.services.kg.entity_extractor import entity_extractor
from app.services.kg.relation_extractor import relation_extractor
from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Knowledge graph builder with community detection capabilities.
    """
    
    def __init__(self,
                 min_edge_weight: float = 0.1,
                 max_graph_size: int = 1000,
                 community_resolution: float = 1.0):
        """
        Initialize knowledge graph builder.
        
        Args:
            min_edge_weight: Minimum edge weight to include in graph
            max_graph_size: Maximum number of nodes in graph
            community_resolution: Resolution parameter for community detection
        """
        self.min_edge_weight = min_edge_weight
        self.max_graph_size = max_graph_size
        self.community_resolution = community_resolution
        
        self.entity_extractor = entity_extractor
        self.relation_extractor = relation_extractor
        self.cache = cache_manager
    
    async def build_knowledge_graph(self, 
                                  text: str,
                                  document_id: Optional[str] = None,
                                  use_cache: bool = True) -> Dict[str, Any]:
        """
        Build knowledge graph from text.
        
        Args:
            text: Input text
            document_id: Optional document ID for caching
            use_cache: Whether to use caching
            
        Returns:
            Knowledge graph with nodes, edges, and communities
        """
        if not text or not text.strip():
            return {'nodes': [], 'edges': [], 'communities': [], 'stats': {}}
        
        try:
            # Check cache
            if use_cache and document_id:
                cache_key = f"knowledge_graph:{document_id}"
                cached_graph = await self.cache.get(cache_key)
                if cached_graph:
                    logger.debug(f"Using cached knowledge graph for document {document_id}")
                    return cached_graph
            
            logger.debug(f"Building knowledge graph from text ({len(text)} chars)")
            
            # Extract entities and relations
            entities = await self.entity_extractor.extract_entities(text, document_id, use_cache)
            relations = await self.relation_extractor.extract_relations(text, entities, document_id, use_cache)
            
            if not entities:
                logger.debug("No entities found, cannot build graph")
                return {'nodes': [], 'edges': [], 'communities': [], 'stats': {}}
            
            # Build NetworkX graph
            graph = await self._build_networkx_graph(entities, relations)
            
            # Detect communities
            communities = await self._detect_communities(graph)
            
            # Generate community reports
            community_reports = await self._generate_community_reports(graph, communities, text)
            
            # Convert to serializable format
            graph_data = await self._serialize_graph(graph, communities, community_reports)
            
            # Calculate statistics
            stats = await self._calculate_graph_stats(graph, communities)
            graph_data['stats'] = stats
            
            # Cache results
            if use_cache and document_id:
                await self.cache.set(f"knowledge_graph:{document_id}", graph_data, ttl=86400)  # 24 hours
            
            logger.debug(f"Built knowledge graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Knowledge graph building failed: {e}")
            return {'nodes': [], 'edges': [], 'communities': [], 'stats': {}, 'error': str(e)}
    
    async def _build_networkx_graph(self, 
                                  entities: List[Dict[str, Any]], 
                                  relations: List[Dict[str, Any]]) -> nx.Graph:
        """Build NetworkX graph from entities and relations."""
        try:
            # Create directed graph
            G = nx.DiGraph()
            
            # Add entity nodes
            for entity in entities:
                node_id = entity['text']
                G.add_node(node_id, 
                          type=entity.get('type', 'Unknown'),
                          confidence=entity.get('final_confidence', entity.get('confidence', 0.5)),
                          methods=entity.get('methods', [entity.get('method', 'unknown')]),
                          occurrence_count=entity.get('occurrence_count', 1))
            
            # Add relation edges
            edge_weights = defaultdict(float)
            edge_data = defaultdict(list)
            
            for relation in relations:
                subject = relation.get('subject', '')
                object_node = relation.get('object', '')
                predicate = relation.get('predicate', 'related_to')
                confidence = relation.get('final_confidence', relation.get('confidence', 0.5))
                
                if subject in G.nodes and object_node in G.nodes and subject != object_node:
                    # Accumulate edge weights for multiple relations between same nodes
                    edge_key = (subject, object_node)
                    edge_weights[edge_key] += confidence
                    edge_data[edge_key].append({
                        'predicate': predicate,
                        'confidence': confidence,
                        'method': relation.get('method', 'unknown')
                    })
            
            # Add edges to graph
            for (source, target), weight in edge_weights.items():
                if weight >= self.min_edge_weight:
                    relations_data = edge_data[(source, target)]
                    
                    # Get primary relation (highest confidence)
                    primary_relation = max(relations_data, key=lambda x: x['confidence'])
                    
                    G.add_edge(source, target,
                             weight=weight,
                             primary_relation=primary_relation['predicate'],
                             relations=relations_data,
                             relation_count=len(relations_data))
            
            logger.debug(f"Built NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            return G
            
        except Exception as e:
            logger.error(f"NetworkX graph building failed: {e}")
            return nx.DiGraph()
    
    async def _detect_communities(self, graph: nx.Graph) -> List[Set[str]]:
        """Detect communities using Leiden algorithm."""
        try:
            if graph.number_of_nodes() < 3:
                # Too few nodes for community detection
                return [set(graph.nodes())] if graph.nodes() else []
            
            # Convert to undirected graph for community detection
            undirected_graph = graph.to_undirected()
            
            # Get adjacency matrix
            adjacency_matrix = nx.adjacency_matrix(undirected_graph, weight='weight')
            
            # Apply Leiden algorithm
            try:
                partition = await trio.to_thread.run_sync(
                    leiden,
                    adjacency_matrix,
                    resolution=self.community_resolution,
                    random_state=42
                )
                
                # Convert partition to communities
                communities = defaultdict(set)
                node_list = list(undirected_graph.nodes())
                
                for i, community_id in enumerate(partition):
                    if i < len(node_list):
                        communities[community_id].add(node_list[i])
                
                # Filter out single-node communities
                filtered_communities = [community for community in communities.values() if len(community) > 1]
                
                # Add single nodes as individual communities if needed
                all_community_nodes = set()
                for community in filtered_communities:
                    all_community_nodes.update(community)
                
                single_nodes = set(graph.nodes()) - all_community_nodes
                for node in single_nodes:
                    filtered_communities.append({node})
                
                logger.debug(f"Detected {len(filtered_communities)} communities")
                
                return filtered_communities
                
            except Exception as leiden_error:
                logger.warning(f"Leiden algorithm failed: {leiden_error}, using fallback method")
                return await self._fallback_community_detection(undirected_graph)
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return [set(graph.nodes())] if graph.nodes() else []
    
    async def _fallback_community_detection(self, graph: nx.Graph) -> List[Set[str]]:
        """Fallback community detection using connected components."""
        try:
            # Use connected components as communities
            communities = []
            
            for component in nx.connected_components(graph):
                if len(component) > 1:
                    communities.append(component)
                else:
                    # Single node community
                    communities.append(component)
            
            logger.debug(f"Fallback community detection found {len(communities)} communities")
            
            return communities
            
        except Exception as e:
            logger.error(f"Fallback community detection failed: {e}")
            return [set(graph.nodes())] if graph.nodes() else []
    
    async def _generate_community_reports(self, 
                                        graph: nx.Graph, 
                                        communities: List[Set[str]], 
                                        original_text: str) -> List[Dict[str, Any]]:
        """Generate reports for each community."""
        try:
            reports = []
            
            for i, community in enumerate(communities):
                if not community:
                    continue
                
                # Get community subgraph
                subgraph = graph.subgraph(community)
                
                # Analyze community
                report = await self._analyze_community(subgraph, i, original_text)
                reports.append(report)
            
            return reports
            
        except Exception as e:
            logger.error(f"Community report generation failed: {e}")
            return []
    
    async def _analyze_community(self, 
                               subgraph: nx.Graph, 
                               community_id: int, 
                               original_text: str) -> Dict[str, Any]:
        """Analyze a single community."""
        try:
            nodes = list(subgraph.nodes())
            edges = list(subgraph.edges())
            
            # Basic statistics
            node_count = len(nodes)
            edge_count = len(edges)
            density = nx.density(subgraph) if node_count > 1 else 0
            
            # Node types analysis
            node_types = Counter()
            for node in nodes:
                node_data = subgraph.nodes[node]
                node_type = node_data.get('type', 'Unknown')
                node_types[node_type] += 1
            
            # Central nodes (highest degree)
            if node_count > 0:
                degrees = dict(subgraph.degree())
                central_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
            else:
                central_nodes = []
            
            # Relation types analysis
            relation_types = Counter()
            for _, _, edge_data in subgraph.edges(data=True):
                primary_relation = edge_data.get('primary_relation', 'unknown')
                relation_types[primary_relation] += 1
            
            # Generate community summary
            summary = await self._generate_community_summary(nodes, relation_types, node_types)
            
            # Generate community theme
            theme = await self._generate_community_theme(nodes, node_types, original_text)
            
            return {
                'community_id': community_id,
                'nodes': nodes,
                'edges': [(u, v) for u, v in edges],
                'node_count': node_count,
                'edge_count': edge_count,
                'density': density,
                'node_types': dict(node_types),
                'relation_types': dict(relation_types),
                'central_nodes': [node for node, degree in central_nodes],
                'summary': summary,
                'theme': theme
            }
            
        except Exception as e:
            logger.error(f"Community analysis failed: {e}")
            return {
                'community_id': community_id,
                'nodes': list(subgraph.nodes()) if subgraph else [],
                'edges': [],
                'error': str(e)
            }
    
    async def _generate_community_summary(self, 
                                        nodes: List[str], 
                                        relation_types: Counter, 
                                        node_types: Counter) -> str:
        """Generate textual summary for community."""
        try:
            if not nodes:
                return "Empty community"
            
            # Basic description
            summary_parts = []
            
            # Node count and types
            if len(nodes) == 1:
                summary_parts.append(f"Single entity: {nodes[0]}")
            else:
                summary_parts.append(f"Community of {len(nodes)} entities")
                
                # Most common node type
                if node_types:
                    most_common_type = node_types.most_common(1)[0]
                    if most_common_type[1] > 1:
                        summary_parts.append(f"primarily {most_common_type[0]} entities")
            
            # Most common relation
            if relation_types:
                most_common_relation = relation_types.most_common(1)[0]
                summary_parts.append(f"connected mainly through '{most_common_relation[0]}' relationships")
            
            # Key entities
            if len(nodes) > 3:
                key_entities = nodes[:3]
                summary_parts.append(f"including {', '.join(key_entities)}")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            logger.error(f"Community summary generation failed: {e}")
            return f"Community with {len(nodes)} entities"
    
    async def _generate_community_theme(self, 
                                      nodes: List[str], 
                                      node_types: Counter, 
                                      original_text: str) -> str:
        """Generate thematic description for community."""
        try:
            if not nodes:
                return "Unknown"
            
            # Analyze node types to determine theme
            if not node_types:
                return "General"
            
            most_common_type = node_types.most_common(1)[0][0]
            
            # Theme mapping based on entity types
            theme_mapping = {
                'Person': 'People & Organizations',
                'Organization': 'Organizations & Institutions',
                'Location': 'Geographic Entities',
                'Technology': 'Technology & Tools',
                'Product': 'Products & Services',
                'Concept': 'Concepts & Ideas',
                'Event': 'Events & Activities',
                'Date': 'Temporal Information',
                'Money': 'Financial Information'
            }
            
            base_theme = theme_mapping.get(most_common_type, 'General')
            
            # Refine theme based on specific entities
            node_text = ' '.join(nodes).lower()
            
            # Technology themes
            if any(tech_word in node_text for tech_word in ['api', 'framework', 'algorithm', 'software', 'system']):
                return 'Technology & Software'
            
            # Business themes
            if any(biz_word in node_text for biz_word in ['company', 'business', 'market', 'industry']):
                return 'Business & Industry'
            
            # Academic themes
            if any(academic_word in node_text for academic_word in ['research', 'study', 'university', 'theory']):
                return 'Academic & Research'
            
            return base_theme
            
        except Exception as e:
            logger.error(f"Community theme generation failed: {e}")
            return "General"
    
    async def _serialize_graph(self, 
                             graph: nx.Graph, 
                             communities: List[Set[str]], 
                             community_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert graph to serializable format."""
        try:
            # Serialize nodes
            nodes = []
            for node_id in graph.nodes():
                node_data = graph.nodes[node_id]
                
                # Find community membership
                community_id = None
                for i, community in enumerate(communities):
                    if node_id in community:
                        community_id = i
                        break
                
                nodes.append({
                    'id': node_id,
                    'type': node_data.get('type', 'Unknown'),
                    'confidence': node_data.get('confidence', 0.5),
                    'methods': node_data.get('methods', []),
                    'occurrence_count': node_data.get('occurrence_count', 1),
                    'community_id': community_id,
                    'degree': graph.degree(node_id)
                })
            
            # Serialize edges
            edges = []
            for source, target in graph.edges():
                edge_data = graph.edges[source, target]
                
                edges.append({
                    'source': source,
                    'target': target,
                    'weight': edge_data.get('weight', 1.0),
                    'primary_relation': edge_data.get('primary_relation', 'related_to'),
                    'relations': edge_data.get('relations', []),
                    'relation_count': edge_data.get('relation_count', 1)
                })
            
            return {
                'nodes': nodes,
                'edges': edges,
                'communities': community_reports
            }
            
        except Exception as e:
            logger.error(f"Graph serialization failed: {e}")
            return {'nodes': [], 'edges': [], 'communities': []}
    
    async def _calculate_graph_stats(self, 
                                   graph: nx.Graph, 
                                   communities: List[Set[str]]) -> Dict[str, Any]:
        """Calculate graph statistics."""
        try:
            stats = {
                'node_count': graph.number_of_nodes(),
                'edge_count': graph.number_of_edges(),
                'density': nx.density(graph) if graph.number_of_nodes() > 1 else 0,
                'community_count': len(communities),
                'is_connected': nx.is_connected(graph.to_undirected()) if graph.number_of_nodes() > 0 else False
            }
            
            # Node type distribution
            node_types = Counter()
            for node_id in graph.nodes():
                node_type = graph.nodes[node_id].get('type', 'Unknown')
                node_types[node_type] += 1
            
            stats['node_type_distribution'] = dict(node_types)
            
            # Degree statistics
            if graph.number_of_nodes() > 0:
                degrees = [graph.degree(node) for node in graph.nodes()]
                stats['avg_degree'] = sum(degrees) / len(degrees)
                stats['max_degree'] = max(degrees)
                stats['min_degree'] = min(degrees)
            else:
                stats['avg_degree'] = 0
                stats['max_degree'] = 0
                stats['min_degree'] = 0
            
            # Community size statistics
            if communities:
                community_sizes = [len(community) for community in communities]
                stats['avg_community_size'] = sum(community_sizes) / len(community_sizes)
                stats['max_community_size'] = max(community_sizes)
                stats['min_community_size'] = min(community_sizes)
            else:
                stats['avg_community_size'] = 0
                stats['max_community_size'] = 0
                stats['min_community_size'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Graph statistics calculation failed: {e}")
            return {'error': str(e)}
    
    async def get_builder_stats(self) -> Dict[str, Any]:
        """Get graph builder statistics."""
        try:
            return {
                'min_edge_weight': self.min_edge_weight,
                'max_graph_size': self.max_graph_size,
                'community_resolution': self.community_resolution,
                'entity_extractor_available': self.entity_extractor is not None,
                'relation_extractor_available': self.relation_extractor is not None
            }
            
        except Exception as e:
            logger.error(f"Builder stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
knowledge_graph_builder = KnowledgeGraphBuilder()