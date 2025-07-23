"""
RAPTOR manager that integrates clustering and summarization.
"""
import logging
from typing import List, Dict, Any, Optional

import trio

from app.services.chunking.raptor import RAPTORClustering, ClusterNode, raptor_clustering
from app.services.chunking.hierarchical_summarizer import HierarchicalSummarizer, hierarchical_summarizer
from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class RAPTORManager:
    """
    Manager that coordinates RAPTOR clustering and hierarchical summarization.
    """
    
    def __init__(self):
        self.clustering = raptor_clustering
        self.summarizer = hierarchical_summarizer
        self.cache = cache_manager
    
    async def build_complete_raptor_tree(self, 
                                       chunks: List[Dict[str, Any]],
                                       document_id: str,
                                       use_summarization: bool = True) -> Optional[ClusterNode]:
        """
        Build complete RAPTOR tree with clustering and summarization.
        
        Args:
            chunks: List of document chunks with text and metadata
            document_id: Document identifier
            use_summarization: Whether to apply hierarchical summarization
            
        Returns:
            Root node of the complete RAPTOR tree
        """
        if not chunks:
            return None
        
        try:
            logger.info(f"Building complete RAPTOR tree for document {document_id}")
            
            # Check cache for complete tree
            cache_key = f"raptor_complete:{document_id}:{use_summarization}"
            cached_tree = await self.cache.get(cache_key)
            if cached_tree:
                logger.info(f"Using cached complete RAPTOR tree for document {document_id}")
                return self.clustering._deserialize_tree(cached_tree)
            
            # Step 1: Build clustering tree
            logger.info("Step 1: Building clustering hierarchy")
            root_node = await self.clustering.build_raptor_tree(chunks, document_id)
            
            if not root_node:
                logger.error("Failed to build clustering tree")
                return None
            
            # Step 2: Apply hierarchical summarization if requested
            if use_summarization:
                logger.info("Step 2: Applying hierarchical summarization")
                root_node = await self.summarizer.summarize_raptor_tree(root_node)
            
            # Step 3: Validate and optimize tree
            logger.info("Step 3: Validating and optimizing tree")
            validation_results = await self._validate_raptor_tree(root_node)
            
            if validation_results.get('critical_issues'):
                logger.warning(f"RAPTOR tree has critical issues: {validation_results['critical_issues']}")
            
            # Cache the complete tree
            if root_node:
                serialized_tree = self.clustering._serialize_tree(root_node)
                await self.cache.set(cache_key, serialized_tree, ttl=86400)  # 24 hours
            
            logger.info(f"Complete RAPTOR tree built successfully")
            
            return root_node
            
        except Exception as e:
            logger.error(f"Complete RAPTOR tree building failed: {e}")
            return None
    
    async def search_raptor_tree(self, 
                                root_node: ClusterNode,
                                query: str,
                                max_results: int = 10,
                                similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search RAPTOR tree for relevant content.
        
        Args:
            root_node: Root of RAPTOR tree
            query: Search query
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of relevant nodes with similarity scores
        """
        try:
            # Generate query embedding
            from app.services.embeddings.embedding_service import embedding_service
            query_embedding = await embedding_service.generate_embedding(query)
            
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search tree at all levels
            results = []
            await self._search_node_recursive(root_node, query_embedding, results, similarity_threshold)
            
            # Sort by similarity score
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top results
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"RAPTOR tree search failed: {e}")
            return []
    
    async def _search_node_recursive(self, 
                                   node: ClusterNode,
                                   query_embedding,
                                   results: List[Dict[str, Any]],
                                   threshold: float):
        """Recursively search nodes in tree."""
        try:
            if node.embedding is not None:
                # Calculate similarity
                import numpy as np
                similarity = np.dot(node.embedding, query_embedding) / (
                    np.linalg.norm(node.embedding) * np.linalg.norm(query_embedding)
                )
                
                if similarity >= threshold:
                    results.append({
                        'node_id': node.node_id,
                        'level': node.level,
                        'similarity': float(similarity),
                        'summary': node.summary,
                        'chunks': node.chunks,
                        'metadata': node.metadata
                    })
            
            # Search children
            for child in node.children:
                await self._search_node_recursive(child, query_embedding, results, threshold)
                
        except Exception as e:
            logger.error(f"Node search failed for {node.node_id}: {e}")
    
    async def get_node_context(self, 
                             root_node: ClusterNode,
                             target_node_id: str,
                             include_siblings: bool = True,
                             include_children: bool = True) -> Dict[str, Any]:
        """
        Get contextual information for a specific node.
        
        Args:
            root_node: Root of RAPTOR tree
            target_node_id: ID of target node
            include_siblings: Whether to include sibling nodes
            include_children: Whether to include child nodes
            
        Returns:
            Context information for the node
        """
        try:
            # Find target node
            target_node = await self._find_node_by_id(root_node, target_node_id)
            
            if not target_node:
                return {'error': f'Node {target_node_id} not found'}
            
            context = {
                'target_node': {
                    'node_id': target_node.node_id,
                    'level': target_node.level,
                    'summary': target_node.summary,
                    'chunks': target_node.chunks,
                    'metadata': target_node.metadata
                },
                'path_to_root': [],
                'siblings': [],
                'children': []
            }
            
            # Build path to root
            current = target_node.parent
            while current:
                context['path_to_root'].append({
                    'node_id': current.node_id,
                    'level': current.level,
                    'summary': current.summary
                })
                current = current.parent
            
            # Get siblings
            if include_siblings and target_node.parent:
                for sibling in target_node.parent.children:
                    if sibling.node_id != target_node_id:
                        context['siblings'].append({
                            'node_id': sibling.node_id,
                            'level': sibling.level,
                            'summary': sibling.summary
                        })
            
            # Get children
            if include_children:
                for child in target_node.children:
                    context['children'].append({
                        'node_id': child.node_id,
                        'level': child.level,
                        'summary': child.summary,
                        'chunks': child.chunks
                    })
            
            return context
            
        except Exception as e:
            logger.error(f"Context retrieval failed for node {target_node_id}: {e}")
            return {'error': str(e)}
    
    async def _find_node_by_id(self, root_node: ClusterNode, node_id: str) -> Optional[ClusterNode]:
        """Find node by ID in tree."""
        if root_node.node_id == node_id:
            return root_node
        
        for child in root_node.children:
            result = await self._find_node_by_id(child, node_id)
            if result:
                return result
        
        return None
    
    async def _validate_raptor_tree(self, root_node: ClusterNode) -> Dict[str, Any]:
        """Validate RAPTOR tree structure and quality."""
        try:
            validation = {
                'total_nodes': 0,
                'leaf_nodes': 0,
                'max_depth': 0,
                'nodes_with_summaries': 0,
                'nodes_with_embeddings': 0,
                'average_cluster_score': 0.0,
                'issues': [],
                'critical_issues': []
            }
            
            await self._validate_node_recursive(root_node, validation, 0)
            
            # Calculate averages and check for issues
            if validation['total_nodes'] > 0:
                summary_coverage = validation['nodes_with_summaries'] / validation['total_nodes']
                embedding_coverage = validation['nodes_with_embeddings'] / validation['total_nodes']
                
                if summary_coverage < 0.9:
                    validation['issues'].append(f"Low summary coverage: {summary_coverage:.2%}")
                
                if embedding_coverage < 0.9:
                    validation['critical_issues'].append(f"Low embedding coverage: {embedding_coverage:.2%}")
                
                if validation['max_depth'] < 2:
                    validation['issues'].append("Tree depth is very shallow")
                elif validation['max_depth'] > 10:
                    validation['issues'].append("Tree depth might be too deep")
            
            return validation
            
        except Exception as e:
            logger.error(f"Tree validation failed: {e}")
            return {'error': str(e)}
    
    async def _validate_node_recursive(self, 
                                     node: ClusterNode, 
                                     validation: Dict[str, Any],
                                     depth: int):
        """Recursively validate nodes."""
        try:
            validation['total_nodes'] += 1
            validation['max_depth'] = max(validation['max_depth'], depth)
            
            if not node.children:
                validation['leaf_nodes'] += 1
            
            if node.summary and node.summary.strip():
                validation['nodes_with_summaries'] += 1
            
            if node.embedding is not None:
                validation['nodes_with_embeddings'] += 1
            
            if hasattr(node, 'cluster_score') and node.cluster_score > 0:
                validation['average_cluster_score'] += node.cluster_score
            
            # Validate children
            for child in node.children:
                await self._validate_node_recursive(child, validation, depth + 1)
                
        except Exception as e:
            logger.error(f"Node validation failed for {node.node_id}: {e}")
    
    async def get_tree_statistics(self, root_node: ClusterNode) -> Dict[str, Any]:
        """Get comprehensive statistics for RAPTOR tree."""
        try:
            stats = {
                'structure': {
                    'total_nodes': 0,
                    'leaf_nodes': 0,
                    'internal_nodes': 0,
                    'max_depth': 0,
                    'average_branching_factor': 0.0
                },
                'content': {
                    'total_chunks': len(root_node.chunks) if root_node.chunks else 0,
                    'nodes_with_summaries': 0,
                    'average_summary_length': 0.0,
                    'summary_lengths': []
                },
                'quality': {
                    'nodes_with_embeddings': 0,
                    'average_cluster_score': 0.0,
                    'cluster_scores': []
                },
                'levels': {}
            }
            
            await self._collect_tree_stats(root_node, stats, 0)
            
            # Calculate derived statistics
            if stats['structure']['total_nodes'] > 0:
                internal_nodes = stats['structure']['internal_nodes']
                if internal_nodes > 0:
                    total_children = sum(len(node.children) for node in await self._get_all_internal_nodes(root_node))
                    stats['structure']['average_branching_factor'] = total_children / internal_nodes
            
            if stats['content']['summary_lengths']:
                stats['content']['average_summary_length'] = sum(stats['content']['summary_lengths']) / len(stats['content']['summary_lengths'])
            
            if stats['quality']['cluster_scores']:
                stats['quality']['average_cluster_score'] = sum(stats['quality']['cluster_scores']) / len(stats['quality']['cluster_scores'])
            
            return stats
            
        except Exception as e:
            logger.error(f"Tree statistics collection failed: {e}")
            return {'error': str(e)}
    
    async def _collect_tree_stats(self, 
                                node: ClusterNode, 
                                stats: Dict[str, Any],
                                depth: int):
        """Recursively collect tree statistics."""
        try:
            stats['structure']['total_nodes'] += 1
            stats['structure']['max_depth'] = max(stats['structure']['max_depth'], depth)
            
            # Level statistics
            if depth not in stats['levels']:
                stats['levels'][depth] = {'nodes': 0, 'avg_children': 0}
            stats['levels'][depth]['nodes'] += 1
            
            if node.children:
                stats['structure']['internal_nodes'] += 1
                stats['levels'][depth]['avg_children'] += len(node.children)
            else:
                stats['structure']['leaf_nodes'] += 1
            
            # Content statistics
            if node.summary and node.summary.strip():
                stats['content']['nodes_with_summaries'] += 1
                # Count tokens in summary
                summary_length = len(node.summary.split())
                stats['content']['summary_lengths'].append(summary_length)
            
            # Quality statistics
            if node.embedding is not None:
                stats['quality']['nodes_with_embeddings'] += 1
            
            if hasattr(node, 'cluster_score') and node.cluster_score > 0:
                stats['quality']['cluster_scores'].append(node.cluster_score)
            
            # Process children
            for child in node.children:
                await self._collect_tree_stats(child, stats, depth + 1)
                
        except Exception as e:
            logger.error(f"Stats collection failed for node {node.node_id}: {e}")
    
    async def _get_all_internal_nodes(self, root_node: ClusterNode) -> List[ClusterNode]:
        """Get all internal (non-leaf) nodes."""
        internal_nodes = []
        
        if root_node.children:
            internal_nodes.append(root_node)
            for child in root_node.children:
                internal_nodes.extend(await self._get_all_internal_nodes(child))
        
        return internal_nodes
    
    async def get_manager_stats(self) -> Dict[str, Any]:
        """Get RAPTOR manager statistics."""
        try:
            clustering_stats = await self.clustering.get_clustering_stats()
            summarizer_stats = await self.summarizer.get_summarizer_stats()
            
            return {
                'clustering': clustering_stats,
                'summarization': summarizer_stats,
                'cache_available': self.cache is not None
            }
            
        except Exception as e:
            logger.error(f"Manager stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
raptor_manager = RAPTORManager()