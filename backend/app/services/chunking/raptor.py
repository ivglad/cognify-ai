"""
RAPTOR hierarchical clustering and summarization system.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import GaussianMixture
from sklearn.metrics import silhouette_score
import umap

import trio

from app.services.embeddings.embedding_service import embedding_service
from app.services.nlp.rag_tokenizer import rag_tokenizer
from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ClusterNode:
    """Represents a node in the RAPTOR hierarchy."""
    node_id: str
    level: int
    chunks: List[str]  # Chunk IDs
    summary: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    children: List['ClusterNode'] = None
    parent: Optional['ClusterNode'] = None
    cluster_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}


class RAPTORClustering:
    """
    RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) system.
    """
    
    def __init__(self,
                 max_clusters_per_level: int = 10,
                 min_cluster_size: int = 3,
                 max_levels: int = 5,
                 umap_n_components: int = 10,
                 umap_n_neighbors: int = 15,
                 umap_min_dist: float = 0.1):
        """
        Initialize RAPTOR clustering system.
        
        Args:
            max_clusters_per_level: Maximum number of clusters per level
            min_cluster_size: Minimum size for a cluster
            max_levels: Maximum hierarchy levels
            umap_n_components: UMAP dimensionality reduction components
            umap_n_neighbors: UMAP neighbors parameter
            umap_min_dist: UMAP minimum distance parameter
        """
        self.max_clusters_per_level = max_clusters_per_level
        self.min_cluster_size = min_cluster_size
        self.max_levels = max_levels
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        
        self.embedding_service = embedding_service
        self.tokenizer = rag_tokenizer
        self.cache = cache_manager
        
        # Initialize UMAP reducer
        self.umap_reducer = None
        self._initialize_umap()
    
    def _initialize_umap(self):
        """Initialize UMAP dimensionality reducer."""
        try:
            self.umap_reducer = umap.UMAP(
                n_components=self.umap_n_components,
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
                metric='cosine',
                random_state=42
            )
            logger.info("UMAP reducer initialized successfully")
        except Exception as e:
            logger.error(f"UMAP initialization failed: {e}")
            self.umap_reducer = None
    
    async def build_raptor_tree(self, 
                              chunks: List[Dict[str, Any]],
                              document_id: str) -> Optional[ClusterNode]:
        """
        Build RAPTOR hierarchical tree from document chunks.
        
        Args:
            chunks: List of document chunks with text and metadata
            document_id: Document identifier
            
        Returns:
            Root node of the RAPTOR tree
        """
        if not chunks:
            return None
        
        try:
            # Initialize tokenizer and embedding service
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            # Check cache for existing tree
            cache_key = f"raptor_tree:{document_id}"
            cached_tree = await self.cache.get(cache_key)
            if cached_tree:
                logger.info(f"Using cached RAPTOR tree for document {document_id}")
                return self._deserialize_tree(cached_tree)
            
            logger.info(f"Building RAPTOR tree for {len(chunks)} chunks")
            
            # Generate embeddings for all chunks
            chunk_embeddings = await self._generate_chunk_embeddings(chunks)
            
            if not chunk_embeddings:
                logger.error("Failed to generate embeddings for chunks")
                return None
            
            # Build hierarchical tree
            root_node = await self._build_hierarchy(chunks, chunk_embeddings, document_id)
            
            # Cache the tree
            if root_node:
                serialized_tree = self._serialize_tree(root_node)
                await self.cache.set(cache_key, serialized_tree, ttl=86400)  # 24 hours
            
            logger.info(f"RAPTOR tree built successfully with {self._count_nodes(root_node)} nodes")
            
            return root_node
            
        except Exception as e:
            logger.error(f"RAPTOR tree building failed: {e}")
            return None
    
    async def _generate_chunk_embeddings(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate embeddings for all chunks."""
        try:
            embeddings = []
            
            # Extract texts
            texts = [chunk.get('text', '') for chunk in chunks]
            
            # Generate embeddings in batches
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Generate embeddings for batch
                batch_embeddings = await self.embedding_service.generate_embeddings(batch_texts)
                
                if batch_embeddings:
                    embeddings.extend(batch_embeddings)
                else:
                    # Fallback: generate individual embeddings
                    for text in batch_texts:
                        embedding = await self.embedding_service.generate_embedding(text)
                        if embedding is not None:
                            embeddings.append(embedding)
                        else:
                            # Use zero vector as fallback
                            embeddings.append(np.zeros(self.embedding_service.embedding_dim))
            
            logger.debug(f"Generated {len(embeddings)} embeddings for chunks")
            return embeddings
            
        except Exception as e:
            logger.error(f"Chunk embedding generation failed: {e}")
            return []
    
    async def _build_hierarchy(self, 
                             chunks: List[Dict[str, Any]], 
                             embeddings: List[np.ndarray],
                             document_id: str) -> Optional[ClusterNode]:
        """Build hierarchical clustering tree."""
        try:
            # Create leaf nodes for chunks
            current_nodes = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                node = ClusterNode(
                    node_id=f"{document_id}_chunk_{i}",
                    level=0,
                    chunks=[chunk.get('chunk_id', f"chunk_{i}")],
                    summary=chunk.get('text', ''),
                    embedding=embedding,
                    metadata={
                        'chunk_index': i,
                        'original_chunk': True,
                        'token_count': chunk.get('metadata', {}).get('token_count', 0)
                    }
                )
                current_nodes.append(node)
            
            # Build hierarchy level by level
            level = 1
            while len(current_nodes) > 1 and level <= self.max_levels:
                logger.debug(f"Building level {level} with {len(current_nodes)} nodes")
                
                # Cluster current level
                next_level_nodes = await self._cluster_level(current_nodes, level, document_id)
                
                if not next_level_nodes or len(next_level_nodes) >= len(current_nodes):
                    # No improvement in clustering, stop
                    break
                
                current_nodes = next_level_nodes
                level += 1
            
            # Return root node (or create one if multiple nodes remain)
            if len(current_nodes) == 1:
                return current_nodes[0]
            elif len(current_nodes) > 1:
                # Create root node combining all remaining nodes
                root_node = await self._create_root_node(current_nodes, document_id)
                return root_node
            else:
                return None
                
        except Exception as e:
            logger.error(f"Hierarchy building failed: {e}")
            return None
    
    async def _cluster_level(self, 
                           nodes: List[ClusterNode], 
                           level: int,
                           document_id: str) -> List[ClusterNode]:
        """Cluster nodes at a specific level."""
        try:
            if len(nodes) < self.min_cluster_size:
                return nodes
            
            # Extract embeddings
            embeddings = np.array([node.embedding for node in nodes])
            
            # Apply dimensionality reduction if needed
            if embeddings.shape[1] > self.umap_n_components and self.umap_reducer:
                try:
                    reduced_embeddings = await trio.to_thread.run_sync(
                        self.umap_reducer.fit_transform, embeddings
                    )
                except Exception as e:
                    logger.warning(f"UMAP reduction failed: {e}, using original embeddings")
                    reduced_embeddings = embeddings
            else:
                reduced_embeddings = embeddings
            
            # Determine optimal number of clusters
            optimal_clusters = await self._find_optimal_clusters(reduced_embeddings)
            
            if optimal_clusters <= 1:
                return nodes
            
            # Perform clustering
            clusters = await self._perform_clustering(reduced_embeddings, optimal_clusters)
            
            # Create cluster nodes
            cluster_nodes = await self._create_cluster_nodes(nodes, clusters, level, document_id)
            
            return cluster_nodes
            
        except Exception as e:
            logger.error(f"Level clustering failed: {e}")
            return nodes
    
    async def _find_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette analysis."""
        try:
            n_samples = embeddings.shape[0]
            max_clusters = min(self.max_clusters_per_level, n_samples // self.min_cluster_size)
            
            if max_clusters <= 1:
                return 1
            
            best_score = -1
            best_n_clusters = 2
            
            # Try different numbers of clusters
            for n_clusters in range(2, max_clusters + 1):
                try:
                    # Fit Gaussian Mixture Model
                    gmm = GaussianMixture(
                        n_components=n_clusters,
                        covariance_type='full',
                        random_state=42
                    )
                    
                    cluster_labels = await trio.to_thread.run_sync(
                        gmm.fit_predict, embeddings
                    )
                    
                    # Calculate silhouette score
                    score = await trio.to_thread.run_sync(
                        silhouette_score, embeddings, cluster_labels
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                        
                except Exception as e:
                    logger.debug(f"Clustering with {n_clusters} clusters failed: {e}")
                    continue
            
            logger.debug(f"Optimal clusters: {best_n_clusters} (score: {best_score:.3f})")
            return best_n_clusters
            
        except Exception as e:
            logger.error(f"Optimal cluster finding failed: {e}")
            return 2
    
    async def _perform_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform Gaussian Mixture clustering."""
        try:
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type='full',
                random_state=42
            )
            
            cluster_labels = await trio.to_thread.run_sync(
                gmm.fit_predict, embeddings
            )
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            # Fallback: assign all to one cluster
            return np.zeros(embeddings.shape[0], dtype=int)
    
    async def _create_cluster_nodes(self, 
                                  nodes: List[ClusterNode], 
                                  cluster_labels: np.ndarray,
                                  level: int,
                                  document_id: str) -> List[ClusterNode]:
        """Create cluster nodes from clustering results."""
        try:
            cluster_nodes = []
            unique_labels = np.unique(cluster_labels)
            
            for cluster_id in unique_labels:
                # Get nodes in this cluster
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_node_list = [nodes[i] for i in cluster_indices]
                
                if len(cluster_node_list) < self.min_cluster_size:
                    # Skip small clusters
                    continue
                
                # Create cluster node
                cluster_node = await self._create_single_cluster_node(
                    cluster_node_list, level, cluster_id, document_id
                )
                
                if cluster_node:
                    cluster_nodes.append(cluster_node)
            
            # Add any remaining nodes that weren't clustered
            for i, node in enumerate(nodes):
                if cluster_labels[i] not in unique_labels or \
                   len(np.where(cluster_labels == cluster_labels[i])[0]) < self.min_cluster_size:
                    cluster_nodes.append(node)
            
            return cluster_nodes
            
        except Exception as e:
            logger.error(f"Cluster node creation failed: {e}")
            return nodes
    
    async def _create_single_cluster_node(self, 
                                        child_nodes: List[ClusterNode],
                                        level: int,
                                        cluster_id: int,
                                        document_id: str) -> Optional[ClusterNode]:
        """Create a single cluster node from child nodes."""
        try:
            # Collect all chunks from child nodes
            all_chunks = []
            for child in child_nodes:
                all_chunks.extend(child.chunks)
            
            # Calculate cluster embedding (centroid)
            embeddings = np.array([child.embedding for child in child_nodes])
            cluster_embedding = np.mean(embeddings, axis=0)
            
            # Generate summary (will be implemented in next task)
            cluster_summary = await self._generate_cluster_summary(child_nodes)
            
            # Calculate cluster quality score
            cluster_score = await self._calculate_cluster_score(child_nodes, cluster_embedding)
            
            # Create cluster node
            cluster_node = ClusterNode(
                node_id=f"{document_id}_cluster_L{level}_C{cluster_id}",
                level=level,
                chunks=all_chunks,
                summary=cluster_summary,
                embedding=cluster_embedding,
                children=child_nodes,
                cluster_score=cluster_score,
                metadata={
                    'cluster_id': cluster_id,
                    'child_count': len(child_nodes),
                    'total_chunks': len(all_chunks),
                    'level': level
                }
            )
            
            # Set parent references
            for child in child_nodes:
                child.parent = cluster_node
            
            return cluster_node
            
        except Exception as e:
            logger.error(f"Single cluster node creation failed: {e}")
            return None
    
    async def _generate_cluster_summary(self, child_nodes: List[ClusterNode]) -> str:
        """Generate summary for a cluster (placeholder - will be replaced by hierarchical summarizer)."""
        try:
            # Placeholder implementation - the actual summarization will be done
            # by HierarchicalSummarizer after the tree is built
            summaries = []
            for child in child_nodes:
                if child.summary:
                    # Take first sentence
                    sentences = child.summary.split('.')
                    if sentences:
                        summaries.append(sentences[0].strip())
            
            if summaries:
                return '. '.join(summaries[:3]) + '.'  # Limit to 3 sentences
            else:
                return f"Cluster of {len(child_nodes)} related content sections."
                
        except Exception as e:
            logger.error(f"Cluster summary generation failed: {e}")
            return f"Cluster summary (level {child_nodes[0].level + 1 if child_nodes else 0})"
    
    async def _calculate_cluster_score(self, 
                                     child_nodes: List[ClusterNode], 
                                     cluster_embedding: np.ndarray) -> float:
        """Calculate cluster quality score."""
        try:
            if not child_nodes:
                return 0.0
            
            # Calculate intra-cluster similarity
            similarities = []
            for child in child_nodes:
                similarity = np.dot(child.embedding, cluster_embedding) / (
                    np.linalg.norm(child.embedding) * np.linalg.norm(cluster_embedding)
                )
                similarities.append(similarity)
            
            # Return average similarity
            return float(np.mean(similarities))
            
        except Exception as e:
            logger.error(f"Cluster score calculation failed: {e}")
            return 0.0
    
    async def _create_root_node(self, nodes: List[ClusterNode], document_id: str) -> ClusterNode:
        """Create root node from remaining nodes."""
        try:
            # Collect all chunks
            all_chunks = []
            for node in nodes:
                all_chunks.extend(node.chunks)
            
            # Calculate root embedding
            embeddings = np.array([node.embedding for node in nodes])
            root_embedding = np.mean(embeddings, axis=0)
            
            # Generate root summary
            root_summary = await self._generate_cluster_summary(nodes)
            
            # Create root node
            root_node = ClusterNode(
                node_id=f"{document_id}_root",
                level=max(node.level for node in nodes) + 1,
                chunks=all_chunks,
                summary=root_summary,
                embedding=root_embedding,
                children=nodes,
                metadata={
                    'is_root': True,
                    'child_count': len(nodes),
                    'total_chunks': len(all_chunks)
                }
            )
            
            # Set parent references
            for child in nodes:
                child.parent = root_node
            
            return root_node
            
        except Exception as e:
            logger.error(f"Root node creation failed: {e}")
            # Return first node as fallback
            return nodes[0] if nodes else None
    
    def _count_nodes(self, root: Optional[ClusterNode]) -> int:
        """Count total nodes in tree."""
        if not root:
            return 0
        
        count = 1
        for child in root.children:
            count += self._count_nodes(child)
        
        return count
    
    def _serialize_tree(self, root: ClusterNode) -> Dict[str, Any]:
        """Serialize tree for caching."""
        try:
            return {
                'node_id': root.node_id,
                'level': root.level,
                'chunks': root.chunks,
                'summary': root.summary,
                'embedding': root.embedding.tolist() if root.embedding is not None else None,
                'cluster_score': root.cluster_score,
                'metadata': root.metadata,
                'children': [self._serialize_tree(child) for child in root.children]
            }
        except Exception as e:
            logger.error(f"Tree serialization failed: {e}")
            return {}
    
    def _deserialize_tree(self, data: Dict[str, Any]) -> Optional[ClusterNode]:
        """Deserialize tree from cache."""
        try:
            node = ClusterNode(
                node_id=data['node_id'],
                level=data['level'],
                chunks=data['chunks'],
                summary=data['summary'],
                embedding=np.array(data['embedding']) if data['embedding'] else None,
                cluster_score=data['cluster_score'],
                metadata=data['metadata']
            )
            
            # Deserialize children
            for child_data in data['children']:
                child_node = self._deserialize_tree(child_data)
                if child_node:
                    child_node.parent = node
                    node.children.append(child_node)
            
            return node
            
        except Exception as e:
            logger.error(f"Tree deserialization failed: {e}")
            return None
    
    async def get_clustering_stats(self) -> Dict[str, Any]:
        """Get clustering system statistics."""
        try:
            return {
                'max_clusters_per_level': self.max_clusters_per_level,
                'min_cluster_size': self.min_cluster_size,
                'max_levels': self.max_levels,
                'umap_components': self.umap_n_components,
                'umap_neighbors': self.umap_n_neighbors,
                'umap_min_dist': self.umap_min_dist,
                'umap_initialized': self.umap_reducer is not None,
                'embedding_service_available': self.embedding_service is not None,
                'tokenizer_initialized': self.tokenizer._initialized if hasattr(self.tokenizer, '_initialized') else False
            }
            
        except Exception as e:
            logger.error(f"Clustering stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
raptor_clustering = RAPTORClustering()