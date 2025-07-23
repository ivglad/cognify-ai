"""
Basic retriever service combining sparse and dense search.
"""
import logging
from typing import List, Dict, Any, Optional
import time

import trio

from app.services.search.sparse_retriever import SparseRetriever
from app.services.embeddings.vector_store import vector_store
from app.services.embeddings.embedding_service import embedding_service
from app.services.search.reranking_manager import reranking_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class BasicRetriever:
    """
    Basic retriever combining sparse (Elasticsearch) and dense (vector) search.
    """
    
    def __init__(self):
        self.sparse_retriever = SparseRetriever()
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self._initialized = False
        
    async def initialize(self):
        """Initialize the retriever."""
        if self._initialized:
            return
            
        try:
            # Initialize vector store
            if not self.vector_store._initialized:
                await self.vector_store.initialize()
            
            # Initialize embedding service
            if not self.embedding_service._initialized:
                await self.embedding_service.initialize()
            
            self._initialized = True
            logger.info("BasicRetriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize BasicRetriever: {e}")
            raise
    
    async def search(self, 
                    query: str,
                    document_ids: Optional[List[str]] = None,
                    search_type: str = "hybrid",
                    top_k: int = 20,
                    filters: Optional[Dict[str, Any]] = None,
                    enable_reranking: bool = True,
                    rerank_type: str = "general") -> Dict[str, Any]:
        """
        Perform search using specified method.
        
        Args:
            query: Search query
            document_ids: Filter by document IDs
            search_type: "sparse", "dense", or "hybrid"
            top_k: Number of results to return
            filters: Additional filters
            
        Returns:
            Search results with metadata
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Get initial search results
            if search_type == "sparse":
                results = await self._sparse_search(query, document_ids, top_k * 2, filters)  # Get more for reranking
            elif search_type == "dense":
                results = await self._dense_search(query, document_ids, top_k * 2, filters)
            elif search_type == "hybrid":
                results = await self._hybrid_search(query, document_ids, top_k * 2, filters)
            else:
                raise ValueError(f"Unknown search type: {search_type}")
            
            # Apply reranking if enabled and we have results
            if enable_reranking and results.get("results") and len(results["results"]) > 1:
                rerank_start = time.time()
                
                try:
                    rerank_result = await reranking_manager.rerank_results(
                        query=query,
                        search_results=results["results"],
                        method="auto",  # Auto-select best method
                        rerank_type=rerank_type,
                        top_k=top_k
                    )
                    
                    results["results"] = rerank_result["results"]
                    results["reranking_time"] = rerank_result["rerank_time"]
                    results["reranking_enabled"] = True
                    results["rerank_method"] = rerank_result["method_used"]
                    results["rerank_type"] = rerank_type
                    results["query_type"] = rerank_result.get("query_type")
                    
                    logger.info(f"Reranking completed: {len(rerank_result['results'])} results in {results['reranking_time']:.3f}s using {rerank_result['method_used']}")
                    
                except Exception as rerank_error:
                    logger.warning(f"Reranking failed, using original results: {rerank_error}")
                    results["results"] = results["results"][:top_k]  # Fallback to original top_k
                    results["reranking_enabled"] = False
                    results["reranking_error"] = str(rerank_error)
            else:
                results["results"] = results["results"][:top_k]
                results["reranking_enabled"] = False
            
            # Add timing information
            results["search_time"] = time.time() - start_time
            results["search_type"] = search_type
            
            logger.info(f"Search completed: {len(results.get('results', []))} results in {results['search_time']:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "results": [],
                "total": 0,
                "search_time": time.time() - start_time,
                "search_type": search_type,
                "error": str(e)
            }
    
    async def _sparse_search(self, 
                           query: str,
                           document_ids: Optional[List[str]],
                           top_k: int,
                           filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform sparse (keyword) search."""
        try:
            results = await self.sparse_retriever.search(
                query=query,
                document_ids=document_ids,
                filters=filters,
                size=top_k
            )
            
            # Normalize results format
            normalized_results = []
            for result in results.get("results", []):
                normalized_results.append({
                    "chunk_id": result.get("chunk_id"),
                    "document_id": result.get("document_id"),
                    "content": result.get("content"),
                    "score": result.get("score", 0.0),
                    "search_type": "sparse",
                    "highlights": result.get("highlights", {}),
                    "metadata": {
                        "chunk_type": result.get("chunk_type"),
                        "page_number": result.get("page_number"),
                        "document_name": result.get("document_name")
                    }
                })
            
            return {
                "results": normalized_results,
                "total": results.get("total", 0),
                "max_score": results.get("max_score", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return {"results": [], "total": 0, "max_score": 0.0}
    
    async def _dense_search(self, 
                          query: str,
                          document_ids: Optional[List[str]],
                          top_k: int,
                          filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform dense (vector) search."""
        try:
            results = await self.vector_store.search_similar_chunks(
                query=query,
                document_ids=document_ids,
                top_k=top_k,
                similarity_threshold=0.0
            )
            
            # Normalize results format
            normalized_results = []
            for result in results:
                normalized_results.append({
                    "chunk_id": result.get("chunk_id"),
                    "document_id": result.get("document_id"),
                    "content": result.get("content"),
                    "score": result.get("similarity", 0.0),
                    "search_type": "dense",
                    "highlights": {},  # Vector search doesn't provide highlights
                    "metadata": {
                        "chunk_type": result.get("chunk_type"),
                        "page_number": result.get("page_number"),
                        "keywords": result.get("keywords", []),
                        "questions": result.get("questions", []),
                        "tags": result.get("tags", [])
                    }
                })
            
            max_score = max([r["score"] for r in normalized_results]) if normalized_results else 0.0
            
            return {
                "results": normalized_results,
                "total": len(normalized_results),
                "max_score": max_score
            }
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return {"results": [], "total": 0, "max_score": 0.0}
    
    async def _hybrid_search(self, 
                           query: str,
                           document_ids: Optional[List[str]],
                           top_k: int,
                           filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform hybrid search combining sparse and dense results."""
        try:
            # Perform both searches concurrently
            async with trio.open_nursery() as nursery:
                sparse_results = {}
                dense_results = {}
                
                async def run_sparse():
                    nonlocal sparse_results
                    sparse_results = await self._sparse_search(query, document_ids, top_k, filters)
                
                async def run_dense():
                    nonlocal dense_results
                    dense_results = await self._dense_search(query, document_ids, top_k, filters)
                
                nursery.start_soon(run_sparse)
                nursery.start_soon(run_dense)
            
            # Fuse results using weighted scoring
            fused_results = await self._fuse_results(
                sparse_results.get("results", []),
                dense_results.get("results", []),
                sparse_weight=settings.SPARSE_WEIGHT,
                dense_weight=settings.DENSE_WEIGHT
            )
            
            # Sort by fused score and limit results
            fused_results.sort(key=lambda x: x["score"], reverse=True)
            fused_results = fused_results[:top_k]
            
            max_score = max([r["score"] for r in fused_results]) if fused_results else 0.0
            
            return {
                "results": fused_results,
                "total": len(fused_results),
                "max_score": max_score,
                "sparse_total": sparse_results.get("total", 0),
                "dense_total": dense_results.get("total", 0)
            }
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {"results": [], "total": 0, "max_score": 0.0}
    
    async def _fuse_results(self, 
                          sparse_results: List[Dict[str, Any]],
                          dense_results: List[Dict[str, Any]],
                          sparse_weight: float = 0.3,
                          dense_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Fuse sparse and dense search results using weighted scoring.
        """
        # Create lookup for dense results
        dense_lookup = {result["chunk_id"]: result for result in dense_results}
        
        # Create lookup for sparse results
        sparse_lookup = {result["chunk_id"]: result for result in sparse_results}
        
        # Get all unique chunk IDs
        all_chunk_ids = set(dense_lookup.keys()) | set(sparse_lookup.keys())
        
        fused_results = []
        
        for chunk_id in all_chunk_ids:
            sparse_result = sparse_lookup.get(chunk_id)
            dense_result = dense_lookup.get(chunk_id)
            
            # Normalize scores (0-1 range)
            sparse_score = 0.0
            dense_score = 0.0
            
            if sparse_result:
                # Normalize Elasticsearch score (typically 0-10+ range)
                sparse_score = min(sparse_result["score"] / 10.0, 1.0)
            
            if dense_result:
                # Cosine similarity is already in 0-1 range
                dense_score = dense_result["score"]
            
            # Calculate fused score
            fused_score = (sparse_weight * sparse_score) + (dense_weight * dense_score)
            
            # Use the result with more complete information
            base_result = dense_result or sparse_result
            
            # Combine highlights from both sources
            highlights = {}
            if sparse_result:
                highlights.update(sparse_result.get("highlights", {}))
            
            # Combine metadata
            metadata = base_result.get("metadata", {})
            if sparse_result and dense_result:
                # Merge metadata from both sources
                sparse_metadata = sparse_result.get("metadata", {})
                metadata.update(sparse_metadata)
            
            fused_result = {
                "chunk_id": chunk_id,
                "document_id": base_result["document_id"],
                "content": base_result["content"],
                "score": fused_score,
                "search_type": "hybrid",
                "highlights": highlights,
                "metadata": metadata,
                "fusion_details": {
                    "sparse_score": sparse_score,
                    "dense_score": dense_score,
                    "sparse_weight": sparse_weight,
                    "dense_weight": dense_weight
                }
            }
            
            fused_results.append(fused_result)
        
        return fused_results
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific chunk by ID.
        
        Args:
            chunk_id: Chunk ID to retrieve
            
        Returns:
            Chunk data or None if not found
        """
        try:
            # Try vector store first (more complete data)
            results = await self.vector_store.search_by_embedding(
                embedding=[0.0] * 256,  # Dummy embedding for ID lookup
                top_k=1
            )
            
            # Filter by chunk_id (this is a workaround - ideally we'd have a direct get method)
            for result in results:
                if result["chunk_id"] == chunk_id:
                    return result
            
            return None
            
        except Exception as e:
            logger.error(f"Chunk retrieval failed for ID {chunk_id}: {e}")
            return None
    
    async def get_retriever_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        try:
            # Get vector store stats
            vector_stats = await self.vector_store.get_embedding_stats()
            
            # Get sparse retriever stats
            sparse_stats = await self.sparse_retriever.get_index_stats()
            
            return {
                "initialized": self._initialized,
                "vector_store": vector_stats,
                "sparse_search": sparse_stats,
                "hybrid_weights": {
                    "sparse_weight": settings.SPARSE_WEIGHT,
                    "dense_weight": settings.DENSE_WEIGHT
                }
            }
            
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {"error": str(e)}


# Global instance
basic_retriever = BasicRetriever()