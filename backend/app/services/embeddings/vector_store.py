"""
Vector store service for Infinity DB with embedding operations.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import uuid
from datetime import datetime

import trio
import numpy as np

from app.db.infinity_client import infinity_client
from app.db.infinity_schemas import COLLECTION_NAMES, YANDEX_EMBEDDING_DIMENSION
from app.services.embeddings.embedding_service import embedding_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store service for managing embeddings in Infinity DB.
    """
    
    def __init__(self):
        self.infinity_client = infinity_client
        self.embedding_service = embedding_service
        self._initialized = False
        
    async def initialize(self):
        """Initialize the vector store."""
        if self._initialized:
            return
            
        try:
            # Ensure Infinity client is connected
            if not self.infinity_client.initialized:
                await self.infinity_client.connect()
            
            # Ensure embedding service is initialized
            if not self.embedding_service._initialized:
                await self.embedding_service.initialize()
            
            self._initialized = True
            logger.info("VectorStore initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {e}")
            raise
    
    async def store_chunk_embedding(self, 
                                  chunk_data: Dict[str, Any],
                                  embedding: Optional[List[float]] = None) -> bool:
        """
        Store chunk with its embedding in vector database.
        
        Args:
            chunk_data: Chunk data including content and metadata
            embedding: Pre-computed embedding (optional)
            
        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Generate embedding if not provided
            if embedding is None:
                content = chunk_data.get('content', '')
                embedding = await self.embedding_service.generate_embedding(content)
            
            # Validate embedding dimension
            if len(embedding) != YANDEX_EMBEDDING_DIMENSION:
                logger.warning(f"Embedding dimension mismatch: expected {YANDEX_EMBEDDING_DIMENSION}, got {len(embedding)}")
                # Pad or truncate as needed
                if len(embedding) < YANDEX_EMBEDDING_DIMENSION:
                    embedding.extend([0.0] * (YANDEX_EMBEDDING_DIMENSION - len(embedding)))
                else:
                    embedding = embedding[:YANDEX_EMBEDDING_DIMENSION]
            
            # Get chunks table
            table = await self.infinity_client.get_table(
                settings.INFINITY_DATABASE,
                COLLECTION_NAMES["chunks"]
            )
            
            # Prepare data for insertion
            insert_data = {
                "chunk_id": chunk_data.get('chunk_id', str(uuid.uuid4())),
                "document_id": chunk_data.get('document_id'),
                "content": chunk_data.get('content', ''),
                "embedding": embedding,
                "chunk_type": chunk_data.get('chunk_type', 'text'),
                "chunk_strategy": chunk_data.get('chunk_strategy', 'naive'),
                "page_number": chunk_data.get('page_number'),
                "token_count": chunk_data.get('token_count', 0),
                "keywords": json.dumps(chunk_data.get('keywords', [])),
                "questions": json.dumps(chunk_data.get('questions', [])),
                "tags": json.dumps(chunk_data.get('tags', [])),
                "created_at": datetime.utcnow()
            }
            
            # Insert data
            table.insert([insert_data])
            
            logger.debug(f"Stored chunk embedding: {chunk_data.get('chunk_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chunk embedding: {e}")
            return False
    
    async def store_batch_chunk_embeddings(self, 
                                         chunks_data: List[Dict[str, Any]],
                                         batch_size: int = 100) -> Dict[str, Any]:
        """
        Store multiple chunks with embeddings in batches.
        
        Args:
            chunks_data: List of chunk data
            batch_size: Batch size for processing
            
        Returns:
            Processing results
        """
        if not self._initialized:
            await self.initialize()
        
        if not chunks_data:
            return {"stored": 0, "errors": []}
        
        try:
            # Generate embeddings for all chunks
            contents = [chunk.get('content', '') for chunk in chunks_data]
            embeddings = await self.embedding_service.generate_batch_embeddings(contents)
            
            # Process in batches
            stored_count = 0
            errors = []
            
            for i in range(0, len(chunks_data), batch_size):
                batch_chunks = chunks_data[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                batch_results = await self._store_chunk_batch(batch_chunks, batch_embeddings)
                stored_count += batch_results["stored"]
                errors.extend(batch_results["errors"])
            
            logger.info(f"Stored {stored_count} chunk embeddings, {len(errors)} errors")
            
            return {
                "stored": stored_count,
                "errors": errors,
                "total": len(chunks_data)
            }
            
        except Exception as e:
            logger.error(f"Batch chunk embedding storage failed: {e}")
            return {"stored": 0, "errors": [str(e)], "total": len(chunks_data)}
    
    async def _store_chunk_batch(self, 
                               chunks_data: List[Dict[str, Any]], 
                               embeddings: List[List[float]]) -> Dict[str, Any]:
        """Store a batch of chunks with embeddings."""
        try:
            # Get chunks table
            table = await self.infinity_client.get_table(
                settings.INFINITY_DATABASE,
                COLLECTION_NAMES["chunks"]
            )
            
            # Prepare batch data
            batch_data = []
            errors = []
            
            for chunk_data, embedding in zip(chunks_data, embeddings):
                try:
                    # Validate embedding dimension
                    if len(embedding) != YANDEX_EMBEDDING_DIMENSION:
                        if len(embedding) < YANDEX_EMBEDDING_DIMENSION:
                            embedding.extend([0.0] * (YANDEX_EMBEDDING_DIMENSION - len(embedding)))
                        else:
                            embedding = embedding[:YANDEX_EMBEDDING_DIMENSION]
                    
                    insert_data = {
                        "chunk_id": chunk_data.get('chunk_id', str(uuid.uuid4())),
                        "document_id": chunk_data.get('document_id'),
                        "content": chunk_data.get('content', ''),
                        "embedding": embedding,
                        "chunk_type": chunk_data.get('chunk_type', 'text'),
                        "chunk_strategy": chunk_data.get('chunk_strategy', 'naive'),
                        "page_number": chunk_data.get('page_number'),
                        "token_count": chunk_data.get('token_count', 0),
                        "keywords": json.dumps(chunk_data.get('keywords', [])),
                        "questions": json.dumps(chunk_data.get('questions', [])),
                        "tags": json.dumps(chunk_data.get('tags', [])),
                        "created_at": datetime.utcnow()
                    }
                    
                    batch_data.append(insert_data)
                    
                except Exception as e:
                    errors.append(f"Chunk {chunk_data.get('chunk_id', 'unknown')}: {str(e)}")
            
            # Insert batch
            if batch_data:
                table.insert(batch_data)
            
            return {
                "stored": len(batch_data),
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Batch insertion failed: {e}")
            return {
                "stored": 0,
                "errors": [str(e)]
            }
    
    async def search_similar_chunks(self, 
                                  query: str,
                                  document_ids: Optional[List[str]] = None,
                                  top_k: int = 20,
                                  similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query: Search query
            document_ids: Filter by document IDs
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar chunks with scores
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            # Get chunks table
            table = await self.infinity_client.get_table(
                settings.INFINITY_DATABASE,
                COLLECTION_NAMES["chunks"]
            )
            
            # Build search query
            search_query = table.output([
                "chunk_id", "document_id", "content", "chunk_type", 
                "page_number", "keywords", "questions", "tags"
            ])
            
            # Add document filter if specified
            if document_ids:
                search_query = search_query.filter(
                    table.document_id.is_in(document_ids)
                )
            
            # Add vector similarity search
            search_query = search_query.match_dense(
                "embedding",
                query_embedding,
                "cosine",
                top_k
            )
            
            # Execute search
            results = search_query.to_result()
            
            # Process results
            processed_results = []
            for result in results:
                # Calculate similarity score (assuming it's provided by Infinity)
                similarity = getattr(result, '_score', 1.0)
                
                if similarity >= similarity_threshold:
                    processed_results.append({
                        "chunk_id": result.chunk_id,
                        "document_id": result.document_id,
                        "content": result.content,
                        "chunk_type": result.chunk_type,
                        "page_number": result.page_number,
                        "keywords": json.loads(result.keywords or "[]"),
                        "questions": json.loads(result.questions or "[]"),
                        "tags": json.loads(result.tags or "[]"),
                        "similarity": similarity
                    })
            
            logger.debug(f"Vector search returned {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def search_by_embedding(self, 
                                embedding: List[float],
                                document_ids: Optional[List[str]] = None,
                                top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search using pre-computed embedding.
        
        Args:
            embedding: Query embedding vector
            document_ids: Filter by document IDs
            top_k: Number of results to return
            
        Returns:
            List of similar chunks with scores
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate embedding dimension
            if len(embedding) != YANDEX_EMBEDDING_DIMENSION:
                logger.warning(f"Embedding dimension mismatch: expected {YANDEX_EMBEDDING_DIMENSION}, got {len(embedding)}")
                return []
            
            # Get chunks table
            table = await self.infinity_client.get_table(
                settings.INFINITY_DATABASE,
                COLLECTION_NAMES["chunks"]
            )
            
            # Build search query
            search_query = table.output([
                "chunk_id", "document_id", "content", "chunk_type", 
                "page_number", "keywords", "questions", "tags"
            ])
            
            # Add document filter if specified
            if document_ids:
                search_query = search_query.filter(
                    table.document_id.is_in(document_ids)
                )
            
            # Add vector similarity search
            search_query = search_query.match_dense(
                "embedding",
                embedding,
                "cosine",
                top_k
            )
            
            # Execute search
            results = search_query.to_result()
            
            # Process results
            processed_results = []
            for result in results:
                similarity = getattr(result, '_score', 1.0)
                
                processed_results.append({
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "content": result.content,
                    "chunk_type": result.chunk_type,
                    "page_number": result.page_number,
                    "keywords": json.loads(result.keywords or "[]"),
                    "questions": json.loads(result.questions or "[]"),
                    "tags": json.loads(result.tags or "[]"),
                    "similarity": similarity
                })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            return []
    
    async def delete_document_embeddings(self, document_id: str) -> bool:
        """
        Delete all embeddings for a document.
        
        Args:
            document_id: Document ID to delete embeddings for
            
        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get chunks table
            table = await self.infinity_client.get_table(
                settings.INFINITY_DATABASE,
                COLLECTION_NAMES["chunks"]
            )
            
            # Delete chunks for document
            table.delete().filter(table.document_id == document_id)
            
            logger.info(f"Deleted embeddings for document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings for document {document_id}: {e}")
            return False
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Get chunks table
            table = await self.infinity_client.get_table(
                settings.INFINITY_DATABASE,
                COLLECTION_NAMES["chunks"]
            )
            
            # Count total chunks
            count_result = table.output(["count(*)"]).to_result()
            total_chunks = count_result[0][0] if count_result else 0
            
            # Get unique documents count
            doc_result = table.output(["count(distinct document_id)"]).to_result()
            unique_documents = doc_result[0][0] if doc_result else 0
            
            return {
                "total_chunks": total_chunks,
                "unique_documents": unique_documents,
                "embedding_dimension": YANDEX_EMBEDDING_DIMENSION,
                "collection_name": COLLECTION_NAMES["chunks"],
                "initialized": self._initialized
            }
            
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {"error": str(e)}


# Global instance
vector_store = VectorStore()