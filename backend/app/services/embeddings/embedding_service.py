"""
Embedding generation service with Yandex Cloud ML integration and caching.
"""
import logging
from typing import List, Dict, Any, Optional, Union
import time

import trio
import numpy as np

from app.core.config import settings
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding generation service with caching and rate limiting.
    """
    
    def __init__(self):
        self.cache_manager = cache_manager
        self.yandex_client = None
        self._rate_limiter = trio.CapacityLimiter(settings.MAX_CONCURRENT_CHATS)
        self._initialized = False
        
    async def initialize(self):
        """Initialize the embedding service."""
        if self._initialized:
            return
            
        try:
            # Initialize Yandex Cloud ML client
            await self._initialize_yandex_client()
            
            self._initialized = True
            logger.info("EmbeddingService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingService: {e}")
            raise
    
    async def _initialize_yandex_client(self):
        """Initialize Yandex Cloud ML client."""
        try:
            # Import Yandex SDK
            from yandex_cloud_ml_sdk import YCloudML
            
            if not settings.YANDEX_API_KEY or not settings.YANDEX_FOLDER_ID:
                logger.warning("Yandex API credentials not configured, using mock embeddings")
                self.yandex_client = None
                return
            
            # Initialize client
            self.yandex_client = YCloudML(
                folder_id=settings.YANDEX_FOLDER_ID,
                auth=settings.YANDEX_API_KEY
            )
            
            logger.info("Yandex Cloud ML client initialized")
            
        except ImportError:
            logger.warning("Yandex Cloud ML SDK not available, using mock embeddings")
            self.yandex_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Yandex client: {e}")
            self.yandex_client = None
    
    async def generate_embedding(self, 
                               text: str, 
                               model: str = None,
                               use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model: Model name (optional, uses default)
            use_cache: Whether to use caching
            
        Returns:
            Embedding vector
        """
        if not self._initialized:
            await self.initialize()
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return self._get_zero_embedding()
        
        model = model or settings.YANDEX_EMBEDDING_MODEL
        
        # Check cache first
        if use_cache:
            cached_embedding = await self.cache_manager.get_embed_cache(model, text)
            if cached_embedding:
                logger.debug(f"Cache hit for embedding: {text[:50]}...")
                return cached_embedding
        
        try:
            # Generate embedding with rate limiting
            async with self._rate_limiter:
                embedding = await self._generate_single_embedding(text, model)
            
            # Cache result
            if use_cache and embedding:
                await self.cache_manager.set_embed_cache(model, text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed for text '{text[:50]}...': {e}")
            return self._get_zero_embedding()
    
    async def generate_batch_embeddings(self, 
                                      texts: List[str], 
                                      model: str = None,
                                      use_cache: bool = True,
                                      batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            model: Model name (optional, uses default)
            use_cache: Whether to use caching
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()
        
        if not texts:
            return []
        
        model = model or settings.YANDEX_EMBEDDING_MODEL
        
        # Process in batches to avoid overwhelming the API
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await self._process_embedding_batch(batch, model, use_cache)
            results.extend(batch_results)
            
            # Small delay between batches to respect rate limits
            if i + batch_size < len(texts):
                await trio.sleep(0.1)
        
        return results
    
    async def _process_embedding_batch(self, 
                                     texts: List[str], 
                                     model: str,
                                     use_cache: bool) -> List[List[float]]:
        """Process a batch of texts for embedding generation."""
        # Check cache for all texts first
        cached_results = {}
        uncached_texts = []
        
        if use_cache:
            for text in texts:
                cached_embedding = await self.cache_manager.get_embed_cache(model, text)
                if cached_embedding:
                    cached_results[text] = cached_embedding
                else:
                    uncached_texts.append(text)
        else:
            uncached_texts = texts
        
        # Generate embeddings for uncached texts
        new_embeddings = {}
        if uncached_texts:
            # Use trio nursery for concurrent processing
            async with trio.open_nursery() as nursery:
                embedding_results = {}
                
                async def generate_and_store(text: str):
                    embedding = await self.generate_embedding(text, model, use_cache=False)
                    embedding_results[text] = embedding
                
                # Start tasks for all uncached texts
                for text in uncached_texts:
                    nursery.start_soon(generate_and_store, text)
            
            new_embeddings = embedding_results
            
            # Cache new embeddings
            if use_cache:
                for text, embedding in new_embeddings.items():
                    await self.cache_manager.set_embed_cache(model, text, embedding)
        
        # Combine cached and new results in original order
        results = []
        for text in texts:
            if text in cached_results:
                results.append(cached_results[text])
            elif text in new_embeddings:
                results.append(new_embeddings[text])
            else:
                results.append(self._get_zero_embedding())
        
        return results
    
    async def _generate_single_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding for a single text using Yandex API."""
        if not self.yandex_client:
            # Return mock embedding if client not available
            return self._generate_mock_embedding(text)
        
        try:
            # Use Yandex Cloud ML SDK
            response = await trio.to_thread.run_sync(
                self._call_yandex_embedding_api,
                text,
                model
            )
            
            if response and 'embedding' in response:
                return response['embedding']
            else:
                logger.warning(f"Invalid response from Yandex API: {response}")
                return self._generate_mock_embedding(text)
                
        except Exception as e:
            logger.error(f"Yandex embedding API call failed: {e}")
            return self._generate_mock_embedding(text)
    
    def _call_yandex_embedding_api(self, text: str, model: str) -> Dict[str, Any]:
        """Call Yandex embedding API (synchronous)."""
        try:
            # This is a placeholder for the actual Yandex API call
            # The exact implementation depends on the Yandex SDK
            
            # Example structure (adjust based on actual SDK):
            # response = self.yandex_client.embeddings.create(
            #     model=model,
            #     text=text
            # )
            # return response.to_dict()
            
            # For now, return mock response
            logger.warning("Using mock Yandex API response")
            return {
                'embedding': self._generate_mock_embedding(text)
            }
            
        except Exception as e:
            logger.error(f"Yandex API call error: {e}")
            raise
    
    def _generate_mock_embedding(self, text: str, dimension: int = 1536) -> List[float]:
        """Generate mock embedding for testing purposes."""
        # Create deterministic embedding based on text hash
        import hashlib
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        np.random.seed(int(text_hash[:8], 16))
        
        # Generate normalized random vector
        embedding = np.random.normal(0, 1, dimension)
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
    def _get_zero_embedding(self, dimension: int = 1536) -> List[float]:
        """Get zero embedding as fallback."""
        return [0.0] * dimension
    
    async def calculate_similarity(self, 
                                 embedding1: List[float], 
                                 embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    async def find_similar_embeddings(self, 
                                    query_embedding: List[float], 
                                    candidate_embeddings: List[List[float]],
                                    top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of similarity results with scores and indices
        """
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = await self.calculate_similarity(query_embedding, candidate)
                similarities.append({
                    'index': i,
                    'similarity': similarity
                })
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Similar embeddings search failed: {e}")
            return []
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        try:
            cache_stats = await self.cache_manager.health_check()
            
            return {
                "initialized": self._initialized,
                "yandex_client_available": self.yandex_client is not None,
                "default_model": settings.YANDEX_EMBEDDING_MODEL,
                "cache_status": cache_stats.get("status", "unknown"),
                "rate_limiter_capacity": self._rate_limiter.total_tokens,
                "rate_limiter_available": self._rate_limiter.available_tokens
            }
            
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {"error": str(e)}
    
    async def clear_embedding_cache(self, pattern: str = "embed:*") -> int:
        """Clear embedding cache."""
        try:
            cleared = await self.cache_manager.clear_pattern(pattern)
            logger.info(f"Cleared {cleared} embedding cache entries")
            return cleared
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
            return 0


# Global instance
embedding_service = EmbeddingService()