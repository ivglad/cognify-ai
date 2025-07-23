"""
Redis caching infrastructure with trio support.
"""
import json
import logging
from typing import Optional, Any, Dict
import threading

import trio
import redis.asyncio as redis
import xxhash

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Redis cache manager with trio compatibility and xxhash optimization.
    """
    _instance: Optional['CacheManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.redis_client: Optional[redis.Redis] = None
            self.initialized = False
            self._connection_lock = trio.Lock()
    
    async def connect(self):
        """
        Connect to Redis with trio support.
        """
        async with self._connection_lock:
            if self.redis_client is None:
                try:
                    # Create Redis connection
                    self.redis_client = redis.Redis(
                        host=settings.REDIS_HOST,
                        port=settings.REDIS_PORT,
                        db=settings.REDIS_DB,
                        password=settings.REDIS_PASSWORD,
                        decode_responses=True,
                        socket_connect_timeout=5,
                        socket_timeout=5,
                        retry_on_timeout=True,
                        health_check_interval=30
                    )
                    
                    # Test connection
                    await self.redis_client.ping()
                    
                    logger.info(f"Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
                    self.initialized = True
                    
                except Exception as e:
                    logger.error(f"Failed to connect to Redis: {e}")
                    raise
    
    async def disconnect(self):
        """
        Disconnect from Redis.
        """
        async with self._connection_lock:
            if self.redis_client:
                try:
                    await self.redis_client.close()
                    self.redis_client = None
                    self.initialized = False
                    logger.info("Disconnected from Redis")
                except Exception as e:
                    logger.error(f"Error disconnecting from Redis: {e}")
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """
        Generate optimized cache key using xxhash.
        """
        hasher = xxhash.xxh64()
        
        # Hash all arguments
        for arg in args:
            if isinstance(arg, (dict, list)):
                hasher.update(json.dumps(arg, sort_keys=True).encode())
            else:
                hasher.update(str(arg).encode())
        
        # Hash keyword arguments
        for key, value in sorted(kwargs.items()):
            hasher.update(key.encode())
            if isinstance(value, (dict, list)):
                hasher.update(json.dumps(value, sort_keys=True).encode())
            else:
                hasher.update(str(value).encode())
        
        return hasher.hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        """
        if not self.initialized:
            await self.connect()
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with optional TTL.
        """
        if not self.initialized:
            await self.connect()
        
        try:
            serialized_value = json.dumps(value, default=str)
            if ttl:
                await self.redis_client.setex(key, ttl, serialized_value)
            else:
                await self.redis_client.set(key, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        """
        if not self.initialized:
            await self.connect()
        
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        """
        if not self.initialized:
            await self.connect()
        
        try:
            result = await self.redis_client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern.
        """
        if not self.initialized:
            await self.connect()
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0
    
    async def get_llm_cache(self, llm_name: str, text: str, history: list, gen_config: dict) -> Optional[str]:
        """
        Get cached LLM response using optimized key generation.
        """
        cache_key = f"llm:{self._generate_cache_key(llm_name, text, history, gen_config)}"
        return await self.get(cache_key)
    
    async def set_llm_cache(self, llm_name: str, text: str, response: str, history: list, gen_config: dict, ttl: int = None) -> bool:
        """
        Cache LLM response with optimized key generation.
        """
        cache_key = f"llm:{self._generate_cache_key(llm_name, text, history, gen_config)}"
        ttl = ttl or settings.LLM_CACHE_TTL
        return await self.set(cache_key, response, ttl)
    
    async def get_embed_cache(self, model_name: str, text: str) -> Optional[list]:
        """
        Get cached embeddings.
        """
        cache_key = f"embed:{self._generate_cache_key(model_name, text)}"
        return await self.get(cache_key)
    
    async def set_embed_cache(self, model_name: str, text: str, embeddings: list, ttl: int = None) -> bool:
        """
        Cache embeddings with compression.
        """
        cache_key = f"embed:{self._generate_cache_key(model_name, text)}"
        ttl = ttl or settings.EMBEDDING_CACHE_TTL
        return await self.set(cache_key, embeddings, ttl)
    
    async def get_search_cache(self, query: str, filters: dict) -> Optional[dict]:
        """
        Get cached search results.
        """
        cache_key = f"search:{self._generate_cache_key(query, filters)}"
        return await self.get(cache_key)
    
    async def set_search_cache(self, query: str, filters: dict, results: dict, ttl: int = None) -> bool:
        """
        Cache search results.
        """
        cache_key = f"search:{self._generate_cache_key(query, filters)}"
        ttl = ttl or settings.SEARCH_CACHE_TTL
        return await self.set(cache_key, results, ttl)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Redis health.
        """
        try:
            if not self.initialized:
                await self.connect()
            
            # Test ping
            pong = await self.redis_client.ping()
            
            # Get info
            info = await self.redis_client.info()
            
            return {
                "status": "healthy",
                "ping": pong,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "uptime": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global instance
cache_manager = CacheManager()