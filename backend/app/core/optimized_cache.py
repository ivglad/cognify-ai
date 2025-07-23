"""
Optimized caching system with xxhash for fast key generation.
"""
import json
import time
import zlib
import pickle
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta

import xxhash
import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class OptimizedCache:
    """High-performance caching with xxhash for key generation."""
    
    def __init__(self, redis_client: Optional[Redis] = None):
        self.redis = redis_client
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.local_cache_max_size = 1000
        self.local_cache_ttl = 300  # 5 minutes
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0,
            'local_hits': 0,
            'redis_hits': 0,
            'compression_saves': 0
        }
        
        # Performance tracking
        self.performance_metrics = {
            'avg_get_time': 0.0,
            'avg_set_time': 0.0,
            'total_operations': 0
        }
    
    def _generate_hash_key(self, *args, **kwargs) -> str:
        """Generate fast hash key using xxhash."""
        try:
            # Create a deterministic string from arguments
            key_data = {
                'args': args,
                'kwargs': sorted(kwargs.items()) if kwargs else {}
            }
            
            # Convert to JSON string for consistent hashing
            key_string = json.dumps(key_data, sort_keys=True, default=str)
            
            # Generate xxhash
            hasher = xxhash.xxh64()
            hasher.update(key_string.encode('utf-8'))
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Hash key generation failed: {e}")
            # Fallback to simple string concatenation
            return f"fallback_{hash(str(args) + str(kwargs))}"
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage efficiency."""
        try:
            # Serialize data
            serialized = pickle.dumps(data)
            
            # Compress if data is large enough
            if len(serialized) > 1024:  # Only compress if > 1KB
                compressed = zlib.compress(serialized, level=6)
                
                # Only use compression if it actually saves space
                if len(compressed) < len(serialized):
                    self.stats['compression_saves'] += 1
                    return b'compressed:' + compressed
            
            return b'raw:' + serialized
            
        except Exception as e:
            logger.error(f"Data compression failed: {e}")
            return pickle.dumps(data)
    
    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data from storage."""
        try:
            if data.startswith(b'compressed:'):
                # Decompress data
                compressed_data = data[11:]  # Remove 'compressed:' prefix
                decompressed = zlib.decompress(compressed_data)
                return pickle.loads(decompressed)
            elif data.startswith(b'raw:'):
                # Raw data
                raw_data = data[4:]  # Remove 'raw:' prefix
                return pickle.loads(raw_data)
            else:
                # Legacy format
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Data decompression failed: {e}")
            return None
    
    def _cleanup_local_cache(self):
        """Clean up expired entries from local cache."""
        try:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self.local_cache.items():
                if current_time > entry['expires_at']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.local_cache[key]
            
            # Also limit cache size
            if len(self.local_cache) > self.local_cache_max_size:
                # Remove oldest entries
                sorted_items = sorted(
                    self.local_cache.items(),
                    key=lambda x: x[1]['created_at']
                )
                
                items_to_remove = len(self.local_cache) - self.local_cache_max_size
                for i in range(items_to_remove):
                    key = sorted_items[i][0]
                    del self.local_cache[key]
                    
        except Exception as e:
            logger.error(f"Local cache cleanup failed: {e}")
    
    async def get_llm_cache(self, 
                           llm_name: str, 
                           text: str, 
                           history: List[Dict[str, Any]], 
                           gen_config: Dict[str, Any]) -> Optional[str]:
        """Get cached LLM response using optimized key generation."""
        start_time = time.time()
        
        try:
            # Generate optimized cache key
            cache_key = f"llm:{self._generate_hash_key(llm_name, text, history, gen_config)}"
            
            # Try local cache first
            if cache_key in self.local_cache:
                entry = self.local_cache[cache_key]
                if time.time() < entry['expires_at']:
                    self.stats['hits'] += 1
                    self.stats['local_hits'] += 1
                    return entry['data']
                else:
                    del self.local_cache[cache_key]
            
            # Try Redis cache
            if self.redis:
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    result = self._decompress_data(cached_data)
                    if result is not None:
                        # Store in local cache for faster access
                        self.local_cache[cache_key] = {
                            'data': result,
                            'created_at': time.time(),
                            'expires_at': time.time() + self.local_cache_ttl
                        }
                        
                        self.stats['hits'] += 1
                        self.stats['redis_hits'] += 1
                        return result
            
            self.stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"LLM cache get failed: {e}")
            self.stats['errors'] += 1
            return None
            
        finally:
            # Update performance metrics
            operation_time = time.time() - start_time
            self._update_performance_metrics('get', operation_time)
    
    async def set_llm_cache(self, 
                           llm_name: str, 
                           text: str, 
                           response: str, 
                           history: List[Dict[str, Any]], 
                           gen_config: Dict[str, Any], 
                           ttl: int = 3600):
        """Cache LLM response with optimized key generation."""
        start_time = time.time()
        
        try:
            # Generate optimized cache key
            cache_key = f"llm:{self._generate_hash_key(llm_name, text, history, gen_config)}"
            
            # Store in local cache
            self.local_cache[cache_key] = {
                'data': response,
                'created_at': time.time(),
                'expires_at': time.time() + min(ttl, self.local_cache_ttl)
            }
            
            # Store in Redis cache
            if self.redis:
                compressed_data = self._compress_data(response)
                await self.redis.setex(cache_key, ttl, compressed_data)
            
            self.stats['sets'] += 1
            
            # Cleanup local cache periodically
            if len(self.local_cache) % 100 == 0:
                self._cleanup_local_cache()
                
        except Exception as e:
            logger.error(f"LLM cache set failed: {e}")
            self.stats['errors'] += 1
            
        finally:
            # Update performance metrics
            operation_time = time.time() - start_time
            self._update_performance_metrics('set', operation_time)
    
    async def get_embed_cache(self, model_name: str, text: str) -> Optional[List[float]]:
        """Get cached embeddings with compression."""
        start_time = time.time()
        
        try:
            # Generate optimized cache key
            cache_key = f"embed:{self._generate_hash_key(model_name, text)}"
            
            # Try local cache first
            if cache_key in self.local_cache:
                entry = self.local_cache[cache_key]
                if time.time() < entry['expires_at']:
                    self.stats['hits'] += 1
                    self.stats['local_hits'] += 1
                    return entry['data']
                else:
                    del self.local_cache[cache_key]
            
            # Try Redis cache
            if self.redis:
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    result = self._decompress_data(cached_data)
                    if result is not None:
                        # Store in local cache
                        self.local_cache[cache_key] = {
                            'data': result,
                            'created_at': time.time(),
                            'expires_at': time.time() + self.local_cache_ttl
                        }
                        
                        self.stats['hits'] += 1
                        self.stats['redis_hits'] += 1
                        return result
            
            self.stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Embedding cache get failed: {e}")
            self.stats['errors'] += 1
            return None
            
        finally:
            operation_time = time.time() - start_time
            self._update_performance_metrics('get', operation_time)
    
    async def set_embed_cache(self, 
                             model_name: str, 
                             text: str, 
                             embeddings: List[float], 
                             ttl: int = 86400):
        """Cache embeddings with compression."""
        start_time = time.time()
        
        try:
            # Generate optimized cache key
            cache_key = f"embed:{self._generate_hash_key(model_name, text)}"
            
            # Store in local cache
            self.local_cache[cache_key] = {
                'data': embeddings,
                'created_at': time.time(),
                'expires_at': time.time() + min(ttl, self.local_cache_ttl)
            }
            
            # Store in Redis cache with compression
            if self.redis:
                compressed_data = self._compress_data(embeddings)
                await self.redis.setex(cache_key, ttl, compressed_data)
            
            self.stats['sets'] += 1
            
            # Cleanup local cache periodically
            if len(self.local_cache) % 100 == 0:
                self._cleanup_local_cache()
                
        except Exception as e:
            logger.error(f"Embedding cache set failed: {e}")
            self.stats['errors'] += 1
            
        finally:
            operation_time = time.time() - start_time
            self._update_performance_metrics('set', operation_time)
    
    async def get_search_cache(self, 
                              query: str, 
                              filters: Dict[str, Any], 
                              search_type: str = "hybrid") -> Optional[Dict[str, Any]]:
        """Get cached search results."""
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = f"search:{self._generate_hash_key(query, filters, search_type)}"
            
            # Try local cache first
            if cache_key in self.local_cache:
                entry = self.local_cache[cache_key]
                if time.time() < entry['expires_at']:
                    self.stats['hits'] += 1
                    self.stats['local_hits'] += 1
                    return entry['data']
                else:
                    del self.local_cache[cache_key]
            
            # Try Redis cache
            if self.redis:
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    result = self._decompress_data(cached_data)
                    if result is not None:
                        # Store in local cache
                        self.local_cache[cache_key] = {
                            'data': result,
                            'created_at': time.time(),
                            'expires_at': time.time() + self.local_cache_ttl
                        }
                        
                        self.stats['hits'] += 1
                        self.stats['redis_hits'] += 1
                        return result
            
            self.stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Search cache get failed: {e}")
            self.stats['errors'] += 1
            return None
            
        finally:
            operation_time = time.time() - start_time
            self._update_performance_metrics('get', operation_time)
    
    async def set_search_cache(self, 
                              query: str, 
                              filters: Dict[str, Any], 
                              results: Dict[str, Any], 
                              search_type: str = "hybrid", 
                              ttl: int = 1800):
        """Cache search results."""
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = f"search:{self._generate_hash_key(query, filters, search_type)}"
            
            # Store in local cache
            self.local_cache[cache_key] = {
                'data': results,
                'created_at': time.time(),
                'expires_at': time.time() + min(ttl, self.local_cache_ttl)
            }
            
            # Store in Redis cache
            if self.redis:
                compressed_data = self._compress_data(results)
                await self.redis.setex(cache_key, ttl, compressed_data)
            
            self.stats['sets'] += 1
            
        except Exception as e:
            logger.error(f"Search cache set failed: {e}")
            self.stats['errors'] += 1
            
        finally:
            operation_time = time.time() - start_time
            self._update_performance_metrics('set', operation_time)
    
    async def delete_pattern(self, pattern: str):
        """Delete cache entries matching pattern."""
        try:
            # Delete from local cache
            keys_to_delete = [key for key in self.local_cache.keys() if pattern in key]
            for key in keys_to_delete:
                del self.local_cache[key]
            
            # Delete from Redis cache
            if self.redis:
                keys = await self.redis.keys(f"*{pattern}*")
                if keys:
                    await self.redis.delete(*keys)
            
            self.stats['deletes'] += len(keys_to_delete)
            
        except Exception as e:
            logger.error(f"Cache pattern delete failed: {e}")
            self.stats['errors'] += 1
    
    async def clear_all(self):
        """Clear all cache entries."""
        try:
            # Clear local cache
            self.local_cache.clear()
            
            # Clear Redis cache
            if self.redis:
                await self.redis.flushdb()
            
            logger.info("All cache entries cleared")
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            self.stats['errors'] += 1
    
    def _update_performance_metrics(self, operation: str, duration: float):
        """Update performance metrics."""
        try:
            self.performance_metrics['total_operations'] += 1
            
            if operation == 'get':
                current_avg = self.performance_metrics['avg_get_time']
                total_ops = self.performance_metrics['total_operations']
                self.performance_metrics['avg_get_time'] = (
                    (current_avg * (total_ops - 1) + duration) / total_ops
                )
            elif operation == 'set':
                current_avg = self.performance_metrics['avg_set_time']
                total_ops = self.performance_metrics['total_operations']
                self.performance_metrics['avg_set_time'] = (
                    (current_avg * (total_ops - 1) + duration) / total_ops
                )
                
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        try:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'statistics': {
                    'hits': self.stats['hits'],
                    'misses': self.stats['misses'],
                    'sets': self.stats['sets'],
                    'deletes': self.stats['deletes'],
                    'errors': self.stats['errors'],
                    'local_hits': self.stats['local_hits'],
                    'redis_hits': self.stats['redis_hits'],
                    'compression_saves': self.stats['compression_saves'],
                    'hit_rate_percent': round(hit_rate, 2),
                    'total_requests': total_requests
                },
                'performance': {
                    'avg_get_time_ms': round(self.performance_metrics['avg_get_time'] * 1000, 2),
                    'avg_set_time_ms': round(self.performance_metrics['avg_set_time'] * 1000, 2),
                    'total_operations': self.performance_metrics['total_operations']
                },
                'local_cache': {
                    'size': len(self.local_cache),
                    'max_size': self.local_cache_max_size,
                    'ttl_seconds': self.local_cache_ttl
                },
                'redis_connected': self.redis is not None
            }
            
        except Exception as e:
            logger.error(f"Cache stats retrieval failed: {e}")
            return {'error': str(e)}
    
    def reset_stats(self):
        """Reset cache statistics."""
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0,
            'local_hits': 0,
            'redis_hits': 0,
            'compression_saves': 0
        }
        
        self.performance_metrics = {
            'avg_get_time': 0.0,
            'avg_set_time': 0.0,
            'total_operations': 0
        }
        
        logger.info("Cache statistics reset")


class CacheManager:
    """Centralized cache management."""
    
    def __init__(self):
        self.redis_client: Optional[Redis] = None
        self.optimized_cache: Optional[OptimizedCache] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize cache manager."""
        try:
            # Initialize Redis connection
            if settings.REDIS_URL:
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=False,  # We handle binary data
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test connection
                await self.redis_client.ping()
                logger.info("Redis connection established")
            else:
                logger.warning("Redis URL not configured, using local cache only")
            
            # Initialize optimized cache
            self.optimized_cache = OptimizedCache(self.redis_client)
            
            self._initialized = True
            logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Cache manager initialization failed: {e}")
            # Initialize with local cache only
            self.optimized_cache = OptimizedCache(None)
            self._initialized = True
    
    async def close(self):
        """Close cache connections."""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis connection closed")
                
        except Exception as e:
            logger.error(f"Cache manager close failed: {e}")
    
    def get_cache(self) -> OptimizedCache:
        """Get optimized cache instance."""
        if not self._initialized:
            raise RuntimeError("Cache manager not initialized")
        
        return self.optimized_cache
    
    async def health_check(self) -> Dict[str, Any]:
        """Check cache system health."""
        try:
            health = {
                'initialized': self._initialized,
                'redis_connected': False,
                'local_cache_active': self.optimized_cache is not None
            }
            
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    health['redis_connected'] = True
                except:
                    health['redis_connected'] = False
            
            if self.optimized_cache:
                health['cache_stats'] = self.optimized_cache.get_cache_stats()
            
            return health
            
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {'error': str(e)}


# Global cache manager instance
cache_manager = CacheManager()