"""
Cache manager for performance optimization.
Handles Redis caching for chart results and expensive calculations.
"""

import os
import json
import pickle
import hashlib
from typing import Any, Optional, Callable
from functools import wraps
import logging

# Try to import Redis, fall back to in-memory cache if not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis not available, using in-memory cache")

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching for expensive calculations."""
    
    def __init__(self):
        self.cache_ttl = 3600  # 1 hour default TTL
        self._memory_cache = {}  # Fallback in-memory cache
        self._chart_cache = {}  # Dedicated chart cache
        self._analytics_cache = {}  # Dedicated analytics cache
        
        if REDIS_AVAILABLE:
            try:
                # Try to connect to Redis
                redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                self.redis_client.ping()  # Test connection
                self.redis_available = True
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
                self.redis_available = False
        else:
            self.redis_available = False
            logger.info("Using in-memory cache (Redis not available)")
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Create a hash of the arguments
        key_data = str(args) + str(sorted(kwargs.items()))
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data."""
        try:
            if self.redis_available:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    return pickle.loads(cached_data)
            else:
                # Use in-memory cache
                if key in self._memory_cache:
                    data, timestamp = self._memory_cache[key]
                    # Check if expired (simple TTL check)
                    if os.path.getmtime(__file__) - timestamp < self.cache_ttl:
                        return data
                    else:
                        del self._memory_cache[key]
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
        
        return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Set cached data."""
        try:
            if self.redis_available:
                ttl = ttl or self.cache_ttl
                self.redis_client.setex(key, ttl, pickle.dumps(data))
                return True
            else:
                # Use in-memory cache
                self._memory_cache[key] = (data, os.path.getmtime(__file__))
                return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cached data."""
        try:
            if self.redis_available:
                self.redis_client.delete(key)
            else:
                self._memory_cache.pop(key, None)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    def clear_portfolio_cache(self, portfolio_id: str):
        """Clear all cache entries for a specific portfolio."""
        try:
            if self.redis_available:
                # Get all keys matching the portfolio pattern
                pattern = f"portfolio:{portfolio_id}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                # Clear from all memory caches
                keys_to_delete = [k for k in self._memory_cache.keys() if k.startswith(f"portfolio:{portfolio_id}:")]
                for key in keys_to_delete:
                    del self._memory_cache[key]
                
                # Clear chart and analytics caches
                self._chart_cache.clear()
                self._analytics_cache.clear()
        except Exception as e:
            logger.warning(f"Cache clear error for portfolio {portfolio_id}: {e}")
    
    def get_chart_cache(self, key: str) -> Optional[Any]:
        """Get cached chart data."""
        return self._chart_cache.get(key)
    
    def set_chart_cache(self, key: str, data: Any):
        """Set cached chart data."""
        self._chart_cache[key] = data
    
    def get_analytics_cache(self, key: str) -> Optional[Any]:
        """Get cached analytics data."""
        return self._analytics_cache.get(key)
    
    def set_analytics_cache(self, key: str, data: Any):
        """Set cached analytics data."""
        self._analytics_cache[key] = data

# Global cache manager instance
cache_manager = CacheManager()

def cache_result(prefix: str, ttl: Optional[int] = None):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_manager._generate_cache_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}, executing...")
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

def cache_chart_result(ttl: Optional[int] = None):
    """Decorator specifically for chart results."""
    return cache_result("chart", ttl or 1800)  # 30 minutes for charts

def cache_analytics_result(ttl: Optional[int] = None):
    """Decorator specifically for analytics results."""
    return cache_result("analytics", ttl or 3600)  # 1 hour for analytics
