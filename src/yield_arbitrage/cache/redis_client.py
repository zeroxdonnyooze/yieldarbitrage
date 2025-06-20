"""Redis connection and client management."""
import redis.asyncio as redis
from typing import Optional
from urllib.parse import urlparse

from ..config.settings import settings

# Global Redis client instance
redis_client: Optional[redis.Redis] = None


async def get_redis() -> redis.Redis:
    """Get Redis client instance."""
    global redis_client
    
    if redis_client is None:
        await init_redis()
    
    return redis_client


async def init_redis() -> None:
    """Initialize Redis connection pool."""
    global redis_client
    
    if redis_client is not None:
        return
    
    redis_url = settings.redis_url
    if not redis_url:
        raise ValueError("Redis URL not configured")
    
    # Parse Redis URL for connection parameters
    parsed_url = urlparse(redis_url)
    
    # Create connection pool with proper configuration
    redis_client = redis.Redis(
        host=parsed_url.hostname or 'localhost',
        port=parsed_url.port or 6379,
        db=int(parsed_url.path[1:]) if parsed_url.path and len(parsed_url.path) > 1 else 0,
        password=parsed_url.password,
        username=parsed_url.username,
        encoding='utf-8',
        decode_responses=True,
        health_check_interval=30,
        socket_keepalive=True,
        socket_keepalive_options={},
        retry_on_timeout=True,
        retry_on_error=[redis.BusyLoadingError, redis.ConnectionError, redis.TimeoutError],
        max_connections=10,
    )
    
    # Test the connection
    try:
        await redis_client.ping()
        print(f"✅ Redis connected to {parsed_url.hostname}:{parsed_url.port}")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        redis_client = None
        raise


async def close_redis() -> None:
    """Close Redis connection."""
    global redis_client
    
    if redis_client is not None:
        await redis_client.aclose()
        redis_client = None
        print("✅ Redis connection closed")


async def health_check() -> bool:
    """Check Redis health status."""
    try:
        client = await get_redis()
        await client.ping()
        return True
    except Exception:
        return False