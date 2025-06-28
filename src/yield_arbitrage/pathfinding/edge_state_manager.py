"""Edge state management for pathfinding with advanced caching and retrieval."""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

from ..graph_engine.models import EdgeState, YieldGraphEdge
from ..data_collector.hybrid_collector import HybridDataCollector
from ..cache.redis_client import get_redis

logger = logging.getLogger(__name__)


class CacheLevel(str, Enum):
    """Cache level for edge state storage."""
    MEMORY = "memory"      # In-memory cache (fastest)
    REDIS = "redis"        # Redis cache (shared across instances)
    FRESH = "fresh"        # Fresh from data collector (slowest)


@dataclass
class CachedEdgeState:
    """Edge state with metadata for caching."""
    state: EdgeState
    cache_level: CacheLevel
    timestamp: float
    access_count: int = 0
    last_access: float = None
    
    def __post_init__(self):
        if self.last_access is None:
            self.last_access = self.timestamp


@dataclass
class StateRetrievalConfig:
    """Configuration for edge state retrieval."""
    memory_cache_ttl_seconds: float = 300.0      # 5 minutes
    redis_cache_ttl_seconds: float = 900.0       # 15 minutes
    max_memory_cache_size: int = 10000           # Maximum edges in memory
    batch_size: int = 50                         # Batch size for updates
    enable_predictive_caching: bool = True       # Pre-fetch likely edges
    cache_eviction_strategy: str = "lru"         # least recently used


class EdgeStateManager:
    """
    Manages edge state retrieval with multi-level caching and batch operations.
    
    This class provides efficient edge state retrieval with:
    - Multi-level caching (memory -> Redis -> fresh)
    - Batch state retrieval for efficiency
    - Predictive pre-fetching of likely edges
    - LRU cache eviction
    - Comprehensive metrics tracking
    """
    
    def __init__(
        self,
        data_collector: HybridDataCollector,
        redis_client = None,
        config: StateRetrievalConfig = None
    ):
        """
        Initialize edge state manager.
        
        Args:
            data_collector: Data collector for fresh state updates
            redis_client: Optional Redis client for distributed caching
            config: State retrieval configuration
        """
        self.data_collector = data_collector
        self.redis_client = redis_client
        self.config = config or StateRetrievalConfig()
        
        # Memory cache
        self._memory_cache: Dict[str, CachedEdgeState] = {}
        self._cache_access_order: List[str] = []  # For LRU tracking
        
        # Batch operations
        self._pending_updates: Set[str] = set()
        self._update_lock = asyncio.Lock()
        self._batch_semaphore = asyncio.Semaphore(5)  # Max concurrent batches
        
        # Metrics
        self._metrics = {
            "memory_hits": 0,
            "redis_hits": 0,
            "fresh_fetches": 0,
            "cache_evictions": 0,
            "batch_updates": 0,
            "failed_retrievals": 0
        }
        
        # Predictive caching
        self._edge_access_history: Dict[str, List[str]] = {}  # edge -> next edges accessed
        self._prefetch_candidates: Set[str] = set()
    
    async def get_edge_state(self, edge_id: str) -> Optional[EdgeState]:
        """
        Get edge state with multi-level caching.
        
        Args:
            edge_id: ID of the edge to get state for
            
        Returns:
            Current EdgeState or None if unavailable
        """
        # Try memory cache first
        cached_state = self._get_from_memory_cache(edge_id)
        if cached_state:
            return cached_state.state
        
        # Try Redis cache if available
        if self.redis_client:
            redis_state = await self._get_from_redis_cache(edge_id)
            if redis_state:
                # Store in memory cache
                await self._store_in_memory_cache(edge_id, redis_state)
                return redis_state.state
        
        # Get fresh state from data collector
        fresh_state = await self._get_fresh_state(edge_id)
        if fresh_state:
            # Store in both caches
            await self._store_in_caches(edge_id, fresh_state)
            
            # Track for predictive caching
            if self.config.enable_predictive_caching:
                await self._track_edge_access(edge_id)
        
        return fresh_state
    
    async def get_edge_states_batch(self, edge_ids: List[str]) -> Dict[str, Optional[EdgeState]]:
        """
        Get multiple edge states efficiently in batch.
        
        Args:
            edge_ids: List of edge IDs to retrieve
            
        Returns:
            Dictionary mapping edge IDs to their states
        """
        results = {}
        edges_to_fetch = []
        
        # Check memory cache first
        for edge_id in edge_ids:
            cached_state = self._get_from_memory_cache(edge_id)
            if cached_state:
                results[edge_id] = cached_state.state
            else:
                edges_to_fetch.append(edge_id)
        
        if not edges_to_fetch:
            return results
        
        # Check Redis cache for remaining edges
        if self.redis_client and edges_to_fetch:
            redis_results = await self._get_batch_from_redis(edges_to_fetch)
            for edge_id, state in redis_results.items():
                if state:
                    results[edge_id] = state.state
                    await self._store_in_memory_cache(edge_id, state)
                    edges_to_fetch.remove(edge_id)
        
        # Fetch remaining edges fresh
        if edges_to_fetch:
            fresh_results = await self._get_fresh_states_batch(edges_to_fetch)
            for edge_id, state in fresh_results.items():
                if state:
                    results[edge_id] = state
                    await self._store_in_caches(edge_id, state)
                else:
                    results[edge_id] = None
        
        return results
    
    async def prefetch_likely_edges(self, current_edge_id: str) -> None:
        """
        Pre-fetch edges likely to be accessed next based on history.
        
        Args:
            current_edge_id: Current edge being accessed
        """
        if not self.config.enable_predictive_caching:
            return
        
        # Get likely next edges from history
        likely_edges = self._edge_access_history.get(current_edge_id, [])[:10]
        
        if likely_edges:
            # Filter out already cached edges
            edges_to_prefetch = [
                edge_id for edge_id in likely_edges 
                if edge_id not in self._memory_cache
            ]
            
            if edges_to_prefetch:
                logger.debug(f"Pre-fetching {len(edges_to_prefetch)} likely edges")
                # Pre-fetch in background
                asyncio.create_task(self.get_edge_states_batch(edges_to_prefetch))
    
    def _get_from_memory_cache(self, edge_id: str) -> Optional[CachedEdgeState]:
        """Get edge state from memory cache if valid."""
        if edge_id not in self._memory_cache:
            return None
        
        cached = self._memory_cache[edge_id]
        current_time = time.time()
        
        # Check TTL
        if current_time - cached.timestamp > self.config.memory_cache_ttl_seconds:
            # Expired, remove from cache
            del self._memory_cache[edge_id]
            if edge_id in self._cache_access_order:
                self._cache_access_order.remove(edge_id)
            return None
        
        # Update access tracking
        cached.access_count += 1
        cached.last_access = current_time
        self._update_cache_access_order(edge_id)
        
        self._metrics["memory_hits"] += 1
        return cached
    
    async def _get_from_redis_cache(self, edge_id: str) -> Optional[CachedEdgeState]:
        """Get edge state from Redis cache if valid."""
        try:
            cache_key = f"edge_state_cache:{edge_id}"
            cached_json = await self.redis_client.get(cache_key)
            
            if not cached_json:
                return None
            
            # Parse cached data
            cached_data = eval(cached_json)  # In production, use json.loads
            state = EdgeState.model_validate(cached_data["state"])
            
            current_time = time.time()
            timestamp = cached_data.get("timestamp", 0)
            
            # Check TTL
            if current_time - timestamp > self.config.redis_cache_ttl_seconds:
                # Expired, remove from Redis
                await self.redis_client.delete(cache_key)
                return None
            
            self._metrics["redis_hits"] += 1
            
            return CachedEdgeState(
                state=state,
                cache_level=CacheLevel.REDIS,
                timestamp=timestamp,
                access_count=cached_data.get("access_count", 0)
            )
            
        except Exception as e:
            logger.warning(f"Failed to get edge {edge_id} from Redis cache: {e}")
            return None
    
    async def _get_batch_from_redis(self, edge_ids: List[str]) -> Dict[str, Optional[CachedEdgeState]]:
        """Get multiple edge states from Redis cache."""
        if not self.redis_client:
            return {}
        
        results = {}
        try:
            # Get all keys
            cache_keys = [f"edge_state_cache:{edge_id}" for edge_id in edge_ids]
            
            # Use Redis pipeline for efficiency
            pipe = self.redis_client.pipeline()
            for key in cache_keys:
                pipe.get(key)
            
            cached_values = await pipe.execute()
            
            current_time = time.time()
            
            for edge_id, cached_json in zip(edge_ids, cached_values):
                if cached_json:
                    try:
                        cached_data = eval(cached_json)  # In production, use json.loads
                        state = EdgeState.model_validate(cached_data["state"])
                        timestamp = cached_data.get("timestamp", 0)
                        
                        # Check TTL
                        if current_time - timestamp <= self.config.redis_cache_ttl_seconds:
                            results[edge_id] = CachedEdgeState(
                                state=state,
                                cache_level=CacheLevel.REDIS,
                                timestamp=timestamp,
                                access_count=cached_data.get("access_count", 0)
                            )
                            self._metrics["redis_hits"] += 1
                        else:
                            # Expired, schedule for deletion
                            asyncio.create_task(self.redis_client.delete(f"edge_state_cache:{edge_id}"))
                    except Exception as e:
                        logger.warning(f"Failed to parse cached state for {edge_id}: {e}")
                
            return results
            
        except Exception as e:
            logger.error(f"Failed to get batch from Redis: {e}")
            return {}
    
    async def _get_fresh_state(self, edge_id: str) -> Optional[EdgeState]:
        """Get fresh edge state from data collector."""
        try:
            # Get edge from graph
            edge = self.data_collector.graph.get_edge(edge_id)
            if not edge:
                logger.warning(f"Edge {edge_id} not found in graph")
                self._metrics["failed_retrievals"] += 1
                return None
            
            # Force update through data collector
            success = await self.data_collector.force_update_edge(edge_id)
            
            if success:
                # Get updated state from graph
                updated_edge = self.data_collector.graph.get_edge(edge_id)
                if updated_edge and updated_edge.state:
                    self._metrics["fresh_fetches"] += 1
                    return updated_edge.state
            
            self._metrics["failed_retrievals"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Failed to get fresh state for {edge_id}: {e}")
            self._metrics["failed_retrievals"] += 1
            return None
    
    async def _get_fresh_states_batch(self, edge_ids: List[str]) -> Dict[str, Optional[EdgeState]]:
        """Get fresh edge states in batch from data collector."""
        results = {}
        
        # Process in smaller batches to avoid overwhelming the system
        batch_size = self.config.batch_size
        
        for i in range(0, len(edge_ids), batch_size):
            batch = edge_ids[i:i + batch_size]
            
            async with self._batch_semaphore:
                batch_tasks = []
                
                for edge_id in batch:
                    task = asyncio.create_task(self._get_fresh_state(edge_id))
                    batch_tasks.append((edge_id, task))
                
                # Wait for batch to complete
                for edge_id, task in batch_tasks:
                    try:
                        state = await task
                        results[edge_id] = state
                    except Exception as e:
                        logger.warning(f"Failed to get fresh state for {edge_id}: {e}")
                        results[edge_id] = None
                
                self._metrics["batch_updates"] += 1
        
        return results
    
    async def _store_in_memory_cache(self, edge_id: str, state: EdgeState) -> None:
        """Store edge state in memory cache with LRU eviction."""
        # Check cache size and evict if necessary
        if len(self._memory_cache) >= self.config.max_memory_cache_size:
            await self._evict_from_memory_cache()
        
        # Create cached state
        cached = CachedEdgeState(
            state=state,
            cache_level=CacheLevel.MEMORY,
            timestamp=time.time()
        )
        
        self._memory_cache[edge_id] = cached
        self._update_cache_access_order(edge_id)
    
    async def _store_in_redis_cache(self, edge_id: str, state: EdgeState) -> None:
        """Store edge state in Redis cache."""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"edge_state_cache:{edge_id}"
            cache_data = {
                "state": state.model_dump(),
                "timestamp": time.time(),
                "access_count": 1
            }
            
            # Store with TTL
            await self.redis_client.set(
                cache_key, 
                str(cache_data),  # In production, use json.dumps
                ex=int(self.config.redis_cache_ttl_seconds)
            )
            
        except Exception as e:
            logger.warning(f"Failed to store edge {edge_id} in Redis cache: {e}")
    
    async def _store_in_caches(self, edge_id: str, state: EdgeState) -> None:
        """Store edge state in both memory and Redis caches."""
        await self._store_in_memory_cache(edge_id, state)
        await self._store_in_redis_cache(edge_id, state)
    
    def _update_cache_access_order(self, edge_id: str) -> None:
        """Update cache access order for LRU tracking."""
        if edge_id in self._cache_access_order:
            self._cache_access_order.remove(edge_id)
        self._cache_access_order.append(edge_id)
    
    async def _evict_from_memory_cache(self) -> None:
        """Evict least recently used items from memory cache."""
        if self.config.cache_eviction_strategy == "lru":
            # Evict least recently used
            if self._cache_access_order:
                edge_to_evict = self._cache_access_order.pop(0)
                if edge_to_evict in self._memory_cache:
                    del self._memory_cache[edge_to_evict]
                    self._metrics["cache_evictions"] += 1
        else:
            # Fallback: evict oldest
            if self._memory_cache:
                oldest_edge = min(
                    self._memory_cache.items(),
                    key=lambda x: x[1].timestamp
                )[0]
                del self._memory_cache[oldest_edge]
                if oldest_edge in self._cache_access_order:
                    self._cache_access_order.remove(oldest_edge)
                self._metrics["cache_evictions"] += 1
    
    async def _track_edge_access(self, edge_id: str) -> None:
        """Track edge access patterns for predictive caching."""
        # Store the last accessed edge ID for pattern tracking
        # This is simplified - in production, would use more sophisticated tracking
        pass
    
    def clear_cache(self) -> None:
        """Clear all memory caches."""
        self._memory_cache.clear()
        self._cache_access_order.clear()
        self._edge_access_history.clear()
        self._prefetch_candidates.clear()
        
        logger.info("Edge state memory cache cleared")
    
    async def clear_redis_cache(self) -> None:
        """Clear all Redis caches."""
        if not self.redis_client:
            return
        
        try:
            # Get all cache keys
            pattern = "edge_state_cache:*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                # Delete all cache keys
                await self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} edge states from Redis cache")
                
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        total_requests = (
            self._metrics["memory_hits"] + 
            self._metrics["redis_hits"] + 
            self._metrics["fresh_fetches"]
        )
        
        if total_requests > 0:
            memory_hit_rate = (self._metrics["memory_hits"] / total_requests) * 100
            redis_hit_rate = (self._metrics["redis_hits"] / total_requests) * 100
            fresh_fetch_rate = (self._metrics["fresh_fetches"] / total_requests) * 100
        else:
            memory_hit_rate = redis_hit_rate = fresh_fetch_rate = 0.0
        
        return {
            "cache_stats": {
                "memory_cache_size": len(self._memory_cache),
                "memory_hit_rate": f"{memory_hit_rate:.1f}%",
                "redis_hit_rate": f"{redis_hit_rate:.1f}%",
                "fresh_fetch_rate": f"{fresh_fetch_rate:.1f}%",
                "total_requests": total_requests,
                **self._metrics
            },
            "cache_config": {
                "memory_ttl_seconds": self.config.memory_cache_ttl_seconds,
                "redis_ttl_seconds": self.config.redis_cache_ttl_seconds,
                "max_memory_size": self.config.max_memory_cache_size,
                "eviction_strategy": self.config.cache_eviction_strategy
            }
        }