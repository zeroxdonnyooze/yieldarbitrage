"""Enhanced hybrid data collector with real-time WebSocket event processing."""
import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass

from yield_arbitrage.graph_engine.models import UniversalYieldGraph, EdgeState, YieldGraphEdge
from yield_arbitrage.protocols.base_adapter import ProtocolAdapter

logger = logging.getLogger(__name__)


class EdgePriority(str, Enum):
    """Priority levels for edge updates."""
    CRITICAL = "critical"      # Sub-second updates via WebSocket
    HIGH = "high"             # 5-30 second updates  
    NORMAL = "normal"         # 5-15 minute updates
    LOW = "low"              # Hourly updates


@dataclass
class EdgeUpdateConfig:
    """Configuration for edge update behavior."""
    priority: EdgePriority
    update_interval_seconds: int
    use_websocket: bool = False
    confidence_threshold: float = 0.5
    max_retries: int = 3
    timeout_seconds: int = 30


@dataclass
class CollectorStats:
    """Statistics for data collection performance."""
    total_edges: int = 0
    edges_by_priority: Dict[EdgePriority, int] = None
    updates_performed: int = 0
    updates_failed: int = 0
    websocket_connections: int = 0
    cache_hits: int = 0
    avg_update_latency_ms: float = 0.0
    last_update_cycle: Optional[datetime] = None
    
    def __post_init__(self):
        if self.edges_by_priority is None:
            self.edges_by_priority = {priority: 0 for priority in EdgePriority}


class EdgePriorityClassifier:
    """Classifies edges based on importance for update frequency."""
    
    def __init__(self):
        self.priority_thresholds = {
            EdgePriority.CRITICAL: {"liquidity_usd": 10_000_000, "volume_24h": 5_000_000},
            EdgePriority.HIGH: {"liquidity_usd": 1_000_000, "volume_24h": 500_000},
            EdgePriority.NORMAL: {"liquidity_usd": 100_000, "volume_24h": 50_000},
            EdgePriority.LOW: {"liquidity_usd": 0, "volume_24h": 0}
        }
    
    def classify_edge(self, edge: YieldGraphEdge) -> EdgePriority:
        """Classify edge priority based on liquidity and volume."""
        try:
            liquidity = edge.state.liquidity_usd or 0
            # Volume would come from edge metadata or external data
            volume = 0  # Placeholder - would be fetched from external sources
            
            # For now, just use liquidity for classification
            if liquidity >= self.priority_thresholds[EdgePriority.CRITICAL]["liquidity_usd"]:
                return EdgePriority.CRITICAL
            elif liquidity >= self.priority_thresholds[EdgePriority.HIGH]["liquidity_usd"]:
                return EdgePriority.HIGH
            elif liquidity >= self.priority_thresholds[EdgePriority.NORMAL]["liquidity_usd"]:
                return EdgePriority.NORMAL
            else:
                return EdgePriority.LOW
            
        except Exception as e:
            logger.warning(f"Error classifying edge {edge.edge_id}: {e}")
            return EdgePriority.NORMAL


class HybridDataCollector:
    """Enhanced data collector with real-time WebSocket event processing."""
    
    def __init__(
        self,
        graph: UniversalYieldGraph,
        redis_client,
        adapters: List[ProtocolAdapter],
        enable_websockets: bool = True
    ):
        self.graph = graph
        self.redis_client = redis_client
        self.adapters = adapters
        self.enable_websockets = enable_websockets
        
        # Edge management
        self.edge_configs: Dict[str, EdgeUpdateConfig] = {}
        self.last_update_times: Dict[str, float] = {}
        self.update_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Priority classification
        self.priority_classifier = EdgePriorityClassifier()
        
        # WebSocket connections (placeholder for future implementation)
        self.websocket_connections: Dict[str, Any] = {}
        
        # Performance tracking
        self.stats = CollectorStats()
        self.update_semaphore = asyncio.Semaphore(50)  # Max concurrent updates
        
        # Default update intervals by priority
        self.default_intervals = {
            EdgePriority.CRITICAL: 1,      # 1 second
            EdgePriority.HIGH: 15,         # 15 seconds
            EdgePriority.NORMAL: 300,      # 5 minutes
            EdgePriority.LOW: 3600         # 1 hour
        }
        
        # Background task management
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = False
    
    async def initialize(self) -> bool:
        """Initialize the data collector and discover edges."""
        try:
            logger.info("Initializing HybridDataCollector...")
            
            # Discover edges from all adapters
            await self._initialize_graph()
            
            # Classify edges and create update configs
            await self._classify_and_configure_edges()
            
            # Initialize WebSocket connections for critical edges
            if self.enable_websockets:
                await self._initialize_websocket_connections()
            
            logger.info(f"HybridDataCollector initialized with {len(self.edge_configs)} edges")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize HybridDataCollector: {e}")
            return False
    
    async def _initialize_graph(self) -> None:
        """Discover edges from all adapters and add to graph."""
        logger.info("Discovering edges from adapters...")
        
        for adapter in self.adapters:
            try:
                logger.info(f"Discovering edges from {adapter.__class__.__name__}")
                discovered_edges = await adapter.discover_edges()
                
                for edge in discovered_edges:
                    if self.graph.add_edge(edge):
                        logger.debug(f"Added edge: {edge.edge_id}")
                        
                        # Get initial state
                        try:
                            initial_state = await adapter.update_edge_state(edge)
                            edge.state = initial_state
                            await self._store_edge_state(edge.edge_id, initial_state)
                            
                        except Exception as e:
                            logger.warning(f"Failed to get initial state for {edge.edge_id}: {e}")
                            await self._store_edge_state(edge.edge_id, edge.state)
                
                logger.info(f"Discovered {len(discovered_edges)} edges from {adapter.__class__.__name__}")
                
            except Exception as e:
                logger.error(f"Error discovering edges from {adapter.__class__.__name__}: {e}")
        
        self.stats.total_edges = len(self.graph.edges)
        logger.info(f"Total edges in graph: {self.stats.total_edges}")
    
    async def _classify_and_configure_edges(self) -> None:
        """Classify all edges and create update configurations."""
        logger.info("Classifying edges and creating update configurations...")
        
        for edge_id, edge in self.graph.edges.items():
            priority = self.priority_classifier.classify_edge(edge)
            
            config = EdgeUpdateConfig(
                priority=priority,
                update_interval_seconds=self.default_intervals[priority],
                use_websocket=(priority == EdgePriority.CRITICAL and self.enable_websockets),
                confidence_threshold=0.5,
                max_retries=3,
                timeout_seconds=30
            )
            
            self.edge_configs[edge_id] = config
            self.stats.edges_by_priority[priority] += 1
            
            logger.debug(f"Edge {edge_id} classified as {priority.value}")
    
    async def _initialize_websocket_connections(self) -> None:
        """Initialize WebSocket connections for critical edges."""
        logger.info("Initializing WebSocket connections for critical edges...")
        
        # This is a placeholder for future WebSocket implementation
        # Would connect to protocol-specific WebSocket endpoints for real-time data
        critical_edges = [
            edge_id for edge_id, config in self.edge_configs.items()
            if config.use_websocket
        ]
        
        logger.info(f"Would initialize WebSocket connections for {len(critical_edges)} critical edges")
        self.stats.websocket_connections = len(critical_edges)
    
    async def start(self) -> None:
        """Start the data collection engine."""
        if self._running:
            logger.warning("HybridDataCollector is already running")
            return
        
        self._running = True
        logger.info("Starting HybridDataCollector...")
        
        # Start background update cycles for different priorities
        for priority in EdgePriority:
            task = asyncio.create_task(
                self._priority_update_cycle(priority),
                name=f"update_cycle_{priority.value}"
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        
        # Start statistics reporting
        stats_task = asyncio.create_task(
            self._periodic_stats_reporting(),
            name="stats_reporting"
        )
        self._background_tasks.add(stats_task)
        stats_task.add_done_callback(self._background_tasks.discard)
        
        logger.info("HybridDataCollector started successfully")
    
    async def stop(self) -> None:
        """Stop the data collection engine."""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping HybridDataCollector...")
        
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close WebSocket connections
        await self._close_websocket_connections()
        
        logger.info("HybridDataCollector stopped")
    
    async def _priority_update_cycle(self, priority: EdgePriority) -> None:
        """Background update cycle for edges of a specific priority."""
        interval = self.default_intervals[priority]
        
        logger.info(f"Starting update cycle for {priority.value} priority edges (interval: {interval}s)")
        
        while self._running:
            try:
                # Get edges for this priority
                edges_to_update = [
                    edge_id for edge_id, config in self.edge_configs.items()
                    if config.priority == priority and not config.use_websocket
                ]
                
                if edges_to_update:
                    logger.debug(f"Updating {len(edges_to_update)} {priority.value} priority edges")
                    await self._batch_update_edges(edges_to_update)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info(f"Update cycle for {priority.value} priority cancelled")
                break
            except Exception as e:
                logger.error(f"Error in {priority.value} priority update cycle: {e}")
                await asyncio.sleep(min(interval, 60))  # Wait before retrying
    
    async def _batch_update_edges(self, edge_ids: List[str]) -> None:
        """Update multiple edges in batch."""
        if not edge_ids:
            return
        
        # Create update tasks
        tasks = []
        for edge_id in edge_ids:
            task = asyncio.create_task(self._update_single_edge(edge_id))
            tasks.append(task)
        
        # Execute updates concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_updates = 0
        for edge_id, result in zip(edge_ids, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to update edge {edge_id}: {result}")
                self.stats.updates_failed += 1
            elif result:  # True means success
                successful_updates += 1
                self.stats.updates_performed += 1
            else:  # False means failure
                self.stats.updates_failed += 1
        
        logger.debug(f"Batch update completed: {successful_updates}/{len(edge_ids)} successful")
    
    async def _update_single_edge(self, edge_id: str) -> bool:
        """Update a single edge's state."""
        async with self.update_semaphore:
            async with self.update_locks[edge_id]:
                try:
                    start_time = time.time()
                    
                    # Check if update is needed
                    config = self.edge_configs.get(edge_id)
                    if not config:
                        return False
                    
                    last_update = self.last_update_times.get(edge_id, 0)
                    if time.time() - last_update < config.update_interval_seconds:
                        return True  # Skip update, too recent
                    
                    # Get edge and find responsible adapter
                    edge = self.graph.get_edge(edge_id)
                    if not edge:
                        return False
                    
                    adapter = self._find_adapter_for_edge(edge)
                    if not adapter:
                        logger.warning(f"No adapter found for edge {edge_id}")
                        return False
                    
                    # Update edge state
                    try:
                        new_state = await asyncio.wait_for(
                            adapter.update_edge_state(edge),
                            timeout=config.timeout_seconds
                        )
                        
                        # Update in graph and cache
                        self.graph.update_edge_state(edge_id, new_state)
                        await self._store_edge_state(edge_id, new_state)
                        
                        # Update timing
                        self.last_update_times[edge_id] = time.time()
                        
                        # Update latency statistics
                        latency_ms = (time.time() - start_time) * 1000
                        self._update_latency_stats(latency_ms)
                        
                        return True
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout updating edge {edge_id}")
                        await self._handle_edge_update_failure(edge, "timeout")
                        return False
                    
                except Exception as e:
                    logger.error(f"Error updating edge {edge_id}: {e}")
                    edge = self.graph.get_edge(edge_id)
                    if edge:
                        await self._handle_edge_update_failure(edge, str(e))
                    return False
    
    def _find_adapter_for_edge(self, edge: YieldGraphEdge) -> Optional[ProtocolAdapter]:
        """Find the adapter responsible for an edge."""
        for adapter in self.adapters:
            if (hasattr(adapter, 'PROTOCOL_NAME') and 
                adapter.PROTOCOL_NAME.lower() == edge.protocol_name.lower() and 
                adapter.chain_name.lower() == edge.chain_name.lower()):
                return adapter
        return None
    
    async def _handle_edge_update_failure(self, edge: YieldGraphEdge, error_msg: str) -> None:
        """Handle edge update failure by creating degraded state."""
        try:
            # Create degraded state with reduced confidence
            degraded_state = EdgeState(
                conversion_rate=edge.state.conversion_rate,
                liquidity_usd=edge.state.liquidity_usd,
                gas_cost_usd=edge.state.gas_cost_usd,
                delta_exposure=edge.state.delta_exposure,
                last_updated_timestamp=time.time(),
                confidence_score=max(0.1, edge.state.confidence_score * 0.5)
            )
            
            self.graph.update_edge_state(edge.edge_id, degraded_state)
            await self._store_edge_state(edge.edge_id, degraded_state)
            
            logger.warning(f"Created degraded state for {edge.edge_id}: {error_msg}")
            
        except Exception as e:
            logger.error(f"Failed to create degraded state for {edge.edge_id}: {e}")
    
    async def _store_edge_state(self, edge_id: str, state: EdgeState) -> None:
        """Store edge state in Redis."""
        try:
            state_json = state.model_dump_json()
            await self.redis_client.set(f"edge_state:{edge_id}", state_json)
            
        except Exception as e:
            logger.error(f"Failed to store edge state for {edge_id}: {e}")
    
    async def _get_edge_state(self, edge_id: str) -> Optional[EdgeState]:
        """Retrieve edge state from Redis."""
        try:
            state_json = await self.redis_client.get(f"edge_state:{edge_id}")
            if state_json:
                return EdgeState.model_validate_json(state_json)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get edge state for {edge_id}: {e}")
            return None
    
    def _update_latency_stats(self, latency_ms: float) -> None:
        """Update rolling average latency statistics."""
        current_avg = self.stats.avg_update_latency_ms
        total_updates = self.stats.updates_performed
        
        if total_updates == 0:
            self.stats.avg_update_latency_ms = latency_ms
        else:
            # Rolling average
            self.stats.avg_update_latency_ms = (
                (current_avg * (total_updates - 1) + latency_ms) / total_updates
            )
    
    async def _periodic_stats_reporting(self) -> None:
        """Periodically log collection statistics."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes
                
                self.stats.last_update_cycle = datetime.now(timezone.utc)
                
                logger.info(
                    f"HybridDataCollector Stats: "
                    f"Edges: {self.stats.total_edges}, "
                    f"Updates: {self.stats.updates_performed}, "
                    f"Failed: {self.stats.updates_failed}, "
                    f"Avg Latency: {self.stats.avg_update_latency_ms:.1f}ms, "
                    f"WebSocket Connections: {self.stats.websocket_connections}"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats reporting: {e}")
    
    async def _close_websocket_connections(self) -> None:
        """Close all WebSocket connections."""
        # Placeholder for future WebSocket cleanup
        logger.info("Closing WebSocket connections...")
        self.stats.websocket_connections = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics."""
        return {
            "total_edges": self.stats.total_edges,
            "edges_by_priority": dict(self.stats.edges_by_priority),
            "updates_performed": self.stats.updates_performed,
            "updates_failed": self.stats.updates_failed,
            "success_rate": (
                self.stats.updates_performed / 
                max(1, self.stats.updates_performed + self.stats.updates_failed)
            ) * 100,
            "websocket_connections": self.stats.websocket_connections,
            "avg_update_latency_ms": self.stats.avg_update_latency_ms,
            "last_update_cycle": self.stats.last_update_cycle.isoformat() if self.stats.last_update_cycle else None,
            "adapters_count": len(self.adapters),
            "background_tasks": len(self._background_tasks),
            "running": self._running
        }
    
    async def force_update_edge(self, edge_id: str) -> bool:
        """Force immediate update of a specific edge."""
        # Reset last update time to force update
        self.last_update_times[edge_id] = 0
        result = await self._update_single_edge(edge_id)
        
        # Update statistics
        if result:
            self.stats.updates_performed += 1
        else:
            self.stats.updates_failed += 1
            
        return result
    
    async def force_update_all(self) -> Dict[str, bool]:
        """Force update of all edges."""
        logger.info("Forcing update of all edges...")
        
        # Get all edge IDs before clearing times
        all_edge_ids = list(self.graph.edges.keys())
        
        # Reset all last update times to force updates
        self.last_update_times.clear()
        
        # Update all edges
        await self._batch_update_edges(all_edge_ids)
        
        # Clear times again after update to show they were reset
        self.last_update_times.clear()
        
        return {edge_id: True for edge_id in all_edge_ids}