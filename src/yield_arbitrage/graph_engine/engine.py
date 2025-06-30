"""Graph Engine Implementation for Yield Arbitrage Discovery."""
import asyncio
import logging
from typing import Dict, List, Optional, Any

from .models import UniversalYieldGraph
from ..pathfinding.beam_search import BeamSearchOptimizer, BeamSearchConfig
from ..data_collector.hybrid_collector import HybridDataCollector
from ..protocols.adapter_registry import ProtocolAdapterRegistry
from ..cache import get_redis

logger = logging.getLogger(__name__)


class GraphEngine:
    """Production-ready graph engine for arbitrage opportunity discovery."""
    
    def __init__(self):
        self.graph: Optional[UniversalYieldGraph] = None
        self.data_collector: Optional[HybridDataCollector] = None
        self.pathfinder: Optional[BeamSearchOptimizer] = None
        self.adapter_registry: Optional[ProtocolAdapterRegistry] = None
        self.redis_client = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the graph engine with real components."""
        logger.info("🚀 Initializing Graph Engine...")
        
        try:
            # Initialize Redis client
            logger.info("📦 Getting Redis client...")
            self.redis_client = await get_redis()
            
            # Initialize the graph
            logger.info("📊 Creating Universal Yield Graph...")
            self.graph = UniversalYieldGraph()
            
            # Initialize adapter registry and load initial protocol adapters
            logger.info("🔌 Loading protocol adapters...")
            await self._initialize_adapters()
            
            # Initialize data collector with dependencies
            logger.info("📡 Setting up data collector...")
            await self._initialize_data_collector()
            
            # Initialize pathfinding with beam search
            logger.info("🔍 Setting up pathfinder...")
            await self._initialize_pathfinder()
            
            self.is_initialized = True
            logger.info("✅ Graph Engine initialized successfully")
            logger.info(f"   Graph has {len(self.graph.edges)} edges")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Graph Engine: {e}")
            raise
    
    async def _initialize_adapters(self):
        """Initialize and load protocol adapters."""
        try:
            from ..protocols.production_registry import ProductionAdapterRegistry
            self.adapter_registry = ProductionAdapterRegistry()
            await self.adapter_registry.initialize()
            
            # Load edges from adapters into graph
            adapters = await self.adapter_registry.get_all_adapters()
            total_edges = 0
            
            for adapter in adapters:
                try:
                    edges = await adapter.get_available_edges()
                    for edge in edges:
                        self.graph.add_edge(edge)
                        total_edges += 1
                    logger.debug(f"Loaded {len(edges)} edges from {adapter.protocol_name}")
                except Exception as e:
                    logger.warning(f"Failed to load edges from {adapter.protocol_name}: {e}")
            
            logger.info(f"✅ Loaded {total_edges} edges from {len(adapters)} adapters")
            
        except ImportError:
            logger.warning("ProductionAdapterRegistry not available, using minimal setup")
            # Create basic adapters for essential protocols
            await self._create_basic_adapters()
    
    async def _create_basic_adapters(self):
        """Create basic adapters for essential protocols when full registry isn't available."""
        logger.info("🔧 Creating basic protocol adapters...")
        
        try:
            from ..protocols.uniswap_v3_adapter import UniswapV3Adapter
            
            # Initialize Uniswap V3 adapter
            uniswap_adapter = UniswapV3Adapter()
            await uniswap_adapter.initialize()
            
            # Load edges from Uniswap V3
            edges = await uniswap_adapter.get_available_edges()
            for edge in edges:
                self.graph.add_edge(edge)
            
            logger.info(f"✅ Loaded {len(edges)} edges from Uniswap V3")
            
        except Exception as e:
            logger.warning(f"Failed to create basic adapters: {e}")
            # Continue without adapters - we can still use cached data
    
    async def _initialize_data_collector(self):
        """Initialize data collector with proper dependencies."""
        try:
            # Get list of adapters for data collector
            adapters = []
            if self.adapter_registry:
                adapters = await self.adapter_registry.get_all_adapters()
            
            self.data_collector = HybridDataCollector(
                graph=self.graph,
                redis_client=self.redis_client,
                adapters=adapters,
                enable_websockets=True
            )
            
            await self.data_collector.initialize()
            logger.info("✅ Data collector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize data collector: {e}")
            # Continue without data collector - pathfinder can work with static data
    
    async def _initialize_pathfinder(self):
        """Initialize beam search pathfinder."""
        try:
            config = BeamSearchConfig(
                beam_width=100,
                max_path_length=8,
                min_profit_threshold=1.0,  # $1 minimum profit
                max_search_time_seconds=30.0
            )
            
            self.pathfinder = BeamSearchOptimizer(
                graph=self.graph,
                data_collector=self.data_collector,
                config=config
            )
            
            logger.info("✅ Pathfinder initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize pathfinder: {e}")
            raise  # Pathfinder is essential
    
    async def discover_opportunities(self, 
                                   start_token: str, 
                                   amount: float = 1000.0,
                                   min_profit_usd: float = 1.0) -> List[Dict[str, Any]]:
        """Discover arbitrage opportunities using beam search."""
        if not self.is_initialized or not self.pathfinder:
            raise RuntimeError("Graph engine not properly initialized")
        
        logger.info(f"🔍 Discovering opportunities from {start_token} with {amount} USD")
        
        try:
            # Use beam search to find profitable paths
            search_result = await self.pathfinder.find_profitable_paths(
                start_token=start_token,
                start_amount=amount,
                min_profit_usd=min_profit_usd
            )
            
            # Convert search paths to opportunity format
            opportunities = []
            for path in search_result.paths:
                if path.status == "complete" and path.net_profit > min_profit_usd:
                    opportunity = {
                        "path_id": f"path_{len(opportunities)}",
                        "start_token": path.start_asset,
                        "end_token": path.end_asset,
                        "start_amount": amount,
                        "expected_output": path.final_amount,
                        "expected_profit": path.net_profit,
                        "path_length": path.path_length,
                        "confidence": path.total_score,
                        "gas_cost": path.total_gas_cost,
                        "edges": [node.edge_path for node in path.nodes],
                        "discovery_time": search_result.search_time_seconds
                    }
                    opportunities.append(opportunity)
            
            logger.info(f"✅ Found {len(opportunities)} profitable opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Failed to discover opportunities: {e}")
            return []
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get current graph statistics."""
        if not self.is_initialized:
            return {"error": "Graph engine not initialized"}
        
        return {
            "total_edges": len(self.graph.edges),
            "total_nodes": len(self.graph.nodes),
            "is_data_collector_active": self.data_collector is not None,
            "is_pathfinder_ready": self.pathfinder is not None,
            "adapter_count": len(await self.adapter_registry.get_all_adapters()) if self.adapter_registry else 0
        }
    
    async def shutdown(self):
        """Shutdown the graph engine."""
        logger.info("🛑 Shutting down Graph Engine...")
        
        if self.data_collector:
            await self.data_collector.shutdown()
        
        if self.adapter_registry:
            await self.adapter_registry.shutdown()
        
        self.is_initialized = False
        logger.info("✅ Graph Engine shutdown complete")


# Global graph engine instance
_graph_engine: Optional[GraphEngine] = None


async def get_graph_engine() -> GraphEngine:
    """Get the global graph engine instance."""
    global _graph_engine
    
    if _graph_engine is None:
        _graph_engine = GraphEngine()
        await _graph_engine.initialize()
    
    return _graph_engine


async def initialize_graph_engine() -> GraphEngine:
    """Initialize the global graph engine."""
    return await get_graph_engine()


async def shutdown_graph_engine():
    """Shutdown the global graph engine."""
    global _graph_engine
    
    if _graph_engine is not None:
        await _graph_engine.shutdown()
        _graph_engine = None