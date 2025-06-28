"""Beam search pathfinding algorithm for yield arbitrage opportunities."""
import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

from ..graph_engine.models import UniversalYieldGraph, YieldGraphEdge, EdgeState, EdgeType
from ..data_collector.hybrid_collector import HybridDataCollector
from .path_models import SearchPath, PathNode, PathStatus
from .edge_state_manager import EdgeStateManager, StateRetrievalConfig
from .path_scorer import NonMLPathScorer, ScoringConfig, ScoringMethod
from .path_validator import PathValidator, ValidationConfig, ValidationResult

logger = logging.getLogger(__name__)



@dataclass
class BeamSearchConfig:
    """Configuration for beam search pathfinding."""
    beam_width: int = 100                    # Maximum paths to keep in beam
    max_path_length: int = 6                 # Maximum edges per path
    min_profit_threshold: float = 0.01       # Minimum profit to consider valid
    max_search_time_seconds: float = 30.0    # Maximum search time
    gas_price_gwei: float = 20.0            # Gas price for cost calculations
    slippage_tolerance: float = 0.01         # Maximum acceptable slippage
    confidence_threshold: float = 0.5        # Minimum confidence to use edge
    max_concurrent_updates: int = 20         # Max concurrent edge state updates


@dataclass
class SearchResult:
    """Result of a beam search operation."""
    paths: List[SearchPath]
    search_time_seconds: float
    edges_evaluated: int
    paths_pruned: int
    timeout_occurred: bool = False
    error_message: Optional[str] = None


class BeamSearchOptimizer:
    """
    Non-ML beam search pathfinding optimizer for yield arbitrage.
    
    This class implements a beam search algorithm to find profitable arbitrage
    paths through the yield graph. It maintains a beam of the most promising
    partial paths and expands them systematically.
    """
    
    def __init__(
        self,
        graph: UniversalYieldGraph,
        data_collector: HybridDataCollector,
        config: BeamSearchConfig = None,
        edge_state_manager: EdgeStateManager = None,
        path_scorer: NonMLPathScorer = None,
        path_validator: PathValidator = None
    ):
        """
        Initialize the beam search optimizer.
        
        Args:
            graph: The yield graph to search
            data_collector: Data collector for real-time edge state updates
            config: Search configuration parameters
            edge_state_manager: Optional edge state manager for advanced caching
            path_scorer: Optional path scorer for advanced scoring algorithms
            path_validator: Optional path validator for comprehensive path validation
        """
        self.graph = graph
        self.data_collector = data_collector
        self.config = config or BeamSearchConfig()
        
        # Edge state management
        if edge_state_manager:
            self.edge_state_manager = edge_state_manager
        else:
            # Create default edge state manager
            state_config = StateRetrievalConfig(
                memory_cache_ttl_seconds=300.0,
                max_memory_cache_size=5000,
                batch_size=min(50, self.config.max_concurrent_updates)
            )
            self.edge_state_manager = EdgeStateManager(
                data_collector=data_collector,
                config=state_config
            )
        
        # Path scoring
        if path_scorer:
            self.path_scorer = path_scorer
        else:
            # Create default path scorer with composite scoring
            scoring_config = ScoringConfig(
                method=ScoringMethod.COMPOSITE,
                gas_price_gwei=self.config.gas_price_gwei,
                min_liquidity_threshold=10_000.0,
                max_acceptable_slippage=self.config.slippage_tolerance
            )
            self.path_scorer = NonMLPathScorer(scoring_config)
        
        # Path validation
        if path_validator:
            self.path_validator = path_validator
        else:
            # Create default path validator
            validation_config = ValidationConfig(
                max_path_length=self.config.max_path_length,
                min_confidence_threshold=self.config.confidence_threshold,
                min_profit_threshold=self.config.min_profit_threshold,
                max_slippage_tolerance=self.config.slippage_tolerance
            )
            self.path_validator = PathValidator(validation_config)
        
        # Search state
        self._current_beam: List[SearchPath] = []
        self._completed_paths: List[SearchPath] = []
        self._search_stats = {
            "edges_evaluated": 0,
            "paths_pruned": 0,
            "state_updates_performed": 0,
            "cache_hits": 0,
            "fresh_fetches": 0
        }
        
        # Search performance tracking
        self._search_semaphore = asyncio.Semaphore(self.config.max_concurrent_updates)
        
    async def find_arbitrage_paths(
        self,
        start_asset_id: str,
        target_asset_id: str,
        initial_amount: float
    ) -> SearchResult:
        """
        Find arbitrage paths from start asset back to target asset.
        
        Args:
            start_asset_id: Asset to start the search from
            target_asset_id: Asset to return to (for arbitrage)
            initial_amount: Amount to start with
            
        Returns:
            SearchResult containing found paths and search statistics
        """
        search_start_time = time.time()
        
        try:
            logger.info(f"Starting beam search: {start_asset_id} -> {target_asset_id}, amount: {initial_amount}")
            
            # Initialize search state
            await self._initialize_search(start_asset_id, initial_amount)
            
            # Main beam search loop
            timeout_occurred = await self._execute_beam_search(target_asset_id, search_start_time)
            
            # Compile results
            search_time = time.time() - search_start_time
            
            # Sort and filter completed paths
            self._post_process_completed_paths()
            
            result = SearchResult(
                paths=self._completed_paths.copy(),
                search_time_seconds=search_time,
                edges_evaluated=self._search_stats["edges_evaluated"],
                paths_pruned=self._search_stats["paths_pruned"],
                timeout_occurred=timeout_occurred
            )
            
            logger.info(f"Beam search completed: {len(result.paths)} paths found in {search_time:.2f}s")
            return result
            
        except Exception as e:
            search_time = time.time() - search_start_time
            logger.error(f"Beam search failed: {e}")
            
            return SearchResult(
                paths=[],
                search_time_seconds=search_time,
                edges_evaluated=self._search_stats["edges_evaluated"],
                paths_pruned=self._search_stats["paths_pruned"],
                error_message=str(e)
            )
    
    async def _initialize_search(self, start_asset_id: str, initial_amount: float) -> None:
        """Initialize the beam search with the starting path."""
        # Reset search state
        self._current_beam = []
        self._completed_paths = []
        self._edge_state_cache = {}
        self._cache_timestamps = {}
        self._search_stats = {
            "edges_evaluated": 0,
            "paths_pruned": 0,
            "state_updates_performed": 0,
            "cache_hits": 0,
            "fresh_fetches": 0
        }
        
        # Create initial path
        initial_node = PathNode(
            asset_id=start_asset_id,
            amount=initial_amount,
            gas_cost_accumulated=0.0,
            confidence_accumulated=1.0,
            edge_path=[]
        )
        
        initial_path = SearchPath(
            nodes=[initial_node],
            total_score=0.0,
            status=PathStatus.ACTIVE
        )
        
        self._current_beam = [initial_path]
        
        logger.debug(f"Initialized beam search with starting asset {start_asset_id}, amount {initial_amount}")
    
    async def _execute_beam_search(self, target_asset_id: str, search_start_time: float) -> bool:
        """
        Execute the main beam search algorithm with enhanced logic.
        
        Args:
            target_asset_id: Asset ID to search for (end of arbitrage cycle)
            search_start_time: Time when search started
            
        Returns:
            True if timeout occurred, False otherwise
        """
        iteration = 0
        consecutive_empty_expansions = 0
        best_profit_seen = 0.0
        
        while self._current_beam and iteration < self.config.max_path_length:
            # Check timeout
            elapsed_time = time.time() - search_start_time
            if elapsed_time > self.config.max_search_time_seconds:
                logger.warning(f"Beam search timeout after {iteration} iterations ({elapsed_time:.2f}s)")
                return True
            
            iteration += 1
            logger.debug(f"Beam search iteration {iteration}, beam size: {len(self._current_beam)}")
            
            # Expand all paths in current beam
            next_beam = await self._expand_beam(target_asset_id)
            
            # Check for expansion progress
            if not next_beam:
                consecutive_empty_expansions += 1
                if consecutive_empty_expansions >= 2:
                    logger.debug("No beam expansion progress for 2 consecutive iterations, terminating")
                    break
            else:
                consecutive_empty_expansions = 0
            
            # Smart beam pruning with adaptive sizing
            self._current_beam = self._prune_beam_adaptive(next_beam, iteration)
            
            # Check for profitable paths and early termination conditions
            current_best_profit = self._get_best_profit()
            if current_best_profit > best_profit_seen:
                best_profit_seen = current_best_profit
                logger.debug(f"New best profit found: {current_best_profit:.6f}")
            
            # Early termination conditions
            if self._should_terminate_early(iteration, best_profit_seen):
                logger.debug(f"Early termination triggered at iteration {iteration}")
                break
            
            # Active path check
            if not any(path.status == PathStatus.ACTIVE for path in self._current_beam):
                logger.debug("No active paths remaining, terminating search")
                break
        
        logger.info(f"Beam search completed after {iteration} iterations, best profit: {best_profit_seen:.6f}")
        return False
    
    async def _expand_beam(self, target_asset_id: str) -> List[SearchPath]:
        """
        Expand all paths in the current beam by one edge.
        Uses optimized batch edge state retrieval for better performance.
        
        Args:
            target_asset_id: Target asset for arbitrage completion
            
        Returns:
            List of expanded paths
        """
        # Use optimized batch expansion by default
        return await self._expand_beam_optimized(target_asset_id)
    
    async def _expand_beam_optimized(self, target_asset_id: str) -> List[SearchPath]:
        """
        Optimized beam expansion using batch edge state retrieval.
        
        Args:
            target_asset_id: Target asset for arbitrage completion
            
        Returns:
            List of expanded paths with batch optimization
        """
        # Collect all edges that need state updates
        all_edge_ids = set()
        path_edge_mapping = []  # Use list instead of dict
        
        for i, path in enumerate(self._current_beam):
            if path.status == PathStatus.ACTIVE:
                current_asset = path.end_asset
                outgoing_edges = self.graph.get_edges_from(current_asset)
                
                valid_edges = []
                for edge in outgoing_edges:
                    # Skip if edge already used in this path
                    if edge.edge_id not in path.nodes[-1].edge_path:
                        # Additional filtering for edge quality
                        if self._is_edge_worth_exploring(edge, path, target_asset_id):
                            valid_edges.append(edge)
                            all_edge_ids.add(edge.edge_id)
                
                if valid_edges:
                    path_edge_mapping.append((path, valid_edges))
        
        # Batch retrieve all edge states
        if all_edge_ids:
            edge_states = await self.edge_state_manager.get_edge_states_batch(list(all_edge_ids))
        else:
            edge_states = {}
        
        # Update statistics
        for edge_id, state in edge_states.items():
            if state:
                self._search_stats["cache_hits"] += 1
            else:
                self._search_stats["fresh_fetches"] += 1
        
        # Expand paths using cached states
        next_beam = []
        for path, edges in path_edge_mapping:
            expanded_paths = await self._expand_single_path_with_states(
                path, edges, edge_states, target_asset_id
            )
            next_beam.extend(expanded_paths)
        
        return next_beam
    
    async def _expand_single_path_with_states(
        self, 
        path: SearchPath, 
        edges: List[YieldGraphEdge],
        edge_states: Dict[str, Optional[EdgeState]],
        target_asset_id: str
    ) -> List[SearchPath]:
        """
        Expand a single path using pre-fetched edge states.
        
        Args:
            path: Path to expand
            edges: List of edges to consider
            edge_states: Pre-fetched edge states
            target_asset_id: Target asset for arbitrage completion
            
        Returns:
            List of new paths created by expanding the given path
        """
        current_amount = path.final_amount
        expanded_paths = []
        
        for edge in edges:
            try:
                # Get pre-fetched edge state
                edge_state = edge_states.get(edge.edge_id)
                if not edge_state:
                    continue
                
                # Validate edge state meets minimum requirements
                if not await self._validate_edge_state(edge, edge_state):
                    continue
                
                # Calculate conversion
                conversion_result = await self._calculate_edge_conversion(
                    edge, edge_state, current_amount
                )
                
                if not conversion_result["success"]:
                    continue
                
                # Create new path node
                new_node = PathNode(
                    asset_id=edge.target_asset_id,
                    amount=conversion_result["output_amount"],
                    gas_cost_accumulated=path.total_gas_cost + conversion_result["gas_cost"],
                    confidence_accumulated=path.nodes[-1].confidence_accumulated * edge_state.confidence_score,
                    edge_path=path.nodes[-1].edge_path + [edge.edge_id]
                )
                
                # Create new path
                new_path = SearchPath(
                    nodes=path.nodes + [new_node],
                    total_score=0.0,  # Will be calculated by scoring function
                    status=PathStatus.ACTIVE
                )
                
                # Validate path before adding to results
                validation_report = await self.path_validator.validate_path(
                    new_path, self.graph, target_asset_id, path.nodes[0].amount
                )
                
                if not validation_report.is_valid:
                    new_path.status = PathStatus.INVALID
                    logger.debug(f"Path validation failed: {validation_report.errors}")
                    continue
                
                # Check if path completes arbitrage cycle
                if edge.target_asset_id == target_asset_id:
                    new_path.status = PathStatus.COMPLETE
                    # Calculate final score for completed path
                    new_path.total_score = await self._score_completed_path(new_path)
                    
                    # Only keep profitable completed paths
                    if new_path.net_profit >= self.config.min_profit_threshold:
                        self._completed_paths.append(new_path)
                        logger.debug(f"Found profitable path: {new_path.net_profit:.4f} profit")
                else:
                    # Score partial path
                    new_path.total_score = await self._score_partial_path(new_path, target_asset_id)
                
                expanded_paths.append(new_path)
                self._search_stats["edges_evaluated"] += 1
                
            except Exception as e:
                logger.warning(f"Error expanding path with edge {edge.edge_id}: {e}")
                continue
        
        return expanded_paths
    
    async def _expand_single_path(self, path: SearchPath, target_asset_id: str) -> List[SearchPath]:
        """
        Expand a single path by exploring all possible next edges.
        
        Args:
            path: Path to expand
            target_asset_id: Target asset for arbitrage completion
            
        Returns:
            List of new paths created by expanding the given path
        """
        current_asset = path.end_asset
        current_amount = path.final_amount
        expanded_paths = []
        
        # Get outgoing edges from current asset
        outgoing_edges = self.graph.get_edges_from(current_asset)
        
        for edge in outgoing_edges:
            try:
                # Skip if edge already used in this path (prevent immediate cycles)
                if edge.edge_id in path.nodes[-1].edge_path:
                    continue
                
                # Additional filtering for edge quality
                if not self._is_edge_worth_exploring(edge, path, target_asset_id):
                    continue
                
                # Get current edge state
                edge_state = await self._get_edge_state(edge.edge_id)
                if not edge_state:
                    continue
                
                # Validate edge state meets minimum requirements
                if not await self._validate_edge_state(edge, edge_state):
                    continue
                
                # Calculate conversion
                conversion_result = await self._calculate_edge_conversion(
                    edge, edge_state, current_amount
                )
                
                if not conversion_result["success"]:
                    continue
                
                # Create new path node
                new_node = PathNode(
                    asset_id=edge.target_asset_id,
                    amount=conversion_result["output_amount"],
                    gas_cost_accumulated=path.total_gas_cost + conversion_result["gas_cost"],
                    confidence_accumulated=path.nodes[-1].confidence_accumulated * edge_state.confidence_score,
                    edge_path=path.nodes[-1].edge_path + [edge.edge_id]
                )
                
                # Create new path
                new_path = SearchPath(
                    nodes=path.nodes + [new_node],
                    total_score=0.0,  # Will be calculated by scoring function
                    status=PathStatus.ACTIVE
                )
                
                # Validate path before adding to results
                validation_report = await self.path_validator.validate_path(
                    new_path, self.graph, target_asset_id, path.nodes[0].amount
                )
                
                if not validation_report.is_valid:
                    new_path.status = PathStatus.INVALID
                    logger.debug(f"Path validation failed: {validation_report.errors}")
                    continue
                
                # Check if path completes arbitrage cycle
                if edge.target_asset_id == target_asset_id:
                    new_path.status = PathStatus.COMPLETE
                    # Calculate final score for completed path
                    new_path.total_score = await self._score_completed_path(new_path)
                    
                    # Only keep profitable completed paths
                    if new_path.net_profit >= self.config.min_profit_threshold:
                        self._completed_paths.append(new_path)
                        logger.debug(f"Found profitable path: {new_path.net_profit:.4f} profit")
                else:
                    # Score partial path
                    new_path.total_score = await self._score_partial_path(new_path, target_asset_id)
                
                expanded_paths.append(new_path)
                self._search_stats["edges_evaluated"] += 1
                
            except Exception as e:
                logger.warning(f"Error expanding path with edge {edge.edge_id}: {e}")
                continue
        
        return expanded_paths
    
    def _prune_beam(self, paths: List[SearchPath]) -> List[SearchPath]:
        """
        Prune the beam to maintain maximum beam width.
        
        Args:
            paths: All paths to consider for the beam
            
        Returns:
            Pruned list of paths within beam width limit
        """
        # Separate active and completed paths
        active_paths = [p for p in paths if p.status == PathStatus.ACTIVE]
        
        # Sort active paths by score (descending)
        active_paths.sort(key=lambda p: p.total_score, reverse=True)
        
        # Keep top paths within beam width
        if len(active_paths) > self.config.beam_width:
            pruned_count = len(active_paths) - self.config.beam_width
            self._search_stats["paths_pruned"] += pruned_count
            
            # Mark pruned paths
            for path in active_paths[self.config.beam_width:]:
                path.status = PathStatus.PRUNED
            
            active_paths = active_paths[:self.config.beam_width]
        
        logger.debug(f"Pruned beam: {len(active_paths)} active paths remaining")
        return active_paths
    
    def _prune_beam_adaptive(self, paths: List[SearchPath], iteration: int) -> List[SearchPath]:
        """
        Adaptive beam pruning that adjusts beam size based on search progress.
        
        Args:
            paths: All paths to consider for the beam
            iteration: Current search iteration
            
        Returns:
            Pruned list of paths with adaptive beam width
        """
        # Separate active and completed paths
        active_paths = [p for p in paths if p.status == PathStatus.ACTIVE]
        
        # Calculate adaptive beam width
        base_width = self.config.beam_width
        
        # Reduce beam width in later iterations to focus search
        if iteration > 3:
            reduction_factor = 0.8 ** (iteration - 3)
            adaptive_width = max(int(base_width * reduction_factor), 10)  # Minimum 10 paths
        else:
            adaptive_width = base_width
        
        # Increase beam width if we're finding very profitable paths
        if self._completed_paths:
            best_profit = max(path.net_profit for path in self._completed_paths)
            if best_profit > self.config.min_profit_threshold * 10:  # Very profitable
                adaptive_width = int(adaptive_width * 1.5)
        
        logger.debug(f"Adaptive beam width: {adaptive_width} (base: {base_width}, iteration: {iteration})")
        
        # Sort and prune
        active_paths.sort(key=lambda p: p.total_score, reverse=True)
        
        if len(active_paths) > adaptive_width:
            pruned_count = len(active_paths) - adaptive_width
            self._search_stats["paths_pruned"] += pruned_count
            
            # Mark pruned paths
            for path in active_paths[adaptive_width:]:
                path.status = PathStatus.PRUNED
            
            active_paths = active_paths[:adaptive_width]
        
        return active_paths
    
    def _get_best_profit(self) -> float:
        """Get the best profit from completed paths."""
        if not self._completed_paths:
            return 0.0
        return max(path.net_profit for path in self._completed_paths)
    
    def _should_terminate_early(self, iteration: int, best_profit: float) -> bool:
        """
        Determine if search should terminate early based on current progress.
        
        Args:
            iteration: Current iteration number
            best_profit: Best profit found so far
            
        Returns:
            True if search should terminate early
        """
        # Don't terminate too early
        if iteration < 2:
            return False
        
        # Terminate if we found excellent profit early
        if best_profit > self.config.min_profit_threshold * 50:  # 50x minimum threshold
            logger.debug(f"Terminating early due to excellent profit: {best_profit:.6f}")
            return True
        
        # Terminate if we have many completed paths and limited beam
        if len(self._completed_paths) >= 5 and len(self._current_beam) < 3:
            return True
        
        # Terminate if beam is getting very small in later iterations
        if iteration > 4 and len(self._current_beam) < 2:
            return True
        
        return False
    
    def _post_process_completed_paths(self) -> None:
        """
        Post-process completed paths to filter and sort them optimally.
        """
        if not self._completed_paths:
            return
        
        # Remove duplicate paths (same sequence of assets)
        unique_paths = []
        seen_sequences = set()
        
        for path in self._completed_paths:
            asset_sequence = tuple(node.asset_id for node in path.nodes)
            if asset_sequence not in seen_sequences:
                seen_sequences.add(asset_sequence)
                unique_paths.append(path)
        
        # Sort by a composite score: net_profit * confidence * efficiency
        def path_composite_score(path: SearchPath) -> float:
            confidence_factor = path.nodes[-1].confidence_accumulated if path.nodes else 0.5
            efficiency_factor = 1.0 / (1.0 + path.path_length * 0.1)  # Shorter paths preferred
            gas_efficiency = 1.0 / (1.0 + path.total_gas_cost * 0.01)  # Lower gas preferred
            
            return path.net_profit * confidence_factor * efficiency_factor * gas_efficiency
        
        unique_paths.sort(key=path_composite_score, reverse=True)
        
        # Limit to top N paths to avoid memory issues
        max_paths = min(50, len(unique_paths))
        self._completed_paths = unique_paths[:max_paths]
        
        logger.info(f"Post-processed {len(self._completed_paths)} unique profitable paths")
    
    def _is_edge_worth_exploring(self, edge: YieldGraphEdge, current_path: SearchPath, target_asset_id: str) -> bool:
        """
        Determine if an edge is worth exploring based on heuristics.
        
        Args:
            edge: The edge to evaluate
            current_path: Current path being extended
            target_asset_id: Target asset for arbitrage completion
            
        Returns:
            True if edge should be explored
        """
        # Always explore edges that complete the arbitrage cycle
        if edge.target_asset_id == target_asset_id:
            return True
        
        # Filter by edge type - prioritize high-value edge types
        priority_edge_types = {EdgeType.TRADE, EdgeType.FLASH_LOAN, EdgeType.BACK_RUN}
        if edge.edge_type not in priority_edge_types and current_path.path_length >= 2:
            return False
        
        # Avoid revisiting same chain too often (for path diversity)
        if current_path.path_length >= 2:
            path_chains = set()
            for node in current_path.nodes[1:]:  # Skip starting node
                if node.edge_path:
                    last_edge_id = node.edge_path[-1]
                    last_edge = self.graph.get_edge(last_edge_id)
                    if last_edge:
                        path_chains.add(last_edge.chain_name)
            
            # If path already uses 2+ different chains, be conservative about adding more
            if len(path_chains) >= 2 and edge.chain_name not in path_chains:
                return False
        
        # Prioritize edges from well-known protocols
        trusted_protocols = {'uniswapv3', 'uniswapv2', 'sushiswap', 'curve', 'balancer', 'aave', 'compound'}
        if current_path.path_length >= 3 and edge.protocol_name not in trusted_protocols:
            return False
        
        # For longer paths, be more selective
        if current_path.path_length >= 4:
            # Only continue if target asset is "close" (reachable in 1-2 hops)
            if not self._is_target_reachable(edge.target_asset_id, target_asset_id, max_hops=2):
                return False
        
        return True
    
    def _is_target_reachable(self, from_asset: str, target_asset: str, max_hops: int) -> bool:
        """
        Check if target asset is reachable within max_hops.
        
        Args:
            from_asset: Starting asset
            target_asset: Target asset to reach
            max_hops: Maximum number of hops to search
            
        Returns:
            True if target is reachable within max_hops
        """
        if from_asset == target_asset:
            return True
        
        if max_hops <= 0:
            return False
        
        # Simple BFS to check reachability
        visited = {from_asset}
        current_level = {from_asset}
        
        for hop in range(max_hops):
            next_level = set()
            for asset in current_level:
                edges = self.graph.get_edges_from(asset)
                for edge in edges:
                    if edge.target_asset_id == target_asset:
                        return True
                    if edge.target_asset_id not in visited:
                        visited.add(edge.target_asset_id)
                        next_level.add(edge.target_asset_id)
            
            if not next_level:
                break
            current_level = next_level
        
        return False
    
    async def _get_edge_state(self, edge_id: str) -> Optional[EdgeState]:
        """
        Retrieve current edge state using the EdgeStateManager.
        
        Args:
            edge_id: ID of the edge to get state for
            
        Returns:
            Current EdgeState or None if unavailable
        """
        try:
            state = await self.edge_state_manager.get_edge_state(edge_id)
            
            if state:
                # Update search statistics based on cache level
                manager_metrics = self.edge_state_manager.get_metrics()
                cache_stats = manager_metrics.get("cache_stats", {})
                
                # Estimate if this was a cache hit or fresh fetch
                total_requests = cache_stats.get("total_requests", 0)
                fresh_fetches = cache_stats.get("fresh_fetches", 0)
                
                if fresh_fetches > self._search_stats.get("fresh_fetches", 0):
                    self._search_stats["fresh_fetches"] += 1
                    self._search_stats["state_updates_performed"] += 1
                else:
                    self._search_stats["cache_hits"] += 1
                
                # Trigger predictive caching for next likely edges
                await self.edge_state_manager.prefetch_likely_edges(edge_id)
            
            return state
            
        except Exception as e:
            logger.warning(f"Failed to get edge state for {edge_id}: {e}")
            return None
    
    async def _validate_edge_state(self, edge: YieldGraphEdge, state: EdgeState) -> bool:
        """
        Validate that edge state meets minimum requirements for pathfinding.
        
        Args:
            edge: The edge to validate
            state: Current state of the edge
            
        Returns:
            True if edge state is valid for pathfinding
        """
        # Check confidence threshold
        if state.confidence_score < self.config.confidence_threshold:
            return False
        
        # Check conversion rate exists
        if state.conversion_rate is None or state.conversion_rate <= 0:
            return False
        
        # Check liquidity if available
        if state.liquidity_usd is not None and state.liquidity_usd < 1000:  # Minimum $1000 liquidity
            return False
        
        # Additional validation can be added here
        return True
    
    async def _calculate_edge_conversion(
        self, 
        edge: YieldGraphEdge, 
        state: EdgeState, 
        input_amount: float
    ) -> Dict[str, Any]:
        """
        Calculate the conversion result for an edge.
        
        Args:
            edge: The edge to calculate conversion for
            state: Current state of the edge
            input_amount: Amount to convert
            
        Returns:
            Dictionary with conversion results
        """
        try:
            # Use edge's calculate_output method
            result = edge.calculate_output(input_amount, state)
            
            if "error" in result:
                return {"success": False, "error": result["error"]}
            
            output_amount = result.get("output_amount", 0.0)
            gas_cost = result.get("gas_cost_usd", state.gas_cost_usd or 15.0)
            
            # Apply slippage tolerance
            slippage_factor = 1.0 - self.config.slippage_tolerance
            adjusted_output = output_amount * slippage_factor
            
            return {
                "success": True,
                "output_amount": adjusted_output,
                "gas_cost": gas_cost,
                "slippage_applied": self.config.slippage_tolerance
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _score_partial_path(self, path: SearchPath, target_asset_id: str) -> float:
        """
        Score a partial path to estimate its potential using NonMLPathScorer.
        
        Args:
            path: The partial path to score
            target_asset_id: Target asset for arbitrage completion
            
        Returns:
            Score for the partial path (higher is better)
        """
        try:
            # Use the advanced path scorer
            initial_amount = path.nodes[0].amount if path.nodes else 1.0
            scoring_breakdown = await self.path_scorer.score_path(
                path, target_asset_id, initial_amount
            )
            
            # Log warnings if any
            if scoring_breakdown.warnings:
                logger.debug(f"Path scoring warnings: {scoring_breakdown.warnings}")
            
            return scoring_breakdown.total_score
            
        except Exception as e:
            logger.warning(f"Error scoring partial path: {e}")
            # Fallback to simple scoring
            return await self._score_simple_fallback(path)
    
    async def _score_completed_path(self, path: SearchPath) -> float:
        """
        Score a completed arbitrage path using NonMLPathScorer.
        
        Args:
            path: The completed path to score
            
        Returns:
            Score for the completed path (higher is better)
        """
        try:
            # Use the advanced path scorer for completed paths
            initial_amount = path.nodes[0].amount if path.nodes else 1.0
            scoring_breakdown = await self.path_scorer.score_path(
                path, path.end_asset, initial_amount
            )
            
            # Log any risk flags for completed paths
            if scoring_breakdown.risk_flags:
                logger.warning(f"Path risk flags: {scoring_breakdown.risk_flags}")
            
            # For completed paths, boost score slightly to prioritize completion
            completion_bonus = 0.1 if path.status == PathStatus.COMPLETE else 0.0
            
            return scoring_breakdown.total_score + completion_bonus
            
        except Exception as e:
            logger.warning(f"Error scoring completed path: {e}")
            # Fallback to simple scoring
            return await self._score_simple_fallback(path)
    
    async def _score_simple_fallback(self, path: SearchPath) -> float:
        """
        Simple fallback scoring when advanced scoring fails.
        
        Args:
            path: The path to score
            
        Returns:
            Simple score based on basic metrics
        """
        if not path.nodes:
            return 0.0
        
        final_amount = path.final_amount
        confidence = path.nodes[-1].confidence_accumulated
        gas_cost = path.total_gas_cost
        
        # Simple scoring: amount * confidence - gas_cost - path_length_penalty
        path_length_penalty = path.path_length * 0.001
        
        score = (final_amount * confidence) - gas_cost - path_length_penalty
        return max(0.0, score)
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get current search statistics."""
        stats = {
            "current_beam_size": len(self._current_beam),
            "completed_paths": len(self._completed_paths),
            "edge_state_cache_size": len(getattr(self, '_edge_state_cache', {})),
            **self._search_stats
        }
        
        # Add validation statistics
        validation_stats = self.path_validator.get_validation_stats()
        stats["validation"] = validation_stats
        
        # Add edge state manager statistics
        if hasattr(self.edge_state_manager, 'get_metrics'):
            edge_manager_stats = self.edge_state_manager.get_metrics()
            stats["edge_state_manager"] = edge_manager_stats
        
        # Add path scorer statistics
        if hasattr(self.path_scorer, 'get_scoring_stats'):
            scorer_stats = self.path_scorer.get_scoring_stats()
            stats["path_scorer"] = scorer_stats
        
        return stats
    
    def get_search_progress(self) -> Dict[str, Any]:
        """
        Get real-time search progress information.
        
        Returns:
            Dictionary with current search progress metrics
        """
        current_best_profit = self._get_best_profit()
        
        return {
            "current_beam_size": len(self._current_beam),
            "completed_paths_count": len(self._completed_paths),
            "best_profit_found": current_best_profit,
            "profitable_paths_count": len([p for p in self._completed_paths if p.net_profit > 0]),
            "search_stats": self._search_stats.copy(),
            "top_3_profits": sorted([p.net_profit for p in self._completed_paths], reverse=True)[:3]
        }