"""
Modular MEV-Aware Execution Router.

This module provides chain-agnostic execution routing based on MEV risk assessment.
It supports Flashbots for Ethereum and private nodes/relays for other chains.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from enum import Enum

from yield_arbitrage.mev_protection.mev_risk_assessor import PathMEVAnalysis, MEVRiskLevel
from yield_arbitrage.execution.enhanced_transaction_builder import BatchExecutionPlan, RouterTransaction

logger = logging.getLogger(__name__)


class ExecutionMethod(str, Enum):
    """Available execution methods across chains."""
    PUBLIC = "public"                    # Standard public mempool
    PUBLIC_PROTECTED = "public_protected" # Public with MEV protection
    PRIVATE_NODE = "private_node"        # Private node submission
    PRIVATE_RELAY = "private_relay"      # Private relay service
    FLASHBOTS = "flashbots"             # Flashbots (Ethereum only)
    FLASHBOTS_BUNDLE = "flashbots_bundle" # Flashbots bundle
    CUSTOM_RELAY = "custom_relay"        # Chain-specific relay


@dataclass
class ExecutionRoute:
    """Selected execution route with all parameters."""
    method: ExecutionMethod
    endpoint: str
    auth_required: bool = False
    
    # Method-specific parameters
    flashbots_params: Optional[Dict[str, Any]] = None
    relay_params: Optional[Dict[str, Any]] = None
    node_params: Optional[Dict[str, Any]] = None
    
    # Transaction parameters
    priority_fee_wei: int = 0
    max_fee_per_gas_wei: int = 0
    bundle_id: Optional[str] = None
    
    # Monitoring
    expected_confirmation_time: int = 15  # seconds
    fallback_method: Optional[ExecutionMethod] = None


@dataclass
class ChainExecutionConfig:
    """Configuration for execution methods on a specific chain."""
    chain_id: int
    chain_name: str
    
    # Available methods
    supports_flashbots: bool = False
    private_node_url: Optional[str] = None
    private_relay_url: Optional[str] = None
    custom_relay_config: Optional[Dict[str, Any]] = None
    
    # Default endpoints
    public_rpc: str = ""
    flashbots_relay: str = ""
    
    # Chain-specific parameters
    min_priority_fee: int = 1_000_000_000  # 1 gwei default
    max_priority_fee: int = 100_000_000_000  # 100 gwei default
    block_time: int = 12  # seconds
    
    # MEV protection features
    has_private_mempool: bool = False
    has_sequencer: bool = False  # L2s with sequencers
    mev_protection_native: bool = False


class ExecutionProvider(ABC):
    """Abstract base class for execution providers."""
    
    @abstractmethod
    async def submit_transaction(
        self,
        transaction: RouterTransaction,
        route: ExecutionRoute
    ) -> Dict[str, Any]:
        """Submit transaction through this provider."""
        pass
    
    @abstractmethod
    async def check_inclusion(
        self,
        tx_hash: str,
        route: ExecutionRoute
    ) -> Dict[str, Any]:
        """Check if transaction was included."""
        pass
    
    @abstractmethod
    def supports_method(self, method: ExecutionMethod) -> bool:
        """Check if this provider supports the given method."""
        pass


class PublicExecutionProvider(ExecutionProvider):
    """Standard public mempool execution."""
    
    def __init__(self, rpc_url: str):
        self.rpc_url = rpc_url
    
    async def submit_transaction(
        self,
        transaction: RouterTransaction,
        route: ExecutionRoute
    ) -> Dict[str, Any]:
        """Submit to public mempool."""
        logger.info(f"Submitting transaction {transaction.segment_id} to public mempool")
        
        # In production, this would use web3.py or ethers
        return {
            "success": True,
            "tx_hash": f"0x{'0' * 64}",  # Mock hash
            "method": "public",
            "submitted_at": 1234567890
        }
    
    async def check_inclusion(self, tx_hash: str, route: ExecutionRoute) -> Dict[str, Any]:
        """Check transaction status."""
        return {
            "included": True,
            "block_number": 12345678,
            "confirmations": 12
        }
    
    def supports_method(self, method: ExecutionMethod) -> bool:
        """Public provider supports public methods."""
        return method in [ExecutionMethod.PUBLIC, ExecutionMethod.PUBLIC_PROTECTED]


class FlashbotsProvider(ExecutionProvider):
    """Flashbots execution provider for Ethereum."""
    
    def __init__(self, private_key: str, relay_url: Optional[str] = None):
        self.private_key = private_key
        self.relay_url = relay_url
        self.flashbots_client = None
    
    async def initialize(self):
        """Initialize Flashbots client."""
        from yield_arbitrage.mev_protection.flashbots_client import FlashbotsClient, FlashbotsNetwork
        
        self.flashbots_client = FlashbotsClient(
            private_key=self.private_key,
            network=FlashbotsNetwork.MAINNET,
            relay_url=self.relay_url
        )
        await self.flashbots_client.initialize()
    
    async def submit_transaction(
        self,
        transaction: RouterTransaction,
        route: ExecutionRoute
    ) -> Dict[str, Any]:
        """Submit via Flashbots."""
        logger.info(f"Submitting transaction {transaction.segment_id} via Flashbots")
        
        if not self.flashbots_client:
            await self.initialize()
        
        # For single transaction, create a simple bundle
        from yield_arbitrage.mev_protection.flashbots_client import FlashbotsBundle
        
        # Convert transaction to Flashbots format
        fb_tx = self.flashbots_client._convert_router_transaction_to_flashbots(
            transaction, 
            route.priority_fee_wei / 1e9,  # Convert to gwei
            True  # Single transaction pays coinbase
        )
        
        target_block = await self.flashbots_client._get_next_block_number()
        
        bundle = FlashbotsBundle(
            transactions=[fb_tx],
            target_block=target_block,
            bundle_id=f"single_{transaction.segment_id}_{target_block}"
        )
        
        # Submit bundle
        response = await self.flashbots_client.submit_bundle(bundle)
        
        return {
            "success": response.success,
            "bundle_hash": response.bundle_hash,
            "method": "flashbots",
            "submitted_at": response.submitted_at,
            "error": response.error
        }
    
    async def check_inclusion(self, bundle_hash: str, route: ExecutionRoute) -> Dict[str, Any]:
        """Check Flashbots bundle status."""
        if not self.flashbots_client:
            return {"included": False, "error": "Client not initialized"}
        
        # Get target block from route params
        target_block = route.flashbots_params.get("target_block", "next")
        if target_block == "next":
            target_block = await self.flashbots_client._get_latest_block_number()
        
        return await self.flashbots_client.check_bundle_inclusion(bundle_hash, target_block)
    
    def supports_method(self, method: ExecutionMethod) -> bool:
        """Flashbots provider supports Flashbots methods."""
        return method in [ExecutionMethod.FLASHBOTS, ExecutionMethod.FLASHBOTS_BUNDLE]


class PrivateNodeProvider(ExecutionProvider):
    """Private node execution provider."""
    
    def __init__(self, node_url: str, auth_token: Optional[str] = None):
        self.node_url = node_url
        self.auth_token = auth_token
    
    async def submit_transaction(
        self,
        transaction: RouterTransaction,
        route: ExecutionRoute
    ) -> Dict[str, Any]:
        """Submit to private node."""
        logger.info(f"Submitting transaction {transaction.segment_id} to private node")
        
        return {
            "success": True,
            "tx_hash": f"0x{'2' * 64}",
            "method": "private_node",
            "submitted_at": 1234567890
        }
    
    async def check_inclusion(self, tx_hash: str, route: ExecutionRoute) -> Dict[str, Any]:
        """Check transaction status on private node."""
        return {
            "included": True,
            "block_number": 12345678,
            "confirmations": 6
        }
    
    def supports_method(self, method: ExecutionMethod) -> bool:
        """Private node supports private methods."""
        return method in [ExecutionMethod.PRIVATE_NODE, ExecutionMethod.PRIVATE_RELAY]


class MEVAwareExecutionRouter:
    """
    Intelligent execution router that selects optimal execution method
    based on MEV risk assessment and chain capabilities.
    """
    
    def __init__(self):
        """Initialize execution router with chain configurations."""
        self.chain_configs = self._initialize_chain_configs()
        self.providers: Dict[str, ExecutionProvider] = {}
        self._initialize_providers()
    
    def _initialize_chain_configs(self) -> Dict[int, ChainExecutionConfig]:
        """Initialize configurations for supported chains."""
        configs = {
            # Ethereum Mainnet
            1: ChainExecutionConfig(
                chain_id=1,
                chain_name="ethereum",
                supports_flashbots=True,
                public_rpc="https://eth-mainnet.g.alchemy.com/v2/",
                flashbots_relay="https://relay.flashbots.net",
                has_private_mempool=True,
                block_time=12
            ),
            
            # Binance Smart Chain
            56: ChainExecutionConfig(
                chain_id=56,
                chain_name="bsc",
                supports_flashbots=False,
                private_relay_url="https://bsc-private-relay.example.com",
                public_rpc="https://bsc-dataseed.binance.org/",
                has_private_mempool=True,
                block_time=3
            ),
            
            # Polygon
            137: ChainExecutionConfig(
                chain_id=137,
                chain_name="polygon",
                supports_flashbots=False,
                private_node_url="https://polygon-private.example.com",
                public_rpc="https://polygon-rpc.com/",
                block_time=2
            ),
            
            # Arbitrum
            42161: ChainExecutionConfig(
                chain_id=42161,
                chain_name="arbitrum",
                supports_flashbots=False,
                public_rpc="https://arb1.arbitrum.io/rpc",
                has_sequencer=True,
                mev_protection_native=True,  # Sequencer provides some protection
                block_time=1
            ),
            
            # Optimism
            10: ChainExecutionConfig(
                chain_id=10,
                chain_name="optimism",
                supports_flashbots=False,
                public_rpc="https://mainnet.optimism.io",
                has_sequencer=True,
                mev_protection_native=True,
                block_time=2
            ),
            
            # Base
            8453: ChainExecutionConfig(
                chain_id=8453,
                chain_name="base",
                supports_flashbots=False,
                public_rpc="https://mainnet.base.org",
                has_sequencer=True,
                mev_protection_native=True,
                block_time=2
            )
        }
        
        return configs
    
    def _initialize_providers(self):
        """Initialize execution providers."""
        # Public provider (available for all chains)
        self.providers["public"] = PublicExecutionProvider("https://default-rpc.com")
        
        # Flashbots provider (Ethereum only) - requires private key
        # In production, this would be injected via config
        flashbots_private_key = "0x" + "0" * 64  # Placeholder private key
        self.providers["flashbots"] = FlashbotsProvider(
            private_key=flashbots_private_key,
            relay_url="https://relay.flashbots.net"
        )
        
        # Private node providers (chain-specific)
        self.providers["private_node"] = PrivateNodeProvider(
            "https://private-node.example.com",
            "auth_token"
        )
    
    def select_execution_route(
        self,
        chain_id: int,
        mev_analysis: PathMEVAnalysis,
        execution_plan: BatchExecutionPlan
    ) -> ExecutionRoute:
        """
        Select optimal execution route based on chain and MEV analysis.
        
        Args:
            chain_id: Target blockchain ID
            mev_analysis: MEV risk analysis for the path
            execution_plan: Execution plan to be routed
            
        Returns:
            Selected execution route with all parameters
        """
        logger.info(
            f"Selecting execution route for chain {chain_id} "
            f"with {mev_analysis.overall_risk_level.value} MEV risk"
        )
        
        config = self.chain_configs.get(chain_id)
        if not config:
            logger.warning(f"Unknown chain {chain_id}, using public execution")
            return self._create_public_route()
        
        # Determine method based on risk and chain capabilities
        method = self._determine_execution_method(config, mev_analysis)
        
        # Create route with appropriate parameters
        route = self._create_execution_route(method, config, mev_analysis)
        
        logger.info(f"Selected execution method: {method.value} via {route.endpoint}")
        
        return route
    
    def _determine_execution_method(
        self,
        config: ChainExecutionConfig,
        mev_analysis: PathMEVAnalysis
    ) -> ExecutionMethod:
        """Determine best execution method for chain and risk level."""
        risk_level = mev_analysis.overall_risk_level
        
        # Low risk - use public mempool
        if risk_level <= MEVRiskLevel.LOW:
            if config.mev_protection_native:
                return ExecutionMethod.PUBLIC  # Chain has native protection
            else:
                return ExecutionMethod.PUBLIC_PROTECTED
        
        # Medium risk - use private infrastructure if available
        if risk_level == MEVRiskLevel.MEDIUM:
            if config.private_node_url:
                return ExecutionMethod.PRIVATE_NODE
            elif config.private_relay_url:
                return ExecutionMethod.PRIVATE_RELAY
            elif config.supports_flashbots:
                return ExecutionMethod.FLASHBOTS
            else:
                return ExecutionMethod.PUBLIC_PROTECTED
        
        # High/Critical risk - use strongest protection available
        if risk_level >= MEVRiskLevel.HIGH:
            if config.supports_flashbots:
                return ExecutionMethod.FLASHBOTS_BUNDLE
            elif config.private_relay_url:
                return ExecutionMethod.PRIVATE_RELAY
            elif config.private_node_url:
                return ExecutionMethod.PRIVATE_NODE
            elif config.has_sequencer:
                # L2 sequencer provides some protection
                return ExecutionMethod.PUBLIC_PROTECTED
            else:
                # Last resort - use public with max protection
                logger.warning(
                    f"High MEV risk on chain {config.chain_name} "
                    "without private execution options"
                )
                return ExecutionMethod.PUBLIC_PROTECTED
        
        return ExecutionMethod.PUBLIC
    
    def _create_execution_route(
        self,
        method: ExecutionMethod,
        config: ChainExecutionConfig,
        mev_analysis: PathMEVAnalysis
    ) -> ExecutionRoute:
        """Create execution route with all necessary parameters."""
        route = ExecutionRoute(
            method=method,
            endpoint=self._get_endpoint(method, config),
            auth_required=method != ExecutionMethod.PUBLIC
        )
        
        # Calculate priority fee based on MEV risk
        base_priority = config.min_priority_fee
        risk_multiplier = 1.0 + (mev_analysis.compounded_risk * 2.0)  # Up to 3x
        
        route.priority_fee_wei = int(base_priority * risk_multiplier)
        route.max_fee_per_gas_wei = route.priority_fee_wei * 10  # Conservative max
        
        # Set method-specific parameters
        if method in [ExecutionMethod.FLASHBOTS, ExecutionMethod.FLASHBOTS_BUNDLE]:
            route.flashbots_params = {
                "target_block": "next",
                "min_bid_wei": route.priority_fee_wei,
                "max_bundle_size": 3,
                "simulation_required": True
            }
            route.bundle_id = f"bundle_{mev_analysis.path_id}"
        
        elif method in [ExecutionMethod.PRIVATE_NODE, ExecutionMethod.PRIVATE_RELAY]:
            route.node_params = {
                "skip_validation": False,
                "priority_inclusion": True
            }
        
        # Set fallback
        if method != ExecutionMethod.PUBLIC:
            route.fallback_method = ExecutionMethod.PUBLIC_PROTECTED
        
        # Adjust confirmation time based on method
        if method == ExecutionMethod.FLASHBOTS_BUNDLE:
            route.expected_confirmation_time = config.block_time * 2  # May take 2 blocks
        else:
            route.expected_confirmation_time = config.block_time
        
        return route
    
    def _get_endpoint(self, method: ExecutionMethod, config: ChainExecutionConfig) -> str:
        """Get endpoint URL for execution method."""
        if method == ExecutionMethod.PUBLIC:
            return config.public_rpc
        elif method == ExecutionMethod.PUBLIC_PROTECTED:
            return config.public_rpc  # Same endpoint, different params
        elif method in [ExecutionMethod.FLASHBOTS, ExecutionMethod.FLASHBOTS_BUNDLE]:
            return config.flashbots_relay
        elif method == ExecutionMethod.PRIVATE_NODE:
            return config.private_node_url or config.public_rpc
        elif method == ExecutionMethod.PRIVATE_RELAY:
            return config.private_relay_url or config.public_rpc
        else:
            return config.public_rpc
    
    def _create_public_route(self) -> ExecutionRoute:
        """Create default public execution route."""
        return ExecutionRoute(
            method=ExecutionMethod.PUBLIC,
            endpoint="https://eth-mainnet.g.alchemy.com/v2/",
            auth_required=False,
            priority_fee_wei=1_000_000_000,  # 1 gwei
            max_fee_per_gas_wei=50_000_000_000  # 50 gwei
        )
    
    async def execute_with_protection(
        self,
        execution_plan: BatchExecutionPlan,
        mev_analysis: PathMEVAnalysis,
        chain_id: int
    ) -> Dict[str, Any]:
        """
        Execute arbitrage plan with MEV protection.
        
        Args:
            execution_plan: Complete execution plan
            mev_analysis: MEV risk analysis
            chain_id: Target chain ID
            
        Returns:
            Execution results
        """
        # Select execution route
        route = self.select_execution_route(chain_id, mev_analysis, execution_plan)
        
        # Get appropriate provider
        provider = self._get_provider_for_method(route.method)
        if not provider:
            raise ValueError(f"No provider available for method {route.method}")
        
        # Execute transactions
        results = []
        for transaction in execution_plan.segments:
            result = await provider.submit_transaction(transaction, route)
            results.append(result)
        
        return {
            "execution_method": route.method.value,
            "endpoint": route.endpoint,
            "transactions": results,
            "estimated_cost_wei": route.priority_fee_wei * sum(
                tx.gas_limit for tx in execution_plan.segments
            )
        }
    
    def _get_provider_for_method(self, method: ExecutionMethod) -> Optional[ExecutionProvider]:
        """Get provider that supports the given method."""
        for provider in self.providers.values():
            if provider.supports_method(method):
                return provider
        return None
    
    def get_chain_capabilities(self, chain_id: int) -> Dict[str, Any]:
        """Get MEV protection capabilities for a chain."""
        config = self.chain_configs.get(chain_id)
        if not config:
            return {"supported": False}
        
        return {
            "supported": True,
            "chain_name": config.chain_name,
            "methods": {
                "flashbots": config.supports_flashbots,
                "private_node": config.private_node_url is not None,
                "private_relay": config.private_relay_url is not None,
                "native_protection": config.mev_protection_native
            },
            "block_time": config.block_time,
            "has_sequencer": config.has_sequencer
        }


# Convenience functions

def create_execution_router() -> MEVAwareExecutionRouter:
    """Create a configured execution router."""
    return MEVAwareExecutionRouter()


def get_chain_execution_method(
    chain_id: int,
    risk_level: MEVRiskLevel
) -> ExecutionMethod:
    """Get recommended execution method for chain and risk level."""
    router = MEVAwareExecutionRouter()
    config = router.chain_configs.get(chain_id)
    
    if not config:
        return ExecutionMethod.PUBLIC
    
    # Create mock MEV analysis with the risk level
    mock_analysis = PathMEVAnalysis(
        path_id="mock",
        total_edges=1,
        overall_risk_level=risk_level
    )
    
    return router._determine_execution_method(config, mock_analysis)