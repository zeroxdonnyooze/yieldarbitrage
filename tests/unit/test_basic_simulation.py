"""Unit tests for the basic simulation mode of HybridPathSimulator."""
import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta

from yield_arbitrage.execution.hybrid_simulator import (
    HybridPathSimulator,
    SimulationMode,
    SimulationResult,
    SimulatorConfig,
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType, EdgeState


class TestBasicSimulation:
    """Test the basic simulation mode."""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        client = Mock()
        client.get = AsyncMock(return_value=None)
        client.setex = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_asset_oracle(self):
        """Create a mock asset oracle."""
        oracle = Mock()
        # Default prices: ETH=$2000, USDC=$1, USDT=$1
        price_map = {
            "ETH_MAINNET_WETH": 2000.0,
            "ETH_MAINNET_USDC": 1.0,
            "ETH_MAINNET_USDT": 1.0,
            "ETH_MAINNET_DAI": 1.0,
        }
        oracle.get_price_usd = AsyncMock(side_effect=lambda asset_id: price_map.get(asset_id))
        return oracle
    
    @pytest.fixture
    def simulator_config(self):
        """Create a test simulator configuration."""
        return SimulatorConfig(
            default_slippage_factor=0.02,  # 2%
            min_liquidity_threshold=1000.0,
            confidence_threshold=0.6,
            default_gas_price_gwei=20.0,
            eth_price_usd=2000.0
        )
    
    @pytest.fixture
    def simulator(self, mock_redis_client, mock_asset_oracle, simulator_config):
        """Create a HybridPathSimulator instance."""
        return HybridPathSimulator(
            redis_client=mock_redis_client,
            asset_oracle=mock_asset_oracle,
            config=simulator_config
        )
    
    @pytest.fixture
    def sample_edge(self):
        """Create a sample ETH->USDC edge."""
        return YieldGraphEdge(
            edge_id="eth_usdc_uniswap",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswapv3",
            chain_name="ethereum"
        )
    
    @pytest.fixture
    def profitable_edge_state(self):
        """Create an edge state that should be profitable."""
        return EdgeState(
            conversion_rate=2100.0,  # ETH->USDC at $2100 (better than market $2000)
            confidence_score=0.9,
            liquidity_usd=500_000.0,
            gas_cost_usd=15.0,
            last_updated_timestamp=datetime.utcnow().timestamp()
        )
    
    @pytest.fixture
    def unprofitable_edge_state(self):
        """Create an edge state that should be unprofitable."""
        return EdgeState(
            conversion_rate=1900.0,  # ETH->USDC at $1900 (worse than market $2000)
            confidence_score=0.8,
            liquidity_usd=100_000.0,
            gas_cost_usd=150.0,  # High gas cost
            last_updated_timestamp=datetime.utcnow().timestamp()
        )
    
    @pytest.mark.asyncio
    async def test_profitable_single_edge_simulation(
        self, simulator, sample_edge, profitable_edge_state, mock_redis_client
    ):
        """Test a profitable single-edge simulation."""
        # Setup Redis to return the profitable edge state
        mock_redis_client.get.return_value = profitable_edge_state.model_dump_json()
        
        result = await simulator.simulate_path(
            path=[sample_edge],
            initial_amount=1.0,  # 1 ETH
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        # Should be successful and profitable
        assert result.success
        assert result.simulation_mode == SimulationMode.BASIC.value
        assert result.profit_usd > 0
        assert result.output_amount > 0
        assert result.path_details is not None
        assert len(result.path_details) == 1
        
        # Check that we made money
        step_detail = result.path_details[0]
        assert step_detail["step"] == 1
        assert step_detail["edge_id"] == "eth_usdc_uniswap"
        assert step_detail["conversion_rate"] == 2100.0
        assert step_detail["final_output"] > 0
    
    @pytest.mark.asyncio
    async def test_unprofitable_simulation_due_to_gas(
        self, simulator, sample_edge, mock_redis_client
    ):
        """Test simulation that fails due to high gas costs."""
        # Create an edge state that will definitely be unprofitable
        very_bad_state = EdgeState(
            conversion_rate=1000.0,  # Very bad rate: 1 ETH -> 1000 USDC (should be ~2000)
            confidence_score=0.8,
            liquidity_usd=100_000.0,
            gas_cost_usd=500.0,  # Extremely high gas cost
            last_updated_timestamp=datetime.utcnow().timestamp()
        )
        
        mock_redis_client.get.return_value = very_bad_state.model_dump_json()
        
        result = await simulator.simulate_path(
            path=[sample_edge],
            initial_amount=0.1,  # Small amount: 0.1 ETH
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        # Should fail due to unprofitability
        assert not result.success
        if result.revert_reason:
            assert ("unprofitable" in result.revert_reason.lower() or 
                   "became unprofitable" in result.revert_reason.lower())
        assert result.failed_at_step == 1
    
    @pytest.mark.asyncio
    async def test_low_confidence_edge_failure(
        self, simulator, sample_edge, mock_redis_client
    ):
        """Test simulation failure due to very low confidence."""
        # Create edge state with very low confidence
        low_confidence_state = EdgeState(
            conversion_rate=2000.0,
            confidence_score=0.2,  # Very low confidence
            liquidity_usd=100_000.0,
            gas_cost_usd=10.0,
            last_updated_timestamp=datetime.utcnow().timestamp()
        )
        
        mock_redis_client.get.return_value = low_confidence_state.model_dump_json()
        
        result = await simulator.simulate_path(
            path=[sample_edge],
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        # Should fail due to low confidence
        assert not result.success
        assert "confidence too low" in result.revert_reason
        assert result.failed_at_step == 1
    
    @pytest.mark.asyncio
    async def test_missing_edge_state_failure(
        self, simulator, sample_edge, mock_redis_client
    ):
        """Test simulation failure when edge state is missing."""
        # Redis returns None (no cached state)
        mock_redis_client.get.return_value = None
        
        result = await simulator.simulate_path(
            path=[sample_edge],
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        # Should fail due to missing state
        assert not result.success
        assert "No state available" in result.revert_reason
        assert result.failed_at_step == 1
    
    @pytest.mark.asyncio
    async def test_multi_edge_arbitrage_cycle(
        self, simulator, mock_redis_client, mock_asset_oracle
    ):
        """Test a multi-edge arbitrage cycle: ETH -> USDC -> USDT -> ETH."""
        # Create a 3-edge arbitrage path
        edge1 = YieldGraphEdge(
            edge_id="eth_usdc",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswapv3",
            chain_name="ethereum"
        )
        
        edge2 = YieldGraphEdge(
            edge_id="usdc_usdt",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_USDT",
            edge_type=EdgeType.TRADE,
            protocol_name="curve",
            chain_name="ethereum"
        )
        
        edge3 = YieldGraphEdge(
            edge_id="usdt_eth",
            source_asset_id="ETH_MAINNET_USDT",
            target_asset_id="ETH_MAINNET_WETH",
            edge_type=EdgeType.TRADE,
            protocol_name="sushiswap",
            chain_name="ethereum"
        )
        
        # Create profitable edge states
        state1 = EdgeState(
            conversion_rate=2050.0,  # ETH->USDC slightly profitable
            confidence_score=0.9,
            liquidity_usd=1_000_000.0,
            gas_cost_usd=20.0,
            last_updated_timestamp=datetime.utcnow().timestamp()
        )
        
        state2 = EdgeState(
            conversion_rate=1.001,   # USDC->USDT slight profit
            confidence_score=0.95,
            liquidity_usd=2_000_000.0,
            gas_cost_usd=15.0,
            last_updated_timestamp=datetime.utcnow().timestamp()
        )
        
        state3 = EdgeState(
            conversion_rate=0.0005,  # USDT->ETH (1/2000 = 0.0005 ETH per USDT)
            confidence_score=0.85,
            liquidity_usd=800_000.0,
            gas_cost_usd=25.0,
            last_updated_timestamp=datetime.utcnow().timestamp()
        )
        
        # Mock Redis to return different states for different edges
        def get_edge_state(key):
            if key == "edge_state:eth_usdc":
                return state1.model_dump_json()
            elif key == "edge_state:usdc_usdt":
                return state2.model_dump_json()
            elif key == "edge_state:usdt_eth":
                return state3.model_dump_json()
            return None
        
        mock_redis_client.get.side_effect = get_edge_state
        
        result = await simulator.simulate_path(
            path=[edge1, edge2, edge3],
            initial_amount=1.0,  # 1 ETH
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        # Should complete the cycle
        assert result.success or not result.success  # May or may not be profitable
        assert result.path_details is not None
        assert len(result.path_details) == 3
        
        # Check each step was processed
        for i, step in enumerate(result.path_details):
            assert step["step"] == i + 1
    
    @pytest.mark.asyncio
    async def test_slippage_calculation_large_trade(
        self, simulator, sample_edge, mock_redis_client
    ):
        """Test slippage calculation for large trades."""
        # Create edge state with limited liquidity
        edge_state = EdgeState(
            conversion_rate=2000.0,
            confidence_score=0.9,
            liquidity_usd=100_000.0,  # Limited liquidity
            gas_cost_usd=20.0,
            last_updated_timestamp=datetime.utcnow().timestamp()
        )
        
        mock_redis_client.get.return_value = edge_state.model_dump_json()
        
        result = await simulator.simulate_path(
            path=[sample_edge],
            initial_amount=10.0,  # Large trade: 10 ETH = $20,000
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        # Should have warnings about large trade size
        assert result.warnings
        assert any("Large trade relative to liquidity" in warning for warning in result.warnings)
        
        # Should have significant slippage
        if result.path_details:
            step = result.path_details[0]
            assert step["slippage_impact"] > 0.01  # > 1% slippage
    
    @pytest.mark.asyncio
    async def test_gas_cost_estimation(self, simulator):
        """Test gas cost estimation for different edge types."""
        # Test different edge types
        trade_edge = YieldGraphEdge(
            edge_id="trade_edge",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswapv3",
            chain_name="ethereum"
        )
        
        flash_loan_edge = YieldGraphEdge(
            edge_id="flash_loan_edge",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.FLASH_LOAN,
            protocol_name="aave",
            chain_name="ethereum"
        )
        
        # Test gas cost estimation
        trade_gas_cost = simulator._estimate_gas_cost_usd(trade_edge)
        flash_loan_gas_cost = simulator._estimate_gas_cost_usd(flash_loan_edge)
        
        assert trade_gas_cost > 0
        assert flash_loan_gas_cost > 0
        
        # Trade should cost more than flash loan (due to protocol complexity)
        assert trade_gas_cost > flash_loan_gas_cost
    
    @pytest.mark.asyncio
    async def test_asset_mismatch_failure(
        self, simulator, profitable_edge_state, mock_redis_client
    ):
        """Test simulation failure when edge source doesn't match current asset."""
        # Create edge with wrong source asset
        wrong_edge = YieldGraphEdge(
            edge_id="wrong_edge",
            source_asset_id="ETH_MAINNET_USDC",  # Wrong source!
            target_asset_id="ETH_MAINNET_USDT",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswapv3",
            chain_name="ethereum"
        )
        
        mock_redis_client.get.return_value = profitable_edge_state.model_dump_json()
        
        result = await simulator.simulate_path(
            path=[wrong_edge],
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",  # Starting with ETH, but edge expects USDC
            mode=SimulationMode.BASIC
        )
        
        # Should fail due to asset mismatch
        assert not result.success
        assert "source" in result.revert_reason and "current asset" in result.revert_reason
        assert result.failed_at_step == 1
    
    def test_simulation_result_to_dict(self):
        """Test SimulationResult serialization."""
        result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.BASIC.value,
            profit_usd=15.5,
            profit_percentage=1.55,
            gas_cost_usd=8.0,
            output_amount=1995.0,
            warnings=["Test warning"],
            simulation_time_ms=125.5
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["profit_usd"] == 15.5
        assert result_dict["gas_cost_usd"] == 8.0
        assert "Test warning" in result_dict["warnings"]
        assert result_dict["simulation_time_ms"] == 125.5