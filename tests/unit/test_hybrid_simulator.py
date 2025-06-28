"""Unit tests for the hybrid path simulator."""
import pytest
from unittest.mock import Mock, AsyncMock

from yield_arbitrage.execution.hybrid_simulator import (
    HybridPathSimulator,
    SimulationMode,
    SimulationResult,
    SimulatorConfig,
    TenderlyConfig
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType, EdgeState


class TestHybridPathSimulator:
    """Test the HybridPathSimulator class."""
    
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
        oracle.get_price_usd = AsyncMock(return_value=2000.0)  # ETH price
        oracle.get_prices_batch = AsyncMock(return_value={})
        return oracle
    
    @pytest.fixture
    def simulator_config(self):
        """Create a test simulator configuration."""
        return SimulatorConfig(
            default_slippage_factor=0.03,
            tenderly_profit_threshold_usd=5.0,
            max_concurrent_simulations=5
        )
    
    @pytest.fixture
    def tenderly_config(self):
        """Create a test Tenderly configuration."""
        return TenderlyConfig(
            api_key="test_api_key",
            project_slug="test_project",
            username="test_user"
        )
    
    @pytest.fixture
    def simulator(self, mock_redis_client, mock_asset_oracle, simulator_config):
        """Create a HybridPathSimulator instance."""
        return HybridPathSimulator(
            redis_client=mock_redis_client,
            asset_oracle=mock_asset_oracle,
            config=simulator_config,
            tenderly_config=None  # No Tenderly for basic tests
        )
    
    @pytest.fixture
    def sample_edge(self):
        """Create a sample edge for testing."""
        return YieldGraphEdge(
            edge_id="eth_usdc_trade",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswapv3",
            chain_name="ethereum"
        )
    
    def test_simulator_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator.redis_client is not None
        assert simulator.asset_oracle is not None
        assert simulator.config is not None
        assert simulator.tenderly_config is None
        assert simulator.tenderly_session is None
    
    def test_simulator_with_tenderly_config(
        self, mock_redis_client, mock_asset_oracle, simulator_config, tenderly_config
    ):
        """Test simulator initialization with Tenderly config."""
        simulator = HybridPathSimulator(
            redis_client=mock_redis_client,
            asset_oracle=mock_asset_oracle,
            config=simulator_config,
            tenderly_config=tenderly_config
        )
        
        assert simulator.tenderly_config is not None
        assert simulator.tenderly_config.api_key == "test_api_key"
    
    @pytest.mark.asyncio
    async def test_simulate_path_empty_path(self, simulator):
        """Test simulation with empty path."""
        result = await simulator.simulate_path(
            path=[],
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH"
        )
        
        assert not result.success
        assert result.revert_reason == "Empty path provided"
    
    @pytest.mark.asyncio
    async def test_simulate_path_invalid_amount(self, simulator, sample_edge):
        """Test simulation with invalid initial amount."""
        result = await simulator.simulate_path(
            path=[sample_edge],
            initial_amount=-1.0,
            start_asset_id="ETH_MAINNET_WETH"
        )
        
        assert not result.success
        assert result.revert_reason == "Initial amount must be positive"
    
    @pytest.mark.asyncio
    async def test_simulate_path_basic_mode(self, simulator, sample_edge):
        """Test simulation in BASIC mode."""
        result = await simulator.simulate_path(
            path=[sample_edge],
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        # Should fail because no edge state is available in Redis
        assert not result.success
        assert result.simulation_mode == SimulationMode.BASIC.value
        assert "No state available" in result.revert_reason
    
    @pytest.mark.asyncio
    async def test_simulate_path_tenderly_not_configured(self, simulator, sample_edge):
        """Test Tenderly mode when not configured."""
        result = await simulator.simulate_path(
            path=[sample_edge],
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.TENDERLY
        )
        
        assert not result.success
        assert result.revert_reason == "Tenderly not configured"
    
    @pytest.mark.asyncio
    async def test_estimate_slippage_impact(self, simulator):
        """Test slippage estimation."""
        # Test with good liquidity
        slippage = await simulator._estimate_slippage_impact(
            trade_amount_usd=1000.0,
            liquidity_usd=1_000_000.0
        )
        assert 0 < slippage < 0.1  # Should be small
        
        # Test with low liquidity
        slippage = await simulator._estimate_slippage_impact(
            trade_amount_usd=10_000.0,
            liquidity_usd=20_000.0
        )
        assert slippage > 0.05  # Should be significant
        
        # Test with no liquidity data
        slippage = await simulator._estimate_slippage_impact(
            trade_amount_usd=1000.0,
            liquidity_usd=None
        )
        assert slippage == simulator.config.default_slippage_factor
    
    @pytest.mark.asyncio
    async def test_get_edge_state(self, simulator, mock_redis_client):
        """Test edge state retrieval from Redis."""
        # Test with no cached state
        state = await simulator._get_edge_state("test_edge_id")
        assert state is None
        mock_redis_client.get.assert_called_with("edge_state:test_edge_id")
        
        # Test with cached state
        mock_redis_client.get.return_value = EdgeState(
            conversion_rate=1500.0,
            confidence_score=0.9
        ).model_dump_json()
        
        state = await simulator._get_edge_state("test_edge_id")
        assert state is not None
        assert state.conversion_rate == 1500.0
        assert state.confidence_score == 0.9
    
    def test_get_stats(self, simulator):
        """Test statistics retrieval."""
        stats = simulator.get_stats()
        
        assert "basic_simulations" in stats
        assert "tenderly_simulations" in stats
        assert "hybrid_simulations" in stats
        assert "simulation_errors" in stats
        assert "total_profit_found_usd" in stats
        
        # All should be zero initially
        assert stats["basic_simulations"] == 0
        assert stats["total_profit_found_usd"] == 0.0
    
    @pytest.mark.asyncio
    async def test_validate_edge(self, simulator, sample_edge):
        """Test single edge validation."""
        result = await simulator.validate_edge(
            edge=sample_edge,
            input_amount=1.0,
            mode=SimulationMode.BASIC
        )
        
        # Should create a single-edge path and simulate
        assert isinstance(result, SimulationResult)
        assert result.simulation_mode == SimulationMode.BASIC.value
    
    @pytest.mark.asyncio
    async def test_simulation_result_to_dict(self):
        """Test SimulationResult to_dict method."""
        result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.BASIC.value,
            profit_usd=10.5,
            profit_percentage=1.05,
            gas_cost_usd=5.0,
            warnings=["Test warning"]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["profit_usd"] == 10.5
        assert result_dict["gas_cost_usd"] == 5.0
        assert "Test warning" in result_dict["warnings"]
    
    @pytest.mark.asyncio
    async def test_simulator_resource_cleanup(self, simulator):
        """Test resource cleanup."""
        # Initialize resources
        await simulator.initialize()
        
        # Clean up
        await simulator.close()
        
        # Should be cleaned up
        assert simulator.tenderly_session is None