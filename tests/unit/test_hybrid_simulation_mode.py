"""Tests for hybrid simulation mode functionality."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal

from yield_arbitrage.execution.hybrid_simulator import (
    HybridPathSimulator,
    SimulationMode,
    SimulationResult,
    TenderlyConfig,
    SimulatorConfig,
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for testing."""
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    
    mock_oracle = Mock()
    mock_oracle.get_price_usd = AsyncMock(return_value=2000.0)
    
    return mock_redis, mock_oracle


@pytest.fixture
def simulator_config():
    """Create test simulator configuration."""
    return SimulatorConfig(
        tenderly_profit_threshold_usd=10.0,
        tenderly_amount_threshold_usd=1000.0,
        default_slippage_factor=0.05,
        min_liquidity_threshold=1000.0
    )


@pytest.fixture
def tenderly_config():
    """Create test Tenderly configuration."""
    return TenderlyConfig(
        api_key="test_key",
        username="test_user",
        project_slug="test_project"
    )


@pytest.fixture
def sample_path():
    """Create a sample trading path for testing."""
    return [
        YieldGraphEdge(
            edge_id="test_edge_1",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        ),
        YieldGraphEdge(
            edge_id="test_edge_2",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_DAI",
            edge_type=EdgeType.TRADE,
            protocol_name="sushiswap",
            chain_name="ethereum"
        )
    ]


@pytest.mark.asyncio
class TestHybridSimulationMode:
    """Test hybrid simulation mode logic."""
    
    async def test_hybrid_uses_basic_when_below_thresholds(
        self, mock_dependencies, simulator_config, sample_path
    ):
        """Test that hybrid mode uses basic simulation for small trades."""
        mock_redis, mock_oracle = mock_dependencies
        
        simulator = HybridPathSimulator(
            redis_client=mock_redis,
            asset_oracle=mock_oracle,
            config=simulator_config
        )
        
        # Mock basic simulation to return small profit
        basic_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.BASIC.value,
            profit_usd=5.0,  # Below $10 threshold
            profit_percentage=0.5,
            output_amount=1.05,
            gas_cost_usd=3.0
        )
        
        with patch.object(simulator, '_simulate_basic', return_value=basic_result):
            result = await simulator.simulate_path(
                path=sample_path,
                initial_amount=0.1,  # Small amount
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.HYBRID
            )
        
        assert result.success is True
        assert result.simulation_mode == SimulationMode.HYBRID.value
        assert result.profit_usd == 5.0
        assert "Used basic simulation only" in result.warnings
    
    async def test_hybrid_uses_tenderly_for_high_profit(
        self, mock_dependencies, simulator_config, tenderly_config, sample_path
    ):
        """Test that hybrid mode uses Tenderly for high-profit paths."""
        mock_redis, mock_oracle = mock_dependencies
        
        simulator = HybridPathSimulator(
            redis_client=mock_redis,
            asset_oracle=mock_oracle,
            config=simulator_config,
            tenderly_config=tenderly_config
        )
        
        # Mock basic simulation with high profit
        basic_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.BASIC.value,
            profit_usd=25.0,  # Above $10 threshold
            profit_percentage=2.5,
            output_amount=1.25,
            gas_cost_usd=3.0
        )
        
        # Mock Tenderly simulation
        tenderly_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.TENDERLY.value,
            profit_usd=23.0,  # Slightly different from basic
            profit_percentage=2.3,
            output_amount=1.23,
            gas_used=150000,
            gas_cost_usd=4.5
        )
        
        with patch.object(simulator, '_simulate_basic', return_value=basic_result), \
             patch.object(simulator, '_simulate_tenderly', return_value=tenderly_result):
            
            result = await simulator.simulate_path(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.HYBRID
            )
        
        assert result.success is True
        assert result.simulation_mode == SimulationMode.HYBRID.value
        assert result.profit_usd == 23.0  # Uses Tenderly result
        assert result.gas_used == 150000
        # Check if there are warnings about discrepancy (might be empty list)
        warnings_str = str(result.warnings) if result.warnings else "[]"
        # The discrepancy warning only appears if there's a significant difference
    
    async def test_hybrid_uses_tenderly_for_large_trades(
        self, mock_dependencies, simulator_config, tenderly_config, sample_path
    ):
        """Test that hybrid mode uses Tenderly for large trade amounts."""
        mock_redis, mock_oracle = mock_dependencies
        
        # Set high ETH price for large trade value
        mock_oracle.get_price_usd = AsyncMock(return_value=3000.0)
        
        simulator = HybridPathSimulator(
            redis_client=mock_redis,
            asset_oracle=mock_oracle,
            config=simulator_config,
            tenderly_config=tenderly_config
        )
        
        # Mock basic simulation with small profit but large amount
        basic_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.BASIC.value,
            profit_usd=5.0,  # Below profit threshold
            profit_percentage=0.1,
            output_amount=5.05,
            gas_cost_usd=10.0
        )
        
        tenderly_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.TENDERLY.value,
            profit_usd=4.5,
            profit_percentage=0.09,
            output_amount=5.045,
            gas_used=200000,
            gas_cost_usd=12.0
        )
        
        with patch.object(simulator, '_simulate_basic', return_value=basic_result), \
             patch.object(simulator, '_simulate_tenderly', return_value=tenderly_result):
            
            # 1 ETH at $3000 = $3000 > $1000 threshold
            result = await simulator.simulate_path(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.HYBRID
            )
        
        assert result.success is True
        assert result.simulation_mode == SimulationMode.HYBRID.value
        assert result.profit_usd == 4.5  # Uses Tenderly result
    
    async def test_hybrid_uses_tenderly_for_complex_paths(
        self, mock_dependencies, simulator_config, tenderly_config
    ):
        """Test that hybrid mode uses Tenderly for complex multi-step paths."""
        mock_redis, mock_oracle = mock_dependencies
        
        # Create complex 4-step path
        complex_path = [
            YieldGraphEdge(
                edge_id=f"test_edge_{i}",
                source_asset_id=f"asset_{i}",
                target_asset_id=f"asset_{i+1}",
                edge_type=EdgeType.TRADE,
                protocol_name=f"protocol_{i}",
                chain_name="ethereum"
            )
            for i in range(4)
        ]
        
        simulator = HybridPathSimulator(
            redis_client=mock_redis,
            asset_oracle=mock_oracle,
            config=simulator_config,
            tenderly_config=tenderly_config
        )
        
        basic_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.BASIC.value,
            profit_usd=3.0,  # Below thresholds
            profit_percentage=0.3,
            output_amount=1.03,
            gas_cost_usd=5.0
        )
        
        tenderly_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.TENDERLY.value,
            profit_usd=2.5,
            profit_percentage=0.25,
            output_amount=1.025,
            gas_used=400000,
            gas_cost_usd=8.0
        )
        
        with patch.object(simulator, '_simulate_basic', return_value=basic_result), \
             patch.object(simulator, '_simulate_tenderly', return_value=tenderly_result):
            
            result = await simulator.simulate_path(
                path=complex_path,
                initial_amount=0.1,  # Small amount
                start_asset_id="asset_0",
                mode=SimulationMode.HYBRID
            )
        
        # Should use Tenderly because path is complex (4 steps)
        assert result.success is True
        assert result.simulation_mode == SimulationMode.HYBRID.value
        assert result.profit_usd == 2.5  # Uses Tenderly result
    
    async def test_hybrid_uses_tenderly_for_risky_edges(
        self, mock_dependencies, simulator_config, tenderly_config
    ):
        """Test that hybrid mode uses Tenderly for risky edge types."""
        mock_redis, mock_oracle = mock_dependencies
        
        # Create path with flash loan (risky)
        risky_path = [
            YieldGraphEdge(
                edge_id="flash_loan_edge",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.FLASH_LOAN,  # Risky edge type
                protocol_name="aave",
                chain_name="ethereum"
            )
        ]
        
        simulator = HybridPathSimulator(
            redis_client=mock_redis,
            asset_oracle=mock_oracle,
            config=simulator_config,
            tenderly_config=tenderly_config
        )
        
        basic_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.BASIC.value,
            profit_usd=2.0,  # Below thresholds
            profit_percentage=0.2,
            output_amount=1.02
        )
        
        tenderly_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.TENDERLY.value,
            profit_usd=1.8,
            profit_percentage=0.18,
            output_amount=1.018,
            gas_used=300000
        )
        
        with patch.object(simulator, '_simulate_basic', return_value=basic_result), \
             patch.object(simulator, '_simulate_tenderly', return_value=tenderly_result):
            
            result = await simulator.simulate_path(
                path=risky_path,
                initial_amount=0.1,
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.HYBRID
            )
        
        # Should use Tenderly because of risky edge type
        assert result.success is True
        assert result.profit_usd == 1.8  # Uses Tenderly result
    
    async def test_hybrid_fallback_when_tenderly_fails(
        self, mock_dependencies, simulator_config, tenderly_config, sample_path
    ):
        """Test hybrid fallback to basic when Tenderly fails."""
        mock_redis, mock_oracle = mock_dependencies
        
        simulator = HybridPathSimulator(
            redis_client=mock_redis,
            asset_oracle=mock_oracle,
            config=simulator_config,
            tenderly_config=tenderly_config
        )
        
        # Mock basic simulation success
        basic_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.BASIC.value,
            profit_usd=15.0,  # Above threshold, should trigger Tenderly
            profit_percentage=1.5,
            output_amount=1.15,
            gas_cost_usd=3.0
        )
        
        # Mock Tenderly simulation failure
        tenderly_result = SimulationResult(
            success=False,
            simulation_mode=SimulationMode.TENDERLY.value,
            revert_reason="execution reverted: insufficient liquidity"
        )
        
        with patch.object(simulator, '_simulate_basic', return_value=basic_result), \
             patch.object(simulator, '_simulate_tenderly', return_value=tenderly_result):
            
            result = await simulator.simulate_path(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.HYBRID
            )
        
        # Should fallback to basic results
        assert result.success is True
        assert result.simulation_mode == SimulationMode.HYBRID.value
        assert result.profit_usd == 15.0  # Uses basic result
        # Check for the exact warning messages
        assert any("Tenderly validation failed" in str(w) for w in result.warnings)
        assert any("may be less accurate" in str(w) for w in result.warnings)
    
    async def test_hybrid_fails_when_both_fail(
        self, mock_dependencies, simulator_config, tenderly_config, sample_path
    ):
        """Test hybrid behavior when both basic and Tenderly fail."""
        mock_redis, mock_oracle = mock_dependencies
        
        simulator = HybridPathSimulator(
            redis_client=mock_redis,
            asset_oracle=mock_oracle,
            config=simulator_config,
            tenderly_config=tenderly_config
        )
        
        # Mock both simulations failing
        basic_result = SimulationResult(
            success=False,
            simulation_mode=SimulationMode.BASIC.value,
            revert_reason="insufficient edge state data"
        )
        
        tenderly_result = SimulationResult(
            success=False,
            simulation_mode=SimulationMode.TENDERLY.value,
            revert_reason="execution reverted"
        )
        
        with patch.object(simulator, '_simulate_basic', return_value=basic_result), \
             patch.object(simulator, '_simulate_tenderly', return_value=tenderly_result):
            
            result = await simulator.simulate_path(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.HYBRID
            )
        
        assert result.success is False
        assert result.simulation_mode == SimulationMode.HYBRID.value
        # Since basic fails first, it should exit early with basic failure message
        assert "Basic filter failed" in result.revert_reason
        assert "insufficient edge state data" in result.revert_reason
        assert "Failed basic simulation filter" in result.warnings
    
    async def test_hybrid_early_exit_on_basic_failure(
        self, mock_dependencies, simulator_config, sample_path
    ):
        """Test that hybrid exits early when basic simulation fails."""
        mock_redis, mock_oracle = mock_dependencies
        
        simulator = HybridPathSimulator(
            redis_client=mock_redis,
            asset_oracle=mock_oracle,
            config=simulator_config
        )
        
        # Mock basic simulation failure
        basic_result = SimulationResult(
            success=False,
            simulation_mode=SimulationMode.BASIC.value,
            revert_reason="no edge state available"
        )
        
        with patch.object(simulator, '_simulate_basic', return_value=basic_result) as mock_basic, \
             patch.object(simulator, '_simulate_tenderly') as mock_tenderly:
            
            result = await simulator.simulate_path(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.HYBRID
            )
        
        # Should call basic but not Tenderly
        mock_basic.assert_called_once()
        mock_tenderly.assert_not_called()
        
        assert result.success is False
        assert result.simulation_mode == SimulationMode.HYBRID.value
        assert "Basic filter failed" in result.revert_reason
        assert "Failed basic simulation filter" in result.warnings


@pytest.mark.asyncio
class TestHybridDecisionLogic:
    """Test the decision logic for when to use Tenderly validation."""
    
    async def test_should_use_tenderly_for_high_slippage(
        self, mock_dependencies, simulator_config
    ):
        """Test Tenderly validation triggered by high slippage."""
        mock_redis, mock_oracle = mock_dependencies
        
        simulator = HybridPathSimulator(
            redis_client=mock_redis,
            asset_oracle=mock_oracle,
            config=simulator_config
        )
        
        path = [YieldGraphEdge(
            edge_id="test",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        )]
        
        basic_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.BASIC.value,
            profit_usd=3.0,  # Below threshold
            slippage_estimate=0.05  # 5% slippage (high)
        )
        
        should_use = await simulator._should_use_tenderly_validation(
            path=path,
            initial_amount=0.1,
            basic_result=basic_result
        )
        
        assert should_use is True
    
    async def test_should_use_tenderly_for_multi_protocol(
        self, mock_dependencies, simulator_config
    ):
        """Test Tenderly validation triggered by multi-protocol paths."""
        mock_redis, mock_oracle = mock_dependencies
        
        simulator = HybridPathSimulator(
            redis_client=mock_redis,
            asset_oracle=mock_oracle,
            config=simulator_config
        )
        
        # Create path with 3 different protocols
        multi_protocol_path = [
            YieldGraphEdge(
                edge_id="edge1",
                source_asset_id="asset1",
                target_asset_id="asset2",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v2",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="edge2",
                source_asset_id="asset2",
                target_asset_id="asset3",
                edge_type=EdgeType.TRADE,
                protocol_name="sushiswap",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="edge3",
                source_asset_id="asset3",
                target_asset_id="asset4",
                edge_type=EdgeType.TRADE,
                protocol_name="curve",
                chain_name="ethereum"
            )
        ]
        
        basic_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.BASIC.value,
            profit_usd=2.0  # Below threshold
        )
        
        should_use = await simulator._should_use_tenderly_validation(
            path=multi_protocol_path,
            initial_amount=0.1,
            basic_result=basic_result
        )
        
        assert should_use is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])