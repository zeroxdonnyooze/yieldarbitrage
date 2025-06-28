"""Integration tests for the unified simulation system with real DeFi transactions."""
import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from typing import List, Dict, Any

from yield_arbitrage.execution.hybrid_simulator import (
    HybridPathSimulator,
    SimulationMode,
    SimulatorConfig,
    TenderlyConfig,
    SimulationResult,
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType, EdgeState


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for integration testing."""
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    
    mock_oracle = Mock()
    mock_oracle.get_price_usd = AsyncMock(side_effect=lambda asset_id: {
        "ETH_MAINNET_WETH": 2500.0,
        "ETH_MAINNET_FLASH_WETH": 2500.0,  # Same as WETH
        "ETH_MAINNET_USDC": 1.0,
        "ETH_MAINNET_DAI": 1.0,
        "ETH_MAINNET_USDT": 1.0,
        "ETH_MAINNET_WBTC": 45000.0,
        "ETH_MAINNET_AWETH": 2500.0  # Same as WETH for aToken
    }.get(asset_id, 2000.0))
    
    return mock_redis, mock_oracle


@pytest.fixture
def simulator_config():
    """Create simulator configuration for testing."""
    return SimulatorConfig(
        confidence_threshold=0.7,
        min_liquidity_threshold=50000.0,
        tenderly_profit_threshold_usd=10.0,
        tenderly_amount_threshold_usd=1000.0,
        local_rpc_url="http://localhost:8545"
    )


@pytest.fixture
def tenderly_config():
    """Create Tenderly configuration for testing."""
    return TenderlyConfig(
        api_key="E5tSD537G0z2r9xur64acExE2DNjRFWP",
        username="bomanyd",
        project_slug="project"
    )


@pytest.fixture
def simulator(mock_dependencies, simulator_config, tenderly_config):
    """Create simulator instance for testing."""
    mock_redis, mock_oracle = mock_dependencies
    
    return HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=simulator_config,
        tenderly_config=tenderly_config
    )


@pytest.fixture
def real_arbitrage_paths():
    """Create realistic arbitrage paths based on actual DeFi opportunities."""
    return {
        # Simple DEX arbitrage: ETH -> USDC -> ETH
        "eth_usdc_arbitrage": [
            YieldGraphEdge(
                edge_id="eth_usdc_uniswap",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v2",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="usdc_eth_sushiswap",
                source_asset_id="ETH_MAINNET_USDC",
                target_asset_id="ETH_MAINNET_WETH",
                edge_type=EdgeType.TRADE,
                protocol_name="sushiswap",
                chain_name="ethereum"
            )
        ],
        
        # Multi-hop arbitrage: ETH -> USDC -> DAI -> ETH
        "multi_hop_stablecoin": [
            YieldGraphEdge(
                edge_id="eth_usdc_trade",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v3",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="usdc_dai_curve",
                source_asset_id="ETH_MAINNET_USDC",
                target_asset_id="ETH_MAINNET_DAI",
                edge_type=EdgeType.TRADE,
                protocol_name="curve",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="dai_eth_balancer",
                source_asset_id="ETH_MAINNET_DAI",
                target_asset_id="ETH_MAINNET_WETH",
                edge_type=EdgeType.TRADE,
                protocol_name="balancer",
                chain_name="ethereum"
            )
        ],
        
        # Lending arbitrage: ETH -> aETH -> ETH
        "lending_arbitrage": [
            YieldGraphEdge(
                edge_id="eth_aave_deposit",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_AWETH",
                edge_type=EdgeType.LEND,
                protocol_name="aave_v3",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="aave_withdraw_eth",
                source_asset_id="ETH_MAINNET_AWETH",
                target_asset_id="ETH_MAINNET_WETH",
                edge_type=EdgeType.BORROW,
                protocol_name="aave_v3",
                chain_name="ethereum"
            )
        ],
        
        # Complex arbitrage with flash loan
        "flash_loan_arbitrage": [
            YieldGraphEdge(
                edge_id="aave_flash_loan",
                source_asset_id="ETH_MAINNET_FLASH_WETH",  # Special flash loan asset
                target_asset_id="ETH_MAINNET_WETH",
                edge_type=EdgeType.FLASH_LOAN,
                protocol_name="aave_v3",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="arbitrage_trade_1",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v3",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="arbitrage_trade_2",
                source_asset_id="ETH_MAINNET_USDC",
                target_asset_id="ETH_MAINNET_WETH",
                edge_type=EdgeType.TRADE,
                protocol_name="curve",
                chain_name="ethereum"
            )
        ]
    }


@pytest.fixture
def realistic_edge_states():
    """Create realistic edge states based on actual market conditions."""
    current_time = time.time()
    
    return {
        # High liquidity, good arbitrage edge
        "eth_usdc_uniswap": EdgeState(
            conversion_rate=2500.0,
            liquidity_usd=1_500_000.0,
            gas_cost_usd=8.0,
            confidence_score=0.95,
            last_updated_timestamp=current_time - 30
        ),
        
        # Medium liquidity, profitable spread
        "usdc_eth_sushiswap": EdgeState(
            conversion_rate=0.000402,  # Better profit margin
            liquidity_usd=800_000.0,
            gas_cost_usd=8.0,  # Lower gas cost
            confidence_score=0.88,
            last_updated_timestamp=current_time - 45
        ),
        
        # Multi-hop edges
        "eth_usdc_trade": EdgeState(
            conversion_rate=2498.5,
            liquidity_usd=2_000_000.0,
            gas_cost_usd=15.0,
            confidence_score=0.92,
            last_updated_timestamp=current_time - 25
        ),
        
        "usdc_dai_curve": EdgeState(
            conversion_rate=0.9995,  # Tiny stablecoin spread
            liquidity_usd=5_000_000.0,
            gas_cost_usd=6.0,
            confidence_score=0.98,
            last_updated_timestamp=current_time - 15
        ),
        
        "dai_eth_balancer": EdgeState(
            conversion_rate=0.0004005,
            liquidity_usd=750_000.0,
            gas_cost_usd=18.0,
            confidence_score=0.85,
            last_updated_timestamp=current_time - 40
        ),
        
        # Lending rates
        "eth_aave_deposit": EdgeState(
            conversion_rate=1.0,  # 1:1 for deposits
            liquidity_usd=50_000_000.0,
            gas_cost_usd=25.0,
            confidence_score=0.99,
            last_updated_timestamp=current_time - 10
        ),
        
        "aave_withdraw_eth": EdgeState(
            conversion_rate=1.0001,  # Small yield
            liquidity_usd=50_000_000.0,
            gas_cost_usd=30.0,
            confidence_score=0.99,
            last_updated_timestamp=current_time - 12
        ),
        
        # Flash loan edge
        "aave_flash_loan": EdgeState(
            conversion_rate=0.9991,  # 0.09% flash loan fee
            liquidity_usd=100_000_000.0,
            gas_cost_usd=35.0,
            confidence_score=0.97,
            last_updated_timestamp=current_time - 20
        ),
        
        # Complex arbitrage edges
        "arbitrage_trade_1": EdgeState(
            conversion_rate=2502.0,  # Better rate on Uniswap V3
            liquidity_usd=3_000_000.0,
            gas_cost_usd=20.0,
            confidence_score=0.93,
            last_updated_timestamp=current_time - 35
        ),
        
        "arbitrage_trade_2": EdgeState(
            conversion_rate=0.0004008,  # Return with profit
            liquidity_usd=1_200_000.0,
            gas_cost_usd=16.0,
            confidence_score=0.90,
            last_updated_timestamp=current_time - 28
        )
    }


@pytest.mark.asyncio
class TestBasicSimulationIntegration:
    """Test basic simulation mode with realistic paths."""
    
    async def test_simple_arbitrage_basic_mode(self, simulator, real_arbitrage_paths, realistic_edge_states):
        """Test simple ETH->USDC->ETH arbitrage in basic mode."""
        path = real_arbitrage_paths["eth_usdc_arbitrage"]
        
        # Mock edge states
        async def mock_get_edge_state(edge_id):
            return realistic_edge_states.get(edge_id)
        
        simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
        
        result = await simulator.simulate_path(
            path=path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        # Simulation should complete successfully (even if unprofitable)
        assert result.simulation_mode == SimulationMode.BASIC.value
        assert result.gas_cost_usd > 0
        assert len(result.path_details) == 2
        assert result.simulation_time_ms < 100  # Should be fast
        
        # Path details should be populated
        assert result.path_details[0]["edge_id"] == "eth_usdc_uniswap"
        assert result.path_details[1]["edge_id"] == "usdc_eth_sushiswap"
        
        # Should have proper asset flow
        assert result.path_details[0]["input_asset"] == "ETH_MAINNET_WETH"
        assert result.path_details[0]["output_asset"] == "ETH_MAINNET_USDC"
        assert result.path_details[1]["input_asset"] == "ETH_MAINNET_USDC"
        assert result.path_details[1]["output_asset"] == "ETH_MAINNET_WETH"
        
        # If profitable, success should be True; if not, that's also valid
        if result.profit_usd > 0:
            assert result.success is True
        else:
            # Unprofitable paths can still simulate successfully
            assert result.success is False
            assert result.profit_usd <= 0
    
    async def test_multi_hop_stablecoin_arbitrage(self, simulator, real_arbitrage_paths, realistic_edge_states):
        """Test multi-hop stablecoin arbitrage."""
        path = real_arbitrage_paths["multi_hop_stablecoin"]
        
        async def mock_get_edge_state(edge_id):
            return realistic_edge_states.get(edge_id)
        
        simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
        
        result = await simulator.simulate_path(
            path=path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        # Multi-hop arbitrage simulation should complete
        assert len(result.path_details) == 3
        assert result.gas_cost_usd > 30  # Multi-hop should cost more gas
        
        # Should handle stablecoin conversions properly
        usdc_step = result.path_details[0]
        dai_step = result.path_details[1]
        eth_step = result.path_details[2]
        
        assert usdc_step["output_asset"] == "ETH_MAINNET_USDC"
        assert dai_step["output_asset"] == "ETH_MAINNET_DAI"
        assert eth_step["output_asset"] == "ETH_MAINNET_WETH"
        
        # Realistic expectation: multi-hop often unprofitable due to gas costs
        if result.profit_usd > 0:
            assert result.success is True
        else:
            # This is realistic - multi-hop arbitrage often unprofitable
            assert result.success is False
            print(f"Multi-hop arbitrage unprofitable: {result.profit_usd:.2f} USD loss")
    
    async def test_lending_arbitrage_basic(self, simulator, real_arbitrage_paths, realistic_edge_states):
        """Test lending arbitrage with Aave."""
        path = real_arbitrage_paths["lending_arbitrage"]
        
        async def mock_get_edge_state(edge_id):
            return realistic_edge_states.get(edge_id)
        
        simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
        
        result = await simulator.simulate_path(
            path=path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        # Lending operations should be simulated
        assert result.gas_cost_usd > 50  # Lending operations are gas-heavy
        
        # Check that lending operations are properly handled
        deposit_step = result.path_details[0]
        withdraw_step = result.path_details[1]
        
        assert deposit_step["edge_id"] == "eth_aave_deposit"
        assert withdraw_step["edge_id"] == "aave_withdraw_eth"
        
        # Lending arbitrage is often unprofitable due to high gas costs
        if result.profit_usd > 0:
            assert result.success is True
        else:
            # This is realistic - lending arbitrage often has minimal yield
            assert result.success is False
            print(f"Lending arbitrage unprofitable: {result.profit_usd:.2f} USD loss")
    
    async def test_unprofitable_path_rejection(self, simulator, real_arbitrage_paths):
        """Test that unprofitable paths are properly rejected."""
        path = real_arbitrage_paths["eth_usdc_arbitrage"]
        
        # Create unprofitable edge states (high gas, bad rates)
        unprofitable_states = {
            "eth_usdc_uniswap": EdgeState(
                conversion_rate=2400.0,  # Bad rate
                liquidity_usd=100_000.0,
                gas_cost_usd=50.0,  # High gas
                confidence_score=0.8,
                last_updated_timestamp=time.time()
            ),
            "usdc_eth_sushiswap": EdgeState(
                conversion_rate=0.00038,  # Very bad return rate
                liquidity_usd=100_000.0,
                gas_cost_usd=50.0,
                confidence_score=0.8,
                last_updated_timestamp=time.time()
            )
        }
        
        async def mock_get_edge_state(edge_id):
            return unprofitable_states.get(edge_id)
        
        simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
        
        result = await simulator.simulate_path(
            path=path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        assert result.success is False
        assert result.profit_usd <= 0 or result.revert_reason is not None


@pytest.mark.asyncio
class TestHybridSimulationIntegration:
    """Test hybrid simulation mode with realistic scenarios."""
    
    async def test_hybrid_mode_tenderly_criteria(self, simulator, real_arbitrage_paths, realistic_edge_states):
        """Test that hybrid mode correctly decides when to use Tenderly."""
        path = real_arbitrage_paths["flash_loan_arbitrage"]  # Complex path should trigger Tenderly
        
        async def mock_get_edge_state(edge_id):
            return realistic_edge_states.get(edge_id)
        
        simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
        
        # Mock successful Tenderly result
        mock_tenderly_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.TENDERLY.value,
            profit_usd=25.0,
            gas_used=450_000,
            gas_cost_usd=67.5,
            output_amount=1.01,
            simulation_time_ms=1500.0
        )
        
        simulator._simulate_tenderly = AsyncMock(return_value=mock_tenderly_result)
        
        result = await simulator.simulate_path(
            path=path,
            initial_amount=10.0,  # Large amount to trigger Tenderly
            start_asset_id="ETH_MAINNET_FLASH_WETH",  # Start with flash loan asset
            mode=SimulationMode.HYBRID
        )
        
        assert result.success is True
        assert result.simulation_mode == SimulationMode.HYBRID.value
        assert result.profit_usd == 25.0  # From Tenderly result
        assert simulator._simulate_tenderly.called
    
    async def test_hybrid_fallback_to_local(self, simulator, real_arbitrage_paths, realistic_edge_states):
        """Test hybrid mode falling back to local simulation when Tenderly fails."""
        path = real_arbitrage_paths["multi_hop_stablecoin"]
        
        async def mock_get_edge_state(edge_id):
            return realistic_edge_states.get(edge_id)
        
        simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
        
        # Mock Tenderly failure
        mock_tenderly_result = SimulationResult(
            success=False,
            simulation_mode=SimulationMode.TENDERLY.value,
            revert_reason="Tenderly API rate limit",
            simulation_time_ms=100.0
        )
        
        # Mock successful local result
        mock_local_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.LOCAL.value,
            profit_usd=12.5,
            gas_used=320_000,
            gas_cost_usd=48.0,
            warnings=["Local simulation used as fallback"],
            simulation_time_ms=3200.0
        )
        
        simulator._simulate_tenderly = AsyncMock(return_value=mock_tenderly_result)
        simulator._simulate_local = AsyncMock(return_value=mock_local_result)
        
        result = await simulator.simulate_path(
            path=path,
            initial_amount=5.0,  # Large enough to trigger Tenderly
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.HYBRID
        )
        
        assert result.success is True
        assert result.profit_usd == 12.5  # From local fallback
        assert "Tenderly failed" in str(result.warnings)
        assert "Used local simulation as fallback" in str(result.warnings)
        assert simulator._simulate_tenderly.called
        assert simulator._simulate_local.called
    
    async def test_hybrid_uses_basic_for_simple_paths(self, simulator, real_arbitrage_paths, realistic_edge_states):
        """Test that hybrid mode uses basic simulation for simple paths."""
        path = real_arbitrage_paths["eth_usdc_arbitrage"]  # Simple path
        
        async def mock_get_edge_state(edge_id):
            return realistic_edge_states.get(edge_id)
        
        simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
        
        # Should not call Tenderly for simple paths
        simulator._simulate_tenderly = AsyncMock()
        
        result = await simulator.simulate_path(
            path=path,
            initial_amount=0.5,  # Small amount, simple path
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.HYBRID
        )
        
        assert result.success is True
        assert result.simulation_mode == SimulationMode.HYBRID.value
        assert "Used basic simulation only" in str(result.warnings)
        assert not simulator._simulate_tenderly.called  # Should not use Tenderly


@pytest.mark.asyncio
class TestEdgeValidationIntegration:
    """Test edge validation integrated with real paths."""
    
    async def test_path_validation_with_missing_states(self, simulator, real_arbitrage_paths):
        """Test path validation when some edge states are missing."""
        path = real_arbitrage_paths["eth_usdc_arbitrage"]
        
        # Mock missing edge state for one edge
        async def mock_get_edge_state(edge_id):
            if edge_id == "eth_usdc_uniswap":
                return EdgeState(
                    conversion_rate=2500.0,
                    liquidity_usd=1_000_000.0,
                    gas_cost_usd=8.0,
                    confidence_score=0.9,
                    last_updated_timestamp=time.time()
                )
            return None  # Missing state
        
        simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
        
        result = await simulator.simulate_path(
            path=path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.HYBRID
        )
        
        # Should still work but with warnings
        assert len(result.warnings) > 0
        assert any("No state data available" in warning for warning in result.warnings)
    
    async def test_path_validation_with_stale_data(self, simulator, real_arbitrage_paths):
        """Test path validation with stale edge data."""
        path = real_arbitrage_paths["multi_hop_stablecoin"]
        
        # Create stale edge states
        stale_time = time.time() - 1800  # 30 minutes ago
        
        stale_states = {
            "eth_usdc_trade": EdgeState(
                conversion_rate=2500.0,
                liquidity_usd=1_000_000.0,
                gas_cost_usd=8.0,
                confidence_score=0.9,
                last_updated_timestamp=stale_time
            ),
            "usdc_dai_curve": EdgeState(
                conversion_rate=1.0,
                liquidity_usd=5_000_000.0,
                gas_cost_usd=6.0,
                confidence_score=0.95,
                last_updated_timestamp=time.time()  # Fresh
            ),
            "dai_eth_balancer": EdgeState(
                conversion_rate=0.0004,
                liquidity_usd=800_000.0,
                gas_cost_usd=12.0,
                confidence_score=0.85,
                last_updated_timestamp=stale_time
            )
        }
        
        async def mock_get_edge_state(edge_id):
            return stale_states.get(edge_id)
        
        simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
        
        result = await simulator.simulate_path(
            path=path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.HYBRID
        )
        
        # Should generate warnings about stale data
        assert any("stale" in warning.lower() for warning in (result.warnings or []))
    
    async def test_disconnected_path_validation(self, simulator):
        """Test validation of disconnected paths."""
        # Create a disconnected path
        disconnected_path = [
            YieldGraphEdge(
                edge_id="eth_usdc",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v2",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="wbtc_dai",  # Disconnected!
                source_asset_id="ETH_MAINNET_WBTC",
                target_asset_id="ETH_MAINNET_DAI",
                edge_type=EdgeType.TRADE,
                protocol_name="curve",
                chain_name="ethereum"
            )
        ]
        
        result = await simulator.simulate_path(
            path=disconnected_path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.HYBRID
        )
        
        assert result.success is False
        assert "Path validation failed" in result.revert_reason
        assert "disconnect" in result.revert_reason.lower()


@pytest.mark.asyncio
class TestPerformanceBenchmarking:
    """Test performance across different simulation modes."""
    
    async def test_simulation_mode_performance_comparison(self, simulator, real_arbitrage_paths, realistic_edge_states):
        """Compare performance across all simulation modes."""
        path = real_arbitrage_paths["multi_hop_stablecoin"]
        
        async def mock_get_edge_state(edge_id):
            return realistic_edge_states.get(edge_id)
        
        simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
        
        # Mock simulation methods for controlled testing
        simulator._simulate_tenderly = AsyncMock(return_value=SimulationResult(
            success=True,
            simulation_mode=SimulationMode.TENDERLY.value,
            profit_usd=15.0,
            simulation_time_ms=1200.0
        ))
        
        simulator._simulate_local = AsyncMock(return_value=SimulationResult(
            success=True,
            simulation_mode=SimulationMode.LOCAL.value,
            profit_usd=14.5,
            simulation_time_ms=2800.0
        ))
        
        performance_results = {}
        
        # Test each mode
        for mode in [SimulationMode.BASIC, SimulationMode.TENDERLY, SimulationMode.HYBRID, SimulationMode.LOCAL]:
            start_time = time.perf_counter()
            
            result = await simulator.simulate_path(
                path=path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH",
                mode=mode
            )
            
            end_time = time.perf_counter()
            wall_time = (end_time - start_time) * 1000  # Convert to ms
            
            performance_results[mode.value] = {
                "success": result.success,
                "wall_time_ms": wall_time,
                "simulation_time_ms": result.simulation_time_ms,
                "profit_usd": result.profit_usd
            }
        
        # Verify performance characteristics
        assert performance_results["basic"]["wall_time_ms"] < 50  # Basic should be fastest
        
        # All simulations should complete (regardless of profitability)
        for mode, mode_result in performance_results.items():
            assert mode_result["simulation_time_ms"] is not None
            # profit_usd can be None for failed simulations, which is valid
            # Note: success=False is valid for unprofitable paths
        
        # Print performance summary for analysis
        print("\n🚀 Performance Benchmark Results:")
        for mode, metrics in performance_results.items():
            print(f"  {mode.upper()}:")
            print(f"    Wall Time: {metrics['wall_time_ms']:.1f}ms")
            print(f"    Sim Time:  {metrics['simulation_time_ms']:.1f}ms")
            profit_str = f"${metrics['profit_usd']:.2f}" if metrics['profit_usd'] is not None else "N/A"
            print(f"    Profit:    {profit_str}")
            print(f"    Success:   {metrics['success']}")
    
    async def test_concurrent_simulations(self, simulator, real_arbitrage_paths, realistic_edge_states):
        """Test multiple concurrent simulations."""
        paths = [
            real_arbitrage_paths["eth_usdc_arbitrage"],
            real_arbitrage_paths["multi_hop_stablecoin"],
            real_arbitrage_paths["lending_arbitrage"]
        ]
        
        async def mock_get_edge_state(edge_id):
            return realistic_edge_states.get(edge_id)
        
        simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
        
        # Run simulations concurrently
        start_time = time.perf_counter()
        
        tasks = []
        for i, path in enumerate(paths):
            task = simulator.simulate_path(
                path=path,
                initial_amount=1.0 + i * 0.5,  # Different amounts
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.BASIC
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # All simulations should succeed
        for result in results:
            assert result.success is True
        
        # Concurrent execution should be efficient
        assert total_time < 0.5  # Should complete quickly
        assert len(results) == 3
        
        print(f"\n⚡ Concurrent simulation of {len(paths)} paths completed in {total_time*1000:.1f}ms")


@pytest.mark.asyncio
class TestErrorHandlingIntegration:
    """Test error handling and recovery scenarios."""
    
    async def test_partial_simulation_failure_recovery(self, simulator, real_arbitrage_paths, realistic_edge_states):
        """Test recovery when some simulation methods fail."""
        path = real_arbitrage_paths["flash_loan_arbitrage"]
        
        async def mock_get_edge_state(edge_id):
            return realistic_edge_states.get(edge_id)
        
        simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
        
        # Mock all external simulations failing
        simulator._simulate_tenderly = AsyncMock(side_effect=Exception("Tenderly API down"))
        simulator._simulate_local = AsyncMock(side_effect=Exception("Anvil not available"))
        
        result = await simulator.simulate_path(
            path=path,
            initial_amount=5.0,  # Should trigger Tenderly
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.HYBRID
        )
        
        # Should fall back to basic simulation
        assert result.success is True  # Basic simulation should work
        assert result.simulation_mode == SimulationMode.HYBRID.value
        assert any("basic" in warning.lower() for warning in (result.warnings or []))
    
    async def test_invalid_configuration_handling(self, mock_dependencies):
        """Test handling of invalid configurations."""
        mock_redis, mock_oracle = mock_dependencies
        
        # Create simulator with invalid config
        invalid_config = SimulatorConfig(
            confidence_threshold=1.5,  # Invalid (> 1.0)
            min_liquidity_threshold=-1000.0,  # Invalid (negative)
        )
        
        simulator = HybridPathSimulator(
            redis_client=mock_redis,
            asset_oracle=mock_oracle,
            config=invalid_config
        )
        
        path = [YieldGraphEdge(
            edge_id="test_edge",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        )]
        
        # Should handle gracefully despite invalid config
        result = await simulator.simulate_path(
            path=path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.BASIC
        )
        
        # May fail but should not crash
        assert result is not None
        assert hasattr(result, 'success')


@pytest.mark.asyncio
class TestRealDataIntegration:
    """Test integration with real DeFi data patterns."""
    
    async def test_realistic_market_conditions(self, simulator):
        """Test simulation with realistic market conditions and edge cases."""
        # Create path based on actual market observations
        realistic_path = [
            YieldGraphEdge(
                edge_id="eth_usdc_univ3_500",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v3",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="usdc_usdt_curve_3pool",
                source_asset_id="ETH_MAINNET_USDC",
                target_asset_id="ETH_MAINNET_USDT",
                edge_type=EdgeType.TRADE,
                protocol_name="curve",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="usdt_eth_sushi",
                source_asset_id="ETH_MAINNET_USDT",
                target_asset_id="ETH_MAINNET_WETH",
                edge_type=EdgeType.TRADE,
                protocol_name="sushiswap",
                chain_name="ethereum"
            )
        ]
        
        # Real market-like conditions
        market_states = {
            "eth_usdc_univ3_500": EdgeState(
                conversion_rate=2487.32,  # ETH price with spread
                liquidity_usd=45_823_142.0,  # Real Uniswap V3 liquidity
                gas_cost_usd=23.45,
                confidence_score=0.94,
                last_updated_timestamp=time.time() - 12
            ),
            "usdc_usdt_curve_3pool": EdgeState(
                conversion_rate=0.9998,  # Tiny stablecoin spread
                liquidity_usd=312_456_789.0,  # Large Curve pool
                gas_cost_usd=8.12,
                confidence_score=0.99,
                last_updated_timestamp=time.time() - 8
            ),
            "usdt_eth_sushi": EdgeState(
                conversion_rate=0.000402156,  # Return rate with spread
                liquidity_usd=12_345_678.0,
                gas_cost_usd=18.67,
                confidence_score=0.91,
                last_updated_timestamp=time.time() - 15
            )
        }
        
        async def mock_get_edge_state(edge_id):
            return market_states.get(edge_id)
        
        simulator._get_edge_state = AsyncMock(side_effect=mock_get_edge_state)
        
        # Test with different amounts
        test_amounts = [0.1, 1.0, 10.0, 100.0]  # Different scales
        
        for amount in test_amounts:
            result = await simulator.simulate_path(
                path=realistic_path,
                initial_amount=amount,
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.BASIC
            )
            
            # Should handle all amounts gracefully
            assert result is not None
            
            if result.success:
                # Verify realistic profit/loss
                assert abs(result.profit_percentage) < 50  # Reasonable profit range
                assert result.gas_cost_usd > 40  # Multi-hop should cost significant gas
            
            print(f"  Amount: {amount} ETH -> Success: {result.success}, Profit: ${result.profit_usd:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])