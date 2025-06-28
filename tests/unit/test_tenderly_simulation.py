"""Unit tests for Tenderly simulation mode in HybridPathSimulator."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from yield_arbitrage.execution.hybrid_simulator import (
    HybridPathSimulator,
    SimulationMode,
    SimulationResult,
    SimulatorConfig,
    TenderlyConfig,
)
from yield_arbitrage.execution.tenderly_client import (
    TenderlySimulationResult,
    TenderlyFork,
    TenderlyNetworkId,
)
from yield_arbitrage.execution.transaction_builder import TransactionBuilder
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType


class TestTenderlySimulation:
    """Test Tenderly simulation mode functionality."""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        client = Mock()
        client.get = AsyncMock(return_value=None)
        return client
    
    @pytest.fixture
    def mock_asset_oracle(self):
        """Create a mock asset oracle."""
        oracle = Mock()
        # ETH price for gas calculations
        oracle.get_price_usd = AsyncMock(return_value=2000.0)
        return oracle
    
    @pytest.fixture
    def tenderly_config(self):
        """Create a Tenderly configuration."""
        return TenderlyConfig(
            api_key="test_api_key",
            project_slug="test_project",
            username="test_user"
        )
    
    @pytest.fixture
    def simulator_config(self):
        """Create a simulator configuration."""
        return SimulatorConfig(
            default_gas_price_gwei=20.0,
            eth_price_usd=2000.0
        )
    
    @pytest.fixture
    def simulator_with_tenderly(
        self, mock_redis_client, mock_asset_oracle, simulator_config, tenderly_config
    ):
        """Create a HybridPathSimulator with Tenderly configuration."""
        return HybridPathSimulator(
            redis_client=mock_redis_client,
            asset_oracle=mock_asset_oracle,
            config=simulator_config,
            tenderly_config=tenderly_config
        )
    
    @pytest.fixture
    def sample_path(self):
        """Create a sample arbitrage path."""
        edge1 = YieldGraphEdge(
            edge_id="eth_usdc_uniswap",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswapv3",
            chain_name="ethereum"
        )
        
        edge2 = YieldGraphEdge(
            edge_id="usdc_eth_sushiswap",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_WETH",
            edge_type=EdgeType.TRADE,
            protocol_name="sushiswap",
            chain_name="ethereum"
        )
        
        return [edge1, edge2]
    
    @pytest.fixture
    def mock_tenderly_client(self):
        """Create a mock Tenderly client."""
        client = Mock()
        client.initialize = AsyncMock()
        client.create_fork = AsyncMock()
        client.delete_fork = AsyncMock(return_value=True)
        client.simulate_transaction = AsyncMock()
        return client
    
    @pytest.mark.asyncio
    async def test_tenderly_simulation_not_configured(
        self, mock_redis_client, mock_asset_oracle, simulator_config, sample_path
    ):
        """Test Tenderly simulation when not configured."""
        # Create simulator without Tenderly config
        simulator = HybridPathSimulator(
            redis_client=mock_redis_client,
            asset_oracle=mock_asset_oracle,
            config=simulator_config,
            tenderly_config=None
        )
        
        result = await simulator.simulate_path(
            path=sample_path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.TENDERLY
        )
        
        assert not result.success
        assert result.simulation_mode == SimulationMode.TENDERLY.value
        assert "not configured" in result.revert_reason
    
    @pytest.mark.asyncio
    async def test_successful_tenderly_simulation(
        self, simulator_with_tenderly, sample_path, mock_tenderly_client
    ):
        """Test successful Tenderly simulation."""
        # Mock fork creation
        mock_fork = TenderlyFork(
            fork_id="test_fork_123",
            network_id="1",
            block_number=18000000,
            created_at=datetime.utcnow()
        )
        mock_tenderly_client.create_fork.return_value = mock_fork
        
        # Mock successful transaction simulations
        successful_sim_result = TenderlySimulationResult(
            success=True,
            gas_used=150000,
            transaction_hash="0xabcdef123456",
            simulation_time_ms=250.0
        )
        mock_tenderly_client.simulate_transaction.return_value = successful_sim_result
        
        # Mock the transaction builder
        with patch('yield_arbitrage.execution.transaction_builder.TransactionBuilder') as mock_builder_class:
            mock_builder = Mock()
            mock_builder.build_path_transactions.return_value = [
                Mock(from_address="0xdead", to_address="0x123", data="0x456"),
                Mock(from_address="0xdead", to_address="0x789", data="0xabc")
            ]
            mock_builder_class.return_value = mock_builder
            
            # Inject the mock client
            simulator_with_tenderly._tenderly_client = mock_tenderly_client
            
            result = await simulator_with_tenderly.simulate_path(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.TENDERLY
            )
        
        assert result.success
        assert result.simulation_mode == SimulationMode.TENDERLY.value
        assert result.gas_used == 300000  # 2 transactions * 150k each
        assert result.gas_cost_usd > 0  # Should have calculated gas cost
        assert result.tenderly_fork_id == "test_fork_123"
        assert len(result.path_details) == 2
        
        # Verify fork was created and deleted
        mock_tenderly_client.create_fork.assert_called_once()
        mock_tenderly_client.delete_fork.assert_called_once_with("test_fork_123")
        
        # Verify transactions were simulated
        assert mock_tenderly_client.simulate_transaction.call_count == 2
    
    @pytest.mark.asyncio
    async def test_tenderly_simulation_failure_at_step(
        self, simulator_with_tenderly, sample_path, mock_tenderly_client
    ):
        """Test Tenderly simulation that fails at a specific step."""
        # Mock fork creation
        mock_fork = TenderlyFork(
            fork_id="test_fork_456",
            network_id="1",
            block_number=18000000,
            created_at=datetime.utcnow()
        )
        mock_tenderly_client.create_fork.return_value = mock_fork
        
        # Mock first transaction success, second transaction failure
        def simulate_side_effect(*args, **kwargs):
            if mock_tenderly_client.simulate_transaction.call_count == 1:
                return TenderlySimulationResult(
                    success=True,
                    gas_used=150000,
                    transaction_hash="0xsuccess",
                    simulation_time_ms=200.0
                )
            else:
                return TenderlySimulationResult(
                    success=False,
                    gas_used=50000,  # Partial gas usage before revert
                    error_message="Transaction reverted",
                    revert_reason="Insufficient allowance",
                    simulation_time_ms=180.0
                )
        
        mock_tenderly_client.simulate_transaction.side_effect = simulate_side_effect
        
        # Mock the transaction builder
        with patch('yield_arbitrage.execution.transaction_builder.TransactionBuilder') as mock_builder_class:
            mock_builder = Mock()
            mock_builder.build_path_transactions.return_value = [
                Mock(from_address="0xdead", to_address="0x123", data="0x456"),
                Mock(from_address="0xdead", to_address="0x789", data="0xabc")
            ]
            mock_builder_class.return_value = mock_builder
            
            # Inject the mock client
            simulator_with_tenderly._tenderly_client = mock_tenderly_client
            
            result = await simulator_with_tenderly.simulate_path(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.TENDERLY
            )
        
        assert not result.success
        assert result.simulation_mode == SimulationMode.TENDERLY.value
        assert result.failed_at_step == 2
        assert "Insufficient allowance" in result.revert_reason
        assert result.gas_used == 200000  # First success + partial second
        assert len(result.path_details) == 2  # Both steps logged
        
        # First step should be marked as successful
        assert result.path_details[0]["success"] is True
        assert result.path_details[1]["success"] is False
        assert result.path_details[1]["revert_reason"] == "Insufficient allowance"
    
    @pytest.mark.asyncio
    async def test_tenderly_fork_creation_failure(
        self, simulator_with_tenderly, sample_path, mock_tenderly_client
    ):
        """Test handling of fork creation failure."""
        # Mock fork creation failure
        mock_tenderly_client.create_fork.side_effect = Exception("API rate limit exceeded")
        
        # Inject the mock client
        simulator_with_tenderly._tenderly_client = mock_tenderly_client
        
        result = await simulator_with_tenderly.simulate_path(
            path=sample_path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.TENDERLY
        )
        
        assert not result.success
        assert result.simulation_mode == SimulationMode.TENDERLY.value
        assert "Fork creation failed" in result.revert_reason
        assert "API rate limit exceeded" in result.revert_reason
    
    @pytest.mark.asyncio
    async def test_tenderly_transaction_build_failure(
        self, simulator_with_tenderly, sample_path, mock_tenderly_client
    ):
        """Test handling of transaction building failure."""
        # Mock fork creation success
        mock_fork = TenderlyFork(
            fork_id="test_fork_789",
            network_id="1",
            block_number=18000000,
            created_at=datetime.utcnow()
        )
        mock_tenderly_client.create_fork.return_value = mock_fork
        
        # Mock transaction builder failure
        with patch('yield_arbitrage.execution.transaction_builder.TransactionBuilder') as mock_builder_class:
            mock_builder = Mock()
            mock_builder.build_path_transactions.side_effect = Exception("Unknown token")
            mock_builder_class.return_value = mock_builder
            
            # Inject the mock client
            simulator_with_tenderly._tenderly_client = mock_tenderly_client
            
            result = await simulator_with_tenderly.simulate_path(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.TENDERLY
            )
        
        assert not result.success
        assert result.simulation_mode == SimulationMode.TENDERLY.value
        assert "Path simulation error" in result.revert_reason
        assert "Unknown token" in result.revert_reason
        
        # Fork should still be cleaned up
        mock_tenderly_client.delete_fork.assert_called_once_with("test_fork_789")
    
    @pytest.mark.asyncio
    async def test_gas_cost_calculation(self, simulator_with_tenderly):
        """Test gas cost calculation in USD."""
        # Test with oracle providing ETH price
        gas_cost = await simulator_with_tenderly._calculate_gas_cost_usd(1000000)  # 1M gas
        
        # Expected: 1M gas * 20 gwei * 1e-9 ETH/gwei / 1e18 wei/ETH * 2000 USD/ETH
        # = 1M * 20 * 1e-9 * 2000 / 1e18 = 0.04 ETH * 2000 = 80 USD
        expected_cost = (1000000 * 20e9 / 1e18) * 2000  # Should be $40
        assert abs(gas_cost - expected_cost) < 0.01  # Within 1 cent
    
    @pytest.mark.asyncio
    async def test_gas_cost_calculation_oracle_failure(
        self, simulator_with_tenderly, mock_asset_oracle
    ):
        """Test gas cost calculation when oracle fails."""
        # Mock oracle failure
        mock_asset_oracle.get_price_usd.return_value = None
        
        gas_cost = await simulator_with_tenderly._calculate_gas_cost_usd(1000000)
        
        # Should fall back to config default (2000 USD)
        expected_cost = (1000000 * 20e9 / 1e18) * 2000
        assert abs(gas_cost - expected_cost) < 0.01
    
    @pytest.mark.asyncio
    async def test_high_gas_usage_warnings(
        self, simulator_with_tenderly, sample_path, mock_tenderly_client
    ):
        """Test warnings for high gas usage."""
        # Mock fork creation
        mock_fork = TenderlyFork(
            fork_id="test_fork_gas",
            network_id="1",
            block_number=18000000,
            created_at=datetime.utcnow()
        )
        mock_tenderly_client.create_fork.return_value = mock_fork
        
        # Mock high gas usage transactions
        high_gas_result = TenderlySimulationResult(
            success=True,
            gas_used=600000,  # High gas usage (> 500k)
            transaction_hash="0xhighgas",
            simulation_time_ms=500.0
        )
        mock_tenderly_client.simulate_transaction.return_value = high_gas_result
        
        # Mock the transaction builder
        with patch('yield_arbitrage.execution.transaction_builder.TransactionBuilder') as mock_builder_class:
            mock_builder = Mock()
            mock_builder.build_path_transactions.return_value = [
                Mock(from_address="0xdead", to_address="0x123", data="0x456"),
                Mock(from_address="0xdead", to_address="0x789", data="0xabc")
            ]
            mock_builder_class.return_value = mock_builder
            
            # Inject the mock client
            simulator_with_tenderly._tenderly_client = mock_tenderly_client
            
            result = await simulator_with_tenderly.simulate_path(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.TENDERLY
            )
        
        assert result.success
        assert result.gas_used == 1200000  # 2 * 600k
        
        # Check for warnings
        assert any("High gas usage" in warning for warning in result.warnings)
        assert any("Very high total gas usage" in warning for warning in result.warnings)
    
    @pytest.mark.asyncio
    async def test_slow_simulation_warnings(
        self, simulator_with_tenderly, sample_path, mock_tenderly_client
    ):
        """Test warnings for slow simulation times."""
        # Mock fork creation
        mock_fork = TenderlyFork(
            fork_id="test_fork_slow",
            network_id="1",
            block_number=18000000,
            created_at=datetime.utcnow()
        )
        mock_tenderly_client.create_fork.return_value = mock_fork
        
        # Mock slow simulation
        slow_sim_result = TenderlySimulationResult(
            success=True,
            gas_used=150000,
            transaction_hash="0xslow",
            simulation_time_ms=1500.0  # Slow simulation (> 1000ms)
        )
        mock_tenderly_client.simulate_transaction.return_value = slow_sim_result
        
        # Mock the transaction builder
        with patch('yield_arbitrage.execution.transaction_builder.TransactionBuilder') as mock_builder_class:
            mock_builder = Mock()
            mock_builder.build_path_transactions.return_value = [
                Mock(from_address="0xdead", to_address="0x123", data="0x456")
            ]
            mock_builder_class.return_value = mock_builder
            
            # Inject the mock client
            simulator_with_tenderly._tenderly_client = mock_tenderly_client
            
            result = await simulator_with_tenderly.simulate_path(
                path=sample_path[:1],  # Single edge path
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.TENDERLY
            )
        
        assert result.success
        
        # Check for slow simulation warning
        assert any("Slow simulation average" in warning for warning in result.warnings)
    
    @pytest.mark.asyncio
    async def test_transaction_count_mismatch(
        self, simulator_with_tenderly, sample_path, mock_tenderly_client
    ):
        """Test handling of transaction count mismatch."""
        # Mock fork creation
        mock_fork = TenderlyFork(
            fork_id="test_fork_mismatch",
            network_id="1",
            block_number=18000000,
            created_at=datetime.utcnow()
        )
        mock_tenderly_client.create_fork.return_value = mock_fork
        
        # Mock transaction builder returning wrong number of transactions
        with patch('yield_arbitrage.execution.transaction_builder.TransactionBuilder') as mock_builder_class:
            mock_builder = Mock()
            # Return only 1 transaction for 2 edges
            mock_builder.build_path_transactions.return_value = [
                Mock(from_address="0xdead", to_address="0x123", data="0x456")
            ]
            mock_builder_class.return_value = mock_builder
            
            # Inject the mock client
            simulator_with_tenderly._tenderly_client = mock_tenderly_client
            
            result = await simulator_with_tenderly.simulate_path(
                path=sample_path,  # 2 edges
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH",
                mode=SimulationMode.TENDERLY
            )
        
        assert not result.success
        assert "Transaction count mismatch" in result.revert_reason
        assert "1 != 2" in result.revert_reason
        
        # Fork should still be cleaned up
        mock_tenderly_client.delete_fork.assert_called_once_with("test_fork_mismatch")