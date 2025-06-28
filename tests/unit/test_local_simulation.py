"""Tests for local simulation fallback functionality."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import subprocess
import tempfile
import time

from yield_arbitrage.execution.hybrid_simulator import (
    HybridPathSimulator,
    SimulationMode,
    SimulatorConfig,
    SimulationResult,
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType, EdgeState


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
def simulator(mock_dependencies):
    """Create simulator for testing."""
    mock_redis, mock_oracle = mock_dependencies
    
    config = SimulatorConfig(
        confidence_threshold=0.7,
        min_liquidity_threshold=10000.0,
        local_rpc_url="http://localhost:8545"
    )
    
    return HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=config
    )


@pytest.fixture
def sample_path():
    """Create a sample path for testing."""
    return [
        YieldGraphEdge(
            edge_id="eth_usdc_test",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        )
    ]


@pytest.mark.asyncio
class TestLocalSimulation:
    """Test local simulation functionality."""
    
    async def test_anvil_not_available(self, simulator, sample_path):
        """Test behavior when Anvil is not available."""
        with patch('subprocess.run') as mock_run:
            # Mock Anvil not being available
            mock_run.side_effect = FileNotFoundError("anvil not found")
            
            result = await simulator._simulate_local(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH"
            )
            
            assert result.success is False
            assert "Anvil not found" in result.revert_reason
            assert result.simulation_mode == SimulationMode.LOCAL.value
    
    async def test_anvil_version_check_fails(self, simulator, sample_path):
        """Test behavior when Anvil version check fails."""
        with patch('subprocess.run') as mock_run:
            # Mock Anvil version check failing
            mock_process = Mock()
            mock_process.returncode = 1
            mock_process.stdout = ""
            mock_run.return_value = mock_process
            
            result = await simulator._simulate_local(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH"
            )
            
            assert result.success is False
            assert "Anvil not available" in result.revert_reason
    
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    @patch('socket.socket')
    async def test_anvil_startup_timeout(self, mock_socket, mock_run, mock_popen, simulator, sample_path):
        """Test behavior when Anvil fails to start within timeout."""
        # Mock Anvil version check success
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "anvil 0.1.0"
        mock_run.return_value = version_result
        
        # Mock Anvil process
        mock_process = Mock()
        mock_popen.return_value = mock_process
        
        # Mock socket connection always failing (Anvil never becomes ready)
        mock_sock_instance = Mock()
        mock_sock_instance.connect_ex.return_value = 1  # Connection failed
        mock_socket.return_value = mock_sock_instance
        
        result = await simulator._simulate_local(
            path=sample_path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH"
        )
        
        assert result.success is False
        assert "Anvil failed to start or become ready" in result.revert_reason
        assert mock_process.terminate.called
    
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    @patch('socket.socket')
    async def test_transaction_simulation_failure(self, mock_socket, mock_run, mock_popen, simulator, sample_path):
        """Test behavior when transaction simulation fails."""
        # Mock Anvil version check success
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "anvil 0.1.0"
        
        # Mock successful cast call (failure case)
        cast_result = Mock()
        cast_result.returncode = 1
        cast_result.stderr = "execution reverted"
        
        # Mock other subprocess calls
        def mock_run_side_effect(*args, **kwargs):
            if "anvil" in args[0][0]:
                return version_result
            elif "cast" in args[0][0] and "call" in args[0]:
                return cast_result
            else:
                success_result = Mock()
                success_result.returncode = 0
                success_result.stdout = "success"
                return success_result
        
        mock_run.side_effect = mock_run_side_effect
        
        # Mock Anvil process
        mock_process = Mock()
        mock_popen.return_value = mock_process
        
        # Mock socket connection success (Anvil becomes ready)
        mock_sock_instance = Mock()
        mock_sock_instance.connect_ex.return_value = 0  # Connection success
        mock_socket.return_value = mock_sock_instance
        
        # Mock transaction builder
        with patch('yield_arbitrage.execution.transaction_builder.TransactionBuilder') as mock_tx_builder:
            mock_builder = Mock()
            mock_tx = Mock()
            mock_tx.from_address = "0xtest"
            mock_tx.to_address = "0xtest2"
            mock_tx.data = "0xdata"
            mock_tx.value = 0
            mock_builder.build_path_transactions.return_value = [mock_tx]
            mock_tx_builder.return_value = mock_builder
            
            result = await simulator._simulate_local(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH"
            )
            
            assert result.success is False
            assert "Step 1 failed" in result.revert_reason
            assert "execution reverted" in result.revert_reason
            assert result.failed_at_step == 1
    
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    @patch('socket.socket')
    async def test_successful_local_simulation(self, mock_socket, mock_run, mock_popen, simulator, sample_path):
        """Test successful local simulation."""
        # Mock Anvil version check success
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "anvil 0.1.0"
        
        # Mock successful subprocess calls
        success_result = Mock()
        success_result.returncode = 0
        success_result.stdout = "150000"  # Gas estimate
        
        send_result = Mock()
        send_result.returncode = 0
        send_result.stdout = "0x1234567890abcdef"  # Transaction hash
        
        def mock_run_side_effect(*args, **kwargs):
            if "anvil" in args[0][0]:
                return version_result
            elif "cast" in args[0][0] and "send" in args[0]:
                return send_result
            else:
                return success_result
        
        mock_run.side_effect = mock_run_side_effect
        
        # Mock Anvil process
        mock_process = Mock()
        mock_popen.return_value = mock_process
        
        # Mock socket connection success
        mock_sock_instance = Mock()
        mock_sock_instance.connect_ex.return_value = 0
        mock_socket.return_value = mock_sock_instance
        
        # Mock transaction builder
        with patch('yield_arbitrage.execution.transaction_builder.TransactionBuilder') as mock_tx_builder:
            mock_builder = Mock()
            mock_tx = Mock()
            mock_tx.from_address = "0xtest"
            mock_tx.to_address = "0xtest2"
            mock_tx.data = "0xdata"
            mock_tx.value = 0
            mock_builder.build_path_transactions.return_value = [mock_tx]
            mock_tx_builder.return_value = mock_builder
            
            result = await simulator._simulate_local(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH"
            )
            
            assert result.success is True
            assert result.simulation_mode == SimulationMode.LOCAL.value
            assert result.gas_used == 150000
            assert result.gas_cost_usd > 0
            assert len(result.path_details) == 1
            assert result.path_details[0]["transaction_hash"] == "0x1234567890abcdef"
            assert result.path_details[0]["simulation_method"] == "anvil_local"
    
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    @patch('socket.socket')
    async def test_high_gas_warning(self, mock_socket, mock_run, mock_popen, simulator, sample_path):
        """Test that high gas usage generates warnings."""
        # Mock Anvil version check success
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "anvil 0.1.0"
        
        # Mock high gas estimate
        high_gas_result = Mock()
        high_gas_result.returncode = 0
        high_gas_result.stdout = "600000"  # High gas
        
        send_result = Mock()
        send_result.returncode = 0
        send_result.stdout = "0x1234567890abcdef"
        
        def mock_run_side_effect(*args, **kwargs):
            if "anvil" in args[0][0]:
                return version_result
            elif "cast" in args[0][0] and "estimate" in args[0]:
                return high_gas_result
            elif "cast" in args[0][0] and "send" in args[0]:
                return send_result
            else:
                success_result = Mock()
                success_result.returncode = 0
                success_result.stdout = "success"
                return success_result
        
        mock_run.side_effect = mock_run_side_effect
        
        # Mock Anvil process and socket
        mock_process = Mock()
        mock_popen.return_value = mock_process
        
        mock_sock_instance = Mock()
        mock_sock_instance.connect_ex.return_value = 0
        mock_socket.return_value = mock_sock_instance
        
        # Mock transaction builder
        with patch('yield_arbitrage.execution.transaction_builder.TransactionBuilder') as mock_tx_builder:
            mock_builder = Mock()
            mock_tx = Mock()
            mock_tx.from_address = "0xtest"
            mock_tx.to_address = "0xtest2"
            mock_tx.data = "0xdata"
            mock_tx.value = 0
            mock_builder.build_path_transactions.return_value = [mock_tx]
            mock_tx_builder.return_value = mock_builder
            
            result = await simulator._simulate_local(
                path=sample_path,
                initial_amount=1.0,
                start_asset_id="ETH_MAINNET_WETH"
            )
            
            assert result.success is True
            assert result.gas_used == 600000
            assert any("High gas usage" in warning for warning in result.warnings)


@pytest.mark.asyncio  
class TestHybridWithLocalFallback:
    """Test hybrid simulation with local fallback."""
    
    async def test_tenderly_fails_local_succeeds(self, simulator, sample_path):
        """Test that local simulation is used when Tenderly fails."""
        # Mock good edge state for basic simulation
        good_state = EdgeState(
            conversion_rate=2500.0,
            liquidity_usd=100000.0,
            gas_cost_usd=5.0,
            confidence_score=0.9,
            last_updated_timestamp=time.time()
        )
        
        simulator._get_edge_state = AsyncMock(return_value=good_state)
        
        # Mock failing Tenderly result
        failing_tenderly_result = SimulationResult(
            success=False,
            simulation_mode=SimulationMode.TENDERLY.value,
            revert_reason="Tenderly API error"
        )
        
        # Mock successful local result
        successful_local_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.LOCAL.value,
            profit_usd=10.0,
            gas_cost_usd=7.5,
            warnings=[]
        )
        
        simulator._simulate_tenderly = AsyncMock(return_value=failing_tenderly_result)
        simulator._simulate_local = AsyncMock(return_value=successful_local_result)
        
        result = await simulator.simulate_path(
            path=sample_path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.HYBRID
        )
        
        assert result.success is True
        assert result.profit_usd == 10.0
        assert "Tenderly failed: Tenderly API error" in result.warnings
        assert "Used local simulation as fallback" in result.warnings
    
    async def test_both_tenderly_and_local_fail(self, simulator, sample_path):
        """Test behavior when both Tenderly and local simulation fail."""
        # Mock good edge state for basic simulation
        good_state = EdgeState(
            conversion_rate=2500.0,
            liquidity_usd=100000.0,
            gas_cost_usd=5.0,
            confidence_score=0.9,
            last_updated_timestamp=time.time()
        )
        
        simulator._get_edge_state = AsyncMock(return_value=good_state)
        
        # Mock failing Tenderly result
        failing_tenderly_result = SimulationResult(
            success=False,
            simulation_mode=SimulationMode.TENDERLY.value,
            revert_reason="Tenderly API error"
        )
        
        # Mock failing local result
        failing_local_result = SimulationResult(
            success=False,
            simulation_mode=SimulationMode.LOCAL.value,
            revert_reason="Anvil setup failed"
        )
        
        simulator._simulate_tenderly = AsyncMock(return_value=failing_tenderly_result)
        simulator._simulate_local = AsyncMock(return_value=failing_local_result)
        
        result = await simulator.simulate_path(
            path=sample_path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.HYBRID
        )
        
        # Should fall back to basic simulation result since both on-chain methods failed
        assert result.success is True  # Basic simulation should succeed
        assert "Local fallback also failed: Anvil setup failed" in result.warnings
    
    async def test_local_fallback_disabled(self, mock_dependencies, sample_path):
        """Test that local fallback is not used when local_rpc_url is not configured."""
        mock_redis, mock_oracle = mock_dependencies
        
        # Configure without local RPC URL
        config = SimulatorConfig(
            confidence_threshold=0.7,
            min_liquidity_threshold=10000.0,
            local_rpc_url=""  # No local RPC configured
        )
        
        simulator = HybridPathSimulator(
            redis_client=mock_redis,
            asset_oracle=mock_oracle,
            config=config
        )
        
        # Mock good edge state for basic simulation
        good_state = EdgeState(
            conversion_rate=2500.0,
            liquidity_usd=100000.0,
            gas_cost_usd=5.0,
            confidence_score=0.9,
            last_updated_timestamp=time.time()
        )
        
        simulator._get_edge_state = AsyncMock(return_value=good_state)
        
        # Mock failing Tenderly result
        failing_tenderly_result = SimulationResult(
            success=False,
            simulation_mode=SimulationMode.TENDERLY.value,
            revert_reason="Tenderly API error"
        )
        
        simulator._simulate_tenderly = AsyncMock(return_value=failing_tenderly_result)
        simulator._simulate_local = AsyncMock()  # Should not be called
        
        result = await simulator.simulate_path(
            path=sample_path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.HYBRID
        )
        
        # Should use basic simulation result, not attempt local fallback
        assert result.success is True  # Basic simulation should succeed
        assert simulator._simulate_local.call_count == 0  # Local should not be called
        assert "Used local simulation as fallback" not in (result.warnings or [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])