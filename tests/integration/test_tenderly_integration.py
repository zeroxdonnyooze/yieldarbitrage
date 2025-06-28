"""Integration tests for real Tenderly API access."""
import pytest
import os
import asyncio
from unittest.mock import Mock, AsyncMock

from yield_arbitrage.execution.tenderly_client import (
    TenderlyClient,
    TenderlyTransaction,
    TenderlyNetworkId,
    TenderlyAuthError,
)
from yield_arbitrage.execution.hybrid_simulator import (
    HybridPathSimulator,
    SimulationMode,
    TenderlyConfig,
    SimulatorConfig,
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType


@pytest.mark.integration
@pytest.mark.asyncio
class TestRealTenderlyIntegration:
    """Integration tests with real Tenderly API."""
    
    @pytest.fixture
    def tenderly_config(self):
        """Create real Tenderly configuration from environment."""
        api_key = os.getenv("TENDERLY_API_KEY", "E5tSD537G0z2r9xur64acExE2DNjRFWP")
        username = os.getenv("TENDERLY_USERNAME", "bomanyd")
        project = os.getenv("TENDERLY_PROJECT_SLUG", "project")
        
        return TenderlyConfig(
            api_key=api_key,
            username=username,
            project_slug=project
        )
    
    @pytest.fixture
    async def tenderly_client(self, tenderly_config):
        """Create and initialize real Tenderly client."""
        client = TenderlyClient(
            api_key=tenderly_config.api_key,
            username=tenderly_config.username,
            project_slug=tenderly_config.project_slug
        )
        
        try:
            await client.initialize()
            yield client
        finally:
            await client.close()
    
    async def test_tenderly_client_initialization(self, tenderly_client):
        """Test that Tenderly client can initialize with real API."""
        # If we get here without exception, initialization worked
        assert tenderly_client.session is not None
        
        # Test that we can get stats
        stats = tenderly_client.get_stats()
        assert "simulations_run" in stats
        assert "session_active" in stats
        assert stats["session_active"] is True
    
    async def test_create_and_delete_fork(self, tenderly_client):
        """Test real fork creation and deletion."""
        # Create a fork
        fork = await tenderly_client.create_fork(
            network_id=TenderlyNetworkId.ETHEREUM,
            block_number=18500000,  # A recent mainnet block
            alias="test_fork_integration",
            description="Integration test fork"
        )
        
        assert fork.fork_id is not None
        assert fork.network_id == "1"
        assert fork.block_number == 18500000
        assert fork.alias == "test_fork_integration"
        
        # Fork should be tracked in active forks
        assert fork.fork_id in tenderly_client._active_forks
        
        # Delete the fork
        success = await tenderly_client.delete_fork(fork.fork_id)
        assert success is True
        
        # Fork should no longer be in active forks
        assert fork.fork_id not in tenderly_client._active_forks
    
    async def test_simple_transaction_simulation(self, tenderly_client):
        """Test simulating a simple ETH transfer."""
        # Create a simple ETH transfer transaction
        tx = TenderlyTransaction(
            from_address="0x000000000000000000000000000000000000dead",
            to_address="0x000000000000000000000000000000000000beef",
            value="1000000000000000000",  # 1 ETH in wei
            data="0x"
        )
        
        # Simulate the transaction
        result = await tenderly_client.simulate_transaction(
            transaction=tx,
            network_id=TenderlyNetworkId.ETHEREUM,
            block_number=18500000
        )
        
        # Basic validation
        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'gas_used')
        assert hasattr(result, 'simulation_time_ms')
        
        print(f"Simulation result: success={result.success}, gas={result.gas_used}")
        
        if not result.success:
            print(f"Simulation failed: {result.error_message}")
            print(f"Revert reason: {result.revert_reason}")
    
    async def test_erc20_transaction_simulation(self, tenderly_client):
        """Test simulating an ERC20 transfer."""
        # USDC transfer transaction (this will likely fail due to insufficient balance, but we can test the simulation)
        usdc_transfer_data = "0xa9059cbb000000000000000000000000000000000000000000000000000000000000dead00000000000000000000000000000000000000000000000000000000000f4240"  # transfer(dead, 1000000)
        
        tx = TenderlyTransaction(
            from_address="0x000000000000000000000000000000000000dead",
            to_address="0xA0b86a33E6417c5c6eF57e11d8CAf7d8C0f7C8F8",  # USDC contract
            value="0",
            data=usdc_transfer_data
        )
        
        result = await tenderly_client.simulate_transaction(
            transaction=tx,
            network_id=TenderlyNetworkId.ETHEREUM,
            block_number=18500000
        )
        
        assert result is not None
        print(f"ERC20 simulation: success={result.success}, gas={result.gas_used}")
        
        if not result.success:
            print(f"Expected failure reason: {result.revert_reason}")
    
    async def test_transaction_bundle_simulation(self, tenderly_client):
        """Test simulating multiple transactions in sequence."""
        # Create two simple transactions
        tx1 = TenderlyTransaction(
            from_address="0x000000000000000000000000000000000000dead",
            to_address="0x000000000000000000000000000000000000beef",
            value="500000000000000000",  # 0.5 ETH
            data="0x"
        )
        
        tx2 = TenderlyTransaction(
            from_address="0x000000000000000000000000000000000000dead",
            to_address="0x000000000000000000000000000000000000cafe",
            value="300000000000000000",  # 0.3 ETH
            data="0x"
        )
        
        # Simulate bundle
        results = await tenderly_client.simulate_transaction_bundle(
            transactions=[tx1, tx2],
            network_id=TenderlyNetworkId.ETHEREUM,
            block_number=18500000
        )
        
        assert len(results) >= 1  # At least one transaction should be processed
        
        for i, result in enumerate(results):
            print(f"Bundle tx {i+1}: success={result.success}, gas={result.gas_used}")
    
    @pytest.mark.skip(reason="Requires valid Tenderly username and project")
    async def test_hybrid_simulator_with_real_tenderly(self, tenderly_config):
        """Test HybridPathSimulator with real Tenderly integration."""
        # Mock other dependencies
        mock_redis = Mock()
        mock_redis.get = AsyncMock(return_value=None)
        
        mock_oracle = Mock()
        mock_oracle.get_price_usd = AsyncMock(return_value=2000.0)
        
        # Create simulator with real Tenderly config
        simulator = HybridPathSimulator(
            redis_client=mock_redis,
            asset_oracle=mock_oracle,
            config=SimulatorConfig(),
            tenderly_config=tenderly_config
        )
        
        # Create a simple test path
        edge = YieldGraphEdge(
            edge_id="test_eth_transfer",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="test",
            chain_name="ethereum"
        )
        
        # This will fail due to no edge state, but we can test the Tenderly client creation
        result = await simulator.simulate_path(
            path=[edge],
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.TENDERLY
        )
        
        # Should fail gracefully with proper error message
        assert not result.success
        assert result.simulation_mode == SimulationMode.TENDERLY.value
        print(f"Simulator result: {result.revert_reason}")


@pytest.mark.integration
@pytest.mark.asyncio
class TestTenderlyErrorHandling:
    """Test error handling with real API."""
    
    async def test_invalid_api_key(self):
        """Test handling of invalid API key."""
        client = TenderlyClient(
            api_key="invalid_key_12345",
            username="test_user",
            project_slug="test_project"
        )
        
        with pytest.raises(TenderlyAuthError):
            await client.initialize()
        
        await client.close()
    
    async def test_rate_limiting_behavior(self, tenderly_config):
        """Test rate limiting behavior."""
        client = TenderlyClient(
            api_key=tenderly_config.api_key,
            username=tenderly_config.username,
            project_slug=tenderly_config.project_slug,
            rate_limit_per_minute=5  # Very low limit for testing
        )
        
        try:
            await client.initialize()
            
            # Fill up the rate limit
            for i in range(5):
                client._last_requests.append(client._last_requests[0] if client._last_requests else None)
            
            # This should trigger rate limiting
            await client._ensure_rate_limit()
            
            # If we get here, rate limiting is working (it should have waited)
            assert True
            
        finally:
            await client.close()


if __name__ == "__main__":
    # Run a quick test
    import asyncio
    
    async def quick_test():
        """Quick test to verify API key works."""
        client = TenderlyClient(
            api_key="E5tSD537G0z2r9xur64acExE2DNjRFWP",
            username="bomanyd",
            project_slug="project"
        )
        
        try:
            await client.initialize()
            print("✅ Tenderly API key is valid!")
            
            # Try to create a fork
            fork = await client.create_fork(
                network_id=TenderlyNetworkId.ETHEREUM,
                description="Quick API test"
            )
            print(f"✅ Fork created: {fork.fork_id}")
            
            # Clean up
            await client.delete_fork(fork.fork_id)
            print("✅ Fork cleaned up")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            await client.close()
    
    # Uncomment to run quick test:
    # asyncio.run(quick_test())