"""Unit tests for BlockchainProvider with multicall integration."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.yield_arbitrage.blockchain_connector.provider import BlockchainProvider, ChainConfig
from src.yield_arbitrage.blockchain_connector.async_multicall import MulticallRequest, MulticallResult
from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider


class TestBlockchainProviderMulticall:
    """Test suite for BlockchainProvider with multicall functionality."""
    
    @pytest.fixture
    async def mock_blockchain_provider(self):
        """Create a mock blockchain provider with multicall support."""
        provider = BlockchainProvider()
        
        # Mock Web3 instances
        mock_w3 = AsyncMock(spec=AsyncWeb3)
        mock_w3.is_connected = AsyncMock(return_value=True)
        
        # Mock eth module
        mock_eth = AsyncMock()
        mock_eth.chain_id = AsyncMock(return_value=1)
        mock_eth.call = AsyncMock(return_value="0x0000000000000000000000000000000000000000000000000de0b6b3a7640000")
        mock_w3.eth = mock_eth
        
        # Mock chain configs
        provider.chain_configs = {
            "ethereum": ChainConfig(
                name="Ethereum",
                chain_id=1,
                rpc_url="https://mock-rpc-url.com",
                multicall_address="0x5BA1e12693Dc8F9c48aAD8770482f4739bEeD696"
            )
        }
        
        provider.web3_instances = {"ethereum": mock_w3}
        
        # Initialize multicall providers
        await provider._initialize_multicall_providers()
        provider._initialized = True
        
        return provider
    
    async def test_multicall_provider_initialization(self, mock_blockchain_provider):
        """Test that multicall providers are initialized correctly."""
        provider = mock_blockchain_provider
        
        assert "ethereum" in provider.multicall_providers
        assert provider.multicall_providers["ethereum"] is not None
        
        multicall_provider = await provider.get_multicall_provider("ethereum")
        assert multicall_provider is not None
        
        await provider.close()
    
    async def test_multicall_token_balances(self, mock_blockchain_provider):
        """Test multicall token balance retrieval."""
        provider = mock_blockchain_provider
        
        token_contracts = [
            "0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0123",  # USDC
            "0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0456",  # DAI
            "0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0789"   # WETH
        ]
        
        holder_address = "0x742d35Cc6634C0532925a3b8D2c5f6e2e1c0b93B"
        
        # Test token balance retrieval
        balance_results = await provider.multicall_token_balances(
            "ethereum", token_contracts, holder_address
        )
        
        assert len(balance_results) == 3
        assert all(token in balance_results for token in token_contracts)
        assert all(result.success for result in balance_results.values())
        assert all(result.function_name == "balanceOf" for result in balance_results.values())
        
        await provider.close()
    
    async def test_multicall_contract_data(self, mock_blockchain_provider):
        """Test multicall contract data retrieval."""
        provider = mock_blockchain_provider
        
        contract_calls = [
            MulticallRequest(
                target="0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0123",
                call_data="0x18160ddd",
                function_name="totalSupply_USDC"
            ),
            MulticallRequest(
                target="0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0456",
                call_data="0x95d89b41",
                function_name="symbol_DAI"
            ),
            MulticallRequest(
                target="0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0789",
                call_data="0x313ce567",
                function_name="decimals_WETH"
            )
        ]
        
        # Test contract data retrieval
        results = await provider.multicall_contract_data("ethereum", contract_calls)
        
        assert len(results) == 3
        assert all(isinstance(result, MulticallResult) for result in results)
        assert all(result.success for result in results)
        
        function_names = [result.function_name for result in results]
        assert "totalSupply_USDC" in function_names
        assert "symbol_DAI" in function_names
        assert "decimals_WETH" in function_names
        
        await provider.close()
    
    async def test_get_defi_protocol_data(self, mock_blockchain_provider):
        """Test DeFi protocol data retrieval."""
        provider = mock_blockchain_provider
        
        protocol_contracts = {
            "AAVE": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
            "COMP": "0xc00e94Cb662C3520282E6f5717214004A7f26888",
            "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"
        }
        
        # Test DeFi protocol data retrieval
        data_results = await provider.get_defi_protocol_data("ethereum", protocol_contracts)
        
        # Should have 3 calls per protocol (totalSupply, symbol, decimals)
        expected_functions = [
            "AAVE_totalSupply", "AAVE_symbol", "AAVE_decimals",
            "COMP_totalSupply", "COMP_symbol", "COMP_decimals", 
            "UNI_totalSupply", "UNI_symbol", "UNI_decimals"
        ]
        
        assert len(data_results) == 9
        for expected_function in expected_functions:
            assert expected_function in data_results
            assert data_results[expected_function].success
        
        await provider.close()
    
    async def test_multicall_error_handling(self, mock_blockchain_provider):
        """Test error handling when multicall is not available."""
        provider = mock_blockchain_provider
        
        # Test with non-existent chain
        with pytest.raises(ValueError, match="Multicall not available for nonexistent"):
            await provider.multicall_token_balances(
                "nonexistent", 
                ["0x123"], 
                "0x456"
            )
        
        await provider.close()
    
    async def test_multicall_with_empty_calls(self, mock_blockchain_provider):
        """Test multicall with empty call list."""
        provider = mock_blockchain_provider
        
        # Test with empty calls
        results = await provider.multicall_contract_data("ethereum", [])
        assert results == []
        
        await provider.close()
    
    async def test_performance_benefit_demonstration(self, mock_blockchain_provider):
        """Demonstrate the performance benefit of multicall vs individual calls."""
        provider = mock_blockchain_provider
        
        # Create a large number of token contracts to test
        token_contracts = [f"0x{'1' * 39}{i}" for i in range(20)]
        holder_address = "0x742d35Cc6634C0532925a3b8D2c5f6e2e1c0b93B"
        
        # Test multicall approach
        import time
        start_time = time.time()
        
        balance_results = await provider.multicall_token_balances(
            "ethereum", token_contracts, holder_address
        )
        
        multicall_time = time.time() - start_time
        
        assert len(balance_results) == 20
        assert all(result.success for result in balance_results.values())
        
        print(f"Multicall execution time for 20 token balances: {multicall_time:.4f}s")
        
        # The performance benefit would be more obvious with real network calls
        # In this test, we're mainly verifying the interface works correctly
        
        await provider.close()


class TestMulticallIntegrationWithRealChains:
    """Integration tests that could be run with real chain data (when RPC URLs are available)."""
    
    @pytest.mark.skip(reason="Requires real RPC endpoints")
    async def test_real_ethereum_multicall(self):
        """Test with real Ethereum RPC (skip by default)."""
        provider = BlockchainProvider()
        
        # This would require actual RPC URLs in environment variables
        # await provider.initialize()
        
        # Real token contracts (USDC, DAI, WETH)
        token_contracts = [
            "0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0123",
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        ]
        
        # Vitalik's address (public)
        holder_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
        
        # balance_results = await provider.multicall_token_balances(
        #     "ethereum", token_contracts, holder_address
        # )
        
        # await provider.close()


# Manual test execution
if __name__ == "__main__":
    async def run_blockchain_multicall_tests():
        """Run blockchain multicall tests manually."""
        print("🧪 Testing BlockchainProvider with Multicall Integration")
        print("=" * 60)
        
        # Create a mock provider
        provider = BlockchainProvider()
        
        # Mock Web3 instances
        mock_w3 = AsyncMock(spec=AsyncWeb3)
        mock_w3.is_connected = AsyncMock(return_value=True)
        
        # Mock eth module
        mock_eth = AsyncMock()
        mock_eth.chain_id = AsyncMock(return_value=1)
        mock_eth.call = AsyncMock(return_value="0x0000000000000000000000000000000000000000000000000de0b6b3a7640000")
        mock_w3.eth = mock_eth
        
        # Mock chain configs
        provider.chain_configs = {
            "ethereum": ChainConfig(
                name="Ethereum",
                chain_id=1,
                rpc_url="https://mock-rpc-url.com",
                multicall_address="0x5BA1e12693Dc8F9c48aAD8770482f4739bEeD696"
            )
        }
        
        provider.web3_instances = {"ethereum": mock_w3}
        
        # Initialize multicall providers
        await provider._initialize_multicall_providers()
        provider._initialized = True
        
        print("✅ Initialized mock blockchain provider with multicall support")
        
        # Test token balances
        token_contracts = [
            "0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0123",  # USDC
            "0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0456",  # DAI
            "0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0789"   # WETH
        ]
        
        holder_address = "0x742d35Cc6634C0532925a3b8D2c5f6e2e1c0b93B"
        
        balance_results = await provider.multicall_token_balances(
            "ethereum", token_contracts, holder_address
        )
        
        print(f"✅ Retrieved balances for {len(balance_results)} tokens:")
        for token, result in balance_results.items():
            print(f"  {token[:10]}... -> Success: {result.success}, Result: {result.result[:20]}...")
        
        # Test DeFi protocol data
        protocol_contracts = {
            "AAVE": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
            "COMP": "0xc00e94Cb662C3520282E6f5717214004A7f26888"
        }
        
        data_results = await provider.get_defi_protocol_data("ethereum", protocol_contracts)
        
        print(f"✅ Retrieved data for {len(data_results)} protocol functions:")
        for function_name, result in data_results.items():
            print(f"  {function_name} -> Success: {result.success}")
        
        await provider.close()
        
        print("\n🎉 All blockchain multicall integration tests completed!")
    
    # Run if executed directly
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        asyncio.run(run_blockchain_multicall_tests())