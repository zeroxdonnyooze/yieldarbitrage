"""Unit tests for the AsyncMulticallProvider implementation."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.yield_arbitrage.blockchain_connector.async_multicall import (
    AsyncMulticallProvider,
    MulticallRequest,
    MulticallResult,
    MulticallBenchmark
)
from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider


class TestAsyncMulticallProvider:
    """Test suite for AsyncMulticallProvider."""
    
    @pytest.fixture
    async def mock_web3(self):
        """Create a mock AsyncWeb3 instance."""
        provider = AsyncMock(spec=AsyncHTTPProvider)
        w3 = AsyncWeb3(provider)
        
        # Mock eth.call method
        w3.eth.call = AsyncMock(return_value="0x0000000000000000000000000000000000000000000000000de0b6b3a7640000")
        
        return w3
    
    @pytest.fixture
    def sample_multicall_requests(self):
        """Sample multicall requests for testing."""
        return [
            MulticallRequest(
                target="0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0123",
                call_data="0x70a08231000000000000000000000000742d35Cc6634C0532925a3b8D2c5f6e2e1c0b93B",
                function_name="balanceOf_USDC"
            ),
            MulticallRequest(
                target="0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0456",
                call_data="0x18160ddd",
                function_name="totalSupply_DAI"
            ),
            MulticallRequest(
                target="0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0789",
                call_data="0x95d89b41",
                function_name="symbol_WETH"
            )
        ]
    
    async def test_initialization(self, mock_web3):
        """Test AsyncMulticallProvider initialization."""
        provider = AsyncMulticallProvider(
            w3=mock_web3,
            multicall_address="0x5BA1e12693Dc8F9c48aAD8770482f4739bEeD696",
            max_batch_size=100
        )
        
        assert provider.w3 == mock_web3
        assert provider.multicall_address == "0x5BA1e12693Dc8F9c48aAD8770482f4739bEeD696"
        assert provider.max_batch_size == 100
        assert hasattr(provider, '_thread_pool')
        
        await provider.close()
    
    async def test_execute_individual_calls_fallback(self, mock_web3, sample_multicall_requests):
        """Test fallback to individual calls."""
        provider = AsyncMulticallProvider(
            w3=mock_web3,
            multicall_address=None,  # No multicall address
            use_multicall_py=False   # Disable multicall.py
        )
        
        results = await provider.execute_calls(sample_multicall_requests)
        
        assert len(results) == 3
        assert all(isinstance(result, MulticallResult) for result in results)
        assert all(result.success for result in results)
        assert all(result.execution_time is not None for result in results)
        
        # Verify call details
        assert results[0].function_name == "balanceOf_USDC"
        assert results[1].function_name == "totalSupply_DAI"
        assert results[2].function_name == "symbol_WETH"
        
        await provider.close()
    
    async def test_execute_calls_with_custom_multicall(self, mock_web3, sample_multicall_requests):
        """Test execution with custom multicall contract."""
        provider = AsyncMulticallProvider(
            w3=mock_web3,
            multicall_address="0x5BA1e12693Dc8F9c48aAD8770482f4739bEeD696",
            use_multicall_py=False
        )
        
        results = await provider.execute_calls(sample_multicall_requests)
        
        assert len(results) == 3
        assert all(result.success for result in results)
        assert all(result.target == req.target for result, req in zip(results, sample_multicall_requests))
        
        await provider.close()
    
    @pytest.mark.skipif(True, reason="multicall.py might not be available in test environment")
    async def test_execute_calls_with_multicall_py(self, mock_web3, sample_multicall_requests):
        """Test execution with multicall.py if available."""
        provider = AsyncMulticallProvider(
            w3=mock_web3,
            multicall_address="0x5BA1e12693Dc8F9c48aAD8770482f4739bEeD696",
            use_multicall_py=True
        )
        
        results = await provider.execute_calls(sample_multicall_requests)
        
        assert len(results) == 3
        assert all(isinstance(result, MulticallResult) for result in results)
        
        await provider.close()
    
    async def test_get_token_balances_convenience_method(self, mock_web3):
        """Test the convenience method for getting token balances."""
        provider = AsyncMulticallProvider(w3=mock_web3, multicall_address=None)
        
        token_contracts = [
            "0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0123",  # USDC
            "0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0456",  # DAI
            "0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0789"   # WETH
        ]
        
        holder_address = "0x742d35Cc6634C0532925a3b8D2c5f6e2e1c0b93B"
        
        balance_results = await provider.get_token_balances(token_contracts, holder_address)
        
        assert len(balance_results) == 3
        assert all(token in balance_results for token in token_contracts)
        assert all(result.function_name == "balanceOf" for result in balance_results.values())
        assert all(result.success for result in balance_results.values())
        
        await provider.close()
    
    async def test_get_multiple_contract_data_convenience_method(self, mock_web3):
        """Test the convenience method for getting multiple contract data."""
        provider = AsyncMulticallProvider(w3=mock_web3, multicall_address=None)
        
        contract_calls = [
            ("0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0123", "0x18160ddd", "totalSupply_USDC"),
            ("0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0456", "0x95d89b41", "symbol_DAI"),
            ("0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0789", "0x313ce567", "decimals_WETH")
        ]
        
        data_results = await provider.get_multiple_contract_data(contract_calls)
        
        assert len(data_results) == 3
        assert "totalSupply_USDC" in data_results
        assert "symbol_DAI" in data_results
        assert "decimals_WETH" in data_results
        assert all(result.success for result in data_results.values())
        
        await provider.close()
    
    async def test_empty_calls_list(self, mock_web3):
        """Test handling of empty calls list."""
        provider = AsyncMulticallProvider(w3=mock_web3)
        
        results = await provider.execute_calls([])
        
        assert results == []
        
        await provider.close()
    
    async def test_error_handling_in_individual_calls(self, mock_web3, sample_multicall_requests):
        """Test error handling when individual calls fail."""
        # Mock eth.call to raise an exception for the second call
        call_count = 0
        
        async def mock_eth_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Contract execution reverted")
            return "0x0000000000000000000000000000000000000000000000000de0b6b3a7640000"
        
        mock_web3.eth.call = mock_eth_call
        
        provider = AsyncMulticallProvider(w3=mock_web3, multicall_address=None)
        
        results = await provider.execute_calls(sample_multicall_requests)
        
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error == "Contract execution reverted"
        assert results[2].success is True
        
        await provider.close()
    
    async def test_large_batch_splitting(self, mock_web3):
        """Test that large batches are split appropriately."""
        provider = AsyncMulticallProvider(w3=mock_web3, max_batch_size=2)
        
        # Create 5 calls (should be split into 3 batches: 2, 2, 1)
        large_call_list = [
            MulticallRequest(
                target=f"0x{'1' * 40}",
                call_data=f"0x{'a' * 8}",
                function_name=f"function_{i}"
            )
            for i in range(5)
        ]
        
        results = await provider.execute_calls(large_call_list)
        
        assert len(results) == 5
        assert all(result.success for result in results)
        
        await provider.close()


class TestMulticallBenchmark:
    """Test suite for MulticallBenchmark."""
    
    @pytest.fixture
    async def mock_web3(self):
        """Create a mock AsyncWeb3 instance."""
        provider = AsyncMock(spec=AsyncHTTPProvider)
        w3 = AsyncWeb3(provider)
        w3.eth.call = AsyncMock(return_value="0x0000000000000000000000000000000000000000000000000de0b6b3a7640000")
        return w3
    
    async def test_benchmark_approaches(self, mock_web3):
        """Test the benchmark functionality."""
        benchmark = MulticallBenchmark(mock_web3)
        
        sample_calls = [
            MulticallRequest(
                target=f"0x{'1' * 40}",
                call_data=f"0x{'a' * 8}",
                function_name=f"function_{i}"
            )
            for i in range(5)
        ]
        
        results = await benchmark.benchmark_approaches(sample_calls, iterations=2)
        
        assert "individual_concurrent" in results
        assert "multicall_contract" in results
        assert "sequential_individual" in results
        
        # Sequential should be slowest
        assert results["sequential_individual"] > results["multicall_contract"]
        assert results["sequential_individual"] > results["individual_concurrent"]
        
        # All times should be positive
        assert all(time > 0 for time in results.values())
        
        print("\n📊 Benchmark Results:")
        for approach, time_taken in sorted(results.items(), key=lambda x: x[1]):
            print(f"{approach:25} | {time_taken:.4f}s")


# Manual test execution
if __name__ == "__main__":
    async def run_multicall_provider_tests():
        """Run the multicall provider tests manually."""
        from web3 import AsyncWeb3
        from web3.providers import AsyncHTTPProvider
        from unittest.mock import AsyncMock
        
        print("🧪 Testing AsyncMulticallProvider")
        print("=" * 40)
        
        # Create mock Web3
        provider = AsyncMock(spec=AsyncHTTPProvider)
        w3 = AsyncWeb3(provider)
        w3.eth.call = AsyncMock(return_value="0x0000000000000000000000000000000000000000000000000de0b6b3a7640000")
        
        # Test basic functionality
        multicall_provider = AsyncMulticallProvider(
            w3=w3,
            multicall_address="0x5BA1e12693Dc8F9c48aAD8770482f4739bEeD696"
        )
        
        # Test token balances
        token_contracts = [
            "0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0123",  # USDC
            "0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0456",  # DAI
            "0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0789"   # WETH
        ]
        
        holder_address = "0x742d35Cc6634C0532925a3b8D2c5f6e2e1c0b93B"
        
        balance_results = await multicall_provider.get_token_balances(token_contracts, holder_address)
        
        print(f"✅ Retrieved balances for {len(balance_results)} tokens")
        for token, result in balance_results.items():
            print(f"  {token[:10]}... -> {result.success} ({result.result[:20]}...)")
        
        # Test benchmark
        benchmark = MulticallBenchmark(w3)
        sample_calls = [
            MulticallRequest(
                target=f"0x{'1' * 40}",
                call_data=f"0x{'a' * 8}",
                function_name=f"function_{i}"
            )
            for i in range(10)
        ]
        
        benchmark_results = await benchmark.benchmark_approaches(sample_calls, iterations=2)
        
        print("\n📊 Benchmark Results:")
        for approach, time_taken in sorted(benchmark_results.items(), key=lambda x: x[1]):
            efficiency = benchmark_results["sequential_individual"] / time_taken
            print(f"{approach:25} | {time_taken:.4f}s | {efficiency:.2f}x faster")
        
        await multicall_provider.close()
        
        print("\n🎉 All tests completed successfully!")
    
    # Run if executed directly
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        asyncio.run(run_multicall_provider_tests())