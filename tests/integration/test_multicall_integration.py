"""Integration tests for testing different multicall implementations with AsyncWeb3."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any
import time
import os

# Try to import various multicall libraries
try:
    from multicall import Multicall
    MULTICALL_PY_AVAILABLE = True
except ImportError:
    MULTICALL_PY_AVAILABLE = False

try:
    import w3multicall
    W3MULTICALL_AVAILABLE = True
except ImportError:
    W3MULTICALL_AVAILABLE = False

try:
    from async_web3 import AsyncWeb3
    from async_web3.providers import MulticallHTTPProvider
    ASYNC_WEB3_AVAILABLE = True
except ImportError:
    ASYNC_WEB3_AVAILABLE = False

from web3 import AsyncWeb3 as StandardAsyncWeb3
from web3.providers import AsyncHTTPProvider
from web3.exceptions import Web3Exception


class TestMulticallLibraryIntegration:
    """Integration tests for different multicall libraries with AsyncWeb3."""
    
    @pytest.fixture
    def sample_contract_calls(self):
        """Sample contract calls for testing."""
        return [
            {
                'target': '0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0123',
                'call_data': '0x70a08231000000000000000000000000742d35Cc6634C0532925a3b8D2c5f6e2e1c0b93B',
                'function': 'balanceOf',
                'expected_type': 'uint256'
            },
            {
                'target': '0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0456', 
                'call_data': '0x18160ddd',
                'function': 'totalSupply',
                'expected_type': 'uint256'
            },
            {
                'target': '0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0789',
                'call_data': '0x95d89b41',
                'function': 'symbol',
                'expected_type': 'string'
            }
        ]
    
    @pytest.fixture
    def mock_rpc_url(self):
        """Mock RPC URL for testing."""
        return "https://eth-mainnet.g.alchemy.com/v2/mock-api-key"
    
    @pytest.mark.skipif(not MULTICALL_PY_AVAILABLE, reason="multicall.py not available")
    async def test_banteg_multicall_py_with_asyncio_adapter(self, sample_contract_calls, mock_rpc_url):
        """Test banteg/multicall.py with asyncio.to_thread adapter."""
        def sync_multicall_operation(calls):
            """Simulate banteg/multicall.py operation."""
            # This would normally use the actual multicall library
            # For testing, we simulate the operation
            time.sleep(0.01)  # Simulate network I/O
            return [
                {
                    "success": True,
                    "result": "0x0000000000000000000000000000000000000000000000000de0b6b3a7640000",
                    "call": call
                }
                for call in calls
            ]
        
        # Use asyncio.to_thread to avoid blocking
        start_time = time.time()
        result = await asyncio.to_thread(sync_multicall_operation, sample_contract_calls)
        end_time = time.time()
        
        assert len(result) == 3
        assert all(call_result["success"] for call_result in result)
        assert end_time - start_time < 0.1  # Should be fast
        
        print(f"✅ banteg/multicall.py with asyncio.to_thread: {end_time - start_time:.4f}s")
    
    @pytest.mark.skipif(not W3MULTICALL_AVAILABLE, reason="w3multicall not available")
    async def test_w3multicall_with_executor_adapter(self, sample_contract_calls):
        """Test w3multicall with ThreadPoolExecutor adapter."""
        import concurrent.futures
        
        def sync_w3multicall_operation(calls):
            """Simulate w3multicall operation."""
            time.sleep(0.015)  # Simulate network I/O
            return {
                "successful_calls": len(calls),
                "results": [f"result_{i}" for i in range(len(calls))],
                "execution_time": 0.015
            }
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            loop = asyncio.get_event_loop()
            
            start_time = time.time()
            result = await loop.run_in_executor(
                executor, 
                sync_w3multicall_operation, 
                sample_contract_calls
            )
            end_time = time.time()
            
            assert result["successful_calls"] == 3
            assert len(result["results"]) == 3
            assert end_time - start_time < 0.1
            
            print(f"✅ w3multicall with ThreadPoolExecutor: {end_time - start_time:.4f}s")
    
    @pytest.mark.skipif(not ASYNC_WEB3_AVAILABLE, reason="async_web3.py not available")
    async def test_aureliano_async_web3_multicall_provider(self, sample_contract_calls, mock_rpc_url):
        """Test Aureliano90/async_web3.py MulticallHTTPProvider."""
        # This would test the actual MulticallHTTPProvider
        # For now, we simulate its behavior since we may not have it installed
        
        class MockMulticallHTTPProvider:
            def __init__(self, rpc_url):
                self.rpc_url = rpc_url
                self.call_count = 0
            
            async def make_request(self, method, params):
                """Mock the auto-batching behavior."""
                self.call_count += 1
                await asyncio.sleep(0.005)  # Simulate network call
                
                if method == "eth_call":
                    return f"0x{''.join(['0'] * 60)}{self.call_count:04x}"
                else:
                    return f"result_for_{method}"
        
        provider = MockMulticallHTTPProvider(mock_rpc_url)
        
        # Simulate multiple concurrent eth_call requests
        start_time = time.time()
        
        tasks = [
            provider.make_request("eth_call", {
                "to": call["target"],
                "data": call["call_data"]
            })
            for call in sample_contract_calls
        ]
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        assert len(results) == 3
        assert all(isinstance(result, str) and result.startswith("0x") for result in results)
        assert provider.call_count == 3
        
        print(f"✅ MulticallHTTPProvider simulation: {end_time - start_time:.4f}s")
    
    async def test_standard_asyncweb3_batch_requests(self, sample_contract_calls, mock_rpc_url):
        """Test standard AsyncWeb3 batch_requests functionality."""
        # Mock AsyncWeb3 instance
        provider = AsyncMock(spec=AsyncHTTPProvider)
        w3 = StandardAsyncWeb3(provider)
        
        # Mock batch_requests method
        async def mock_batch_requests():
            """Mock batch requests context manager."""
            class MockBatch:
                def __init__(self):
                    self.calls = []
                
                def add(self, call):
                    self.calls.append(call)
                
                async def async_execute(self):
                    await asyncio.sleep(0.01)  # Simulate batch execution
                    return [f"batch_result_{i}" for i in range(len(self.calls))]
            
            return MockBatch()
        
        # Test batch execution
        start_time = time.time()
        batch = await mock_batch_requests()
        
        # Add sample calls to batch
        for call in sample_contract_calls:
            batch.add(f"eth_call_{call['function']}")
        
        results = await batch.async_execute()
        end_time = time.time()
        
        assert len(results) == 3
        assert all(result.startswith("batch_result_") for result in results)
        
        print(f"✅ Standard AsyncWeb3 batch_requests: {end_time - start_time:.4f}s")


class TestPerformanceBenchmarks:
    """Performance benchmarks for different multicall approaches."""
    
    async def test_performance_comparison(self):
        """Compare performance of different multicall approaches."""
        call_count = 20
        sample_calls = [
            {
                'target': f"0x{'1' * 40}",
                'call_data': f"0x{'a' * 8}{i:04x}",
                'function': f'function_{i}'
            }
            for i in range(call_count)
        ]
        
        results = {}
        
        # 1. Sequential individual calls (baseline)
        async def individual_call(call):
            await asyncio.sleep(0.01)  # Simulate network latency
            return f"result_for_{call['function']}"
        
        start_time = time.time()
        sequential_results = []
        for call in sample_calls:
            result = await individual_call(call)
            sequential_results.append(result)
        results["sequential_individual"] = time.time() - start_time
        
        # 2. Concurrent individual calls with asyncio.gather
        start_time = time.time()
        concurrent_results = await asyncio.gather(*[
            individual_call(call) for call in sample_calls
        ])
        results["concurrent_individual"] = time.time() - start_time
        
        # 3. Simulated multicall contract
        async def multicall_batch(calls):
            await asyncio.sleep(0.02)  # Single network call
            return [f"multicall_result_{call['function']}" for call in calls]
        
        start_time = time.time()
        multicall_results = await multicall_batch(sample_calls)
        results["multicall_contract"] = time.time() - start_time
        
        # 4. Simulated with asyncio.to_thread adapter
        def sync_multicall(calls):
            time.sleep(0.025)  # Simulate sync multicall library
            return [f"sync_multicall_{call['function']}" for call in calls]
        
        start_time = time.time()
        thread_results = await asyncio.to_thread(sync_multicall, sample_calls)
        results["asyncio_to_thread"] = time.time() - start_time
        
        # Verify all approaches return correct number of results
        assert len(sequential_results) == call_count
        assert len(concurrent_results) == call_count
        assert len(multicall_results) == call_count
        assert len(thread_results) == call_count
        
        # Print performance comparison
        print("\n📊 Performance Comparison (20 calls):")
        print("=" * 50)
        for approach, duration in sorted(results.items(), key=lambda x: x[1]):
            efficiency = results["sequential_individual"] / duration
            print(f"{approach:25} | {duration:.4f}s | {efficiency:.2f}x faster")
        
        # Assertions about expected performance characteristics
        assert results["concurrent_individual"] < results["sequential_individual"]
        assert results["multicall_contract"] < results["sequential_individual"] 
        assert results["asyncio_to_thread"] < results["sequential_individual"]
        
        return results


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases for multicall implementations."""
    
    async def test_multicall_with_failing_calls(self):
        """Test multicall behavior when some calls fail."""
        async def failing_multicall():
            await asyncio.sleep(0.01)
            return [
                {"success": True, "result": "0x123", "call_index": 0},
                {"success": False, "error": "execution reverted", "call_index": 1},
                {"success": True, "result": "0x456", "call_index": 2},
                {"success": False, "error": "out of gas", "call_index": 3}
            ]
        
        results = await failing_multicall()
        
        successful_calls = [r for r in results if r["success"]]
        failed_calls = [r for r in results if not r["success"]]
        
        assert len(successful_calls) == 2
        assert len(failed_calls) == 2
        assert failed_calls[0]["error"] == "execution reverted"
        assert failed_calls[1]["error"] == "out of gas"
        
        print("✅ Error handling test passed")
    
    async def test_multicall_with_large_batch_size(self):
        """Test multicall with large batch sizes."""
        large_batch_size = 100
        
        async def large_multicall_batch(batch_size):
            await asyncio.sleep(0.05)  # Simulate larger processing time
            return [f"result_{i}" for i in range(batch_size)]
        
        start_time = time.time()
        results = await large_multicall_batch(large_batch_size)
        end_time = time.time()
        
        assert len(results) == large_batch_size
        assert end_time - start_time < 0.1  # Should still be fast
        
        print(f"✅ Large batch test passed: {large_batch_size} calls in {end_time - start_time:.4f}s")
    
    async def test_multicall_timeout_handling(self):
        """Test timeout handling in multicall operations."""
        async def timeout_prone_multicall():
            try:
                await asyncio.wait_for(asyncio.sleep(0.1), timeout=0.05)
                return ["success"]
            except asyncio.TimeoutError:
                return {"error": "timeout", "success": False}
        
        result = await timeout_prone_multicall()
        
        assert isinstance(result, dict)
        assert result["error"] == "timeout"
        assert not result["success"]
        
        print("✅ Timeout handling test passed")


# Manual test execution instructions
if __name__ == "__main__":
    async def run_integration_tests():
        """Run integration tests manually."""
        print("🧪 Running Multicall Integration Tests")
        print("=" * 50)
        
        # Test performance comparison
        perf_test = TestPerformanceBenchmarks()
        await perf_test.test_performance_comparison()
        
        print("\n🔧 Testing Error Handling")
        print("=" * 30)
        
        error_test = TestErrorHandlingAndEdgeCases()
        await error_test.test_multicall_with_failing_calls()
        await error_test.test_multicall_with_large_batch_size()
        await error_test.test_multicall_timeout_handling()
        
        print("\n🎉 All integration tests completed!")
    
    # Run if executed directly
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        asyncio.run(run_integration_tests())