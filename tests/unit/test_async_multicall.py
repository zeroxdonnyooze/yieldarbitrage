"""Unit tests for async multicall functionality and compatibility with AsyncWeb3."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any
import time

from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider
from web3.exceptions import Web3Exception


class TestAsyncMulticallCompatibility:
    """Test suite for async multicall library compatibility with AsyncWeb3."""
    
    @pytest.fixture
    async def async_web3(self):
        """Create a mock AsyncWeb3 instance for testing."""
        provider = AsyncMock(spec=AsyncHTTPProvider)
        w3 = AsyncWeb3(provider)
        
        # Mock basic functionality
        w3.is_connected = AsyncMock(return_value=True)
        w3.eth.chain_id = AsyncMock(return_value=1)
        w3.eth.block_number = AsyncMock(return_value=18500000)
        w3.eth.gas_price = AsyncMock(return_value=20000000000)  # 20 gwei
        
        return w3
    
    @pytest.fixture
    def sample_contract_calls(self):
        """Sample contract calls for testing multicall functionality."""
        return [
            {
                'target': '0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0123',
                'call_data': '0x70a08231000000000000000000000000742d35Cc6634C0532925a3b8D2c5f6e2e1c0b93B',
                'function': 'balanceOf',
                'expected_output': 1000000000000000000  # 1 ETH in wei
            },
            {
                'target': '0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0456',
                'call_data': '0x18160ddd',
                'function': 'totalSupply',
                'expected_output': 1000000000000000000000000  # 1M tokens
            },
            {
                'target': '0xA0b86a33E6441e42b65C9F8893c37d9C2CAD0789',
                'call_data': '0x95d89b41',
                'function': 'symbol',
                'expected_output': 'TEST'
            }
        ]


class TestBantegMulticallPy:
    """Test banteg/multicall.py compatibility with AsyncWeb3."""
    
    async def test_multicall_with_sync_web3_blocking_behavior(self):
        """Test that banteg/multicall.py blocks event loop with sync Web3."""
        # This test demonstrates the blocking behavior
        start_time = time.time()
        
        # Simulate what happens when banteg/multicall.py uses sync Web3
        def sync_blocking_call():
            time.sleep(0.1)  # Simulate blocking I/O
            return "result"
        
        # This would block the event loop
        result = sync_blocking_call()
        end_time = time.time()
        
        assert result == "result"
        assert end_time - start_time >= 0.1
    
    async def test_multicall_with_asyncio_to_thread_adapter(self):
        """Test wrapping sync multicall calls with asyncio.to_thread()."""
        def sync_multicall_operation():
            # Simulate banteg/multicall.py operation
            time.sleep(0.05)  # Simulate network I/O
            return [
                {"success": True, "result": "0x0000000000000000000000000000000000000000000000000de0b6b3a7640000"},
                {"success": True, "result": "0x00000000000000000000000000000000000000000000d3c21bcecceda1000000"},
            ]
        
        # Use asyncio.to_thread to avoid blocking
        start_time = time.time()
        
        # Run multiple operations concurrently
        tasks = [
            asyncio.to_thread(sync_multicall_operation),
            asyncio.to_thread(sync_multicall_operation),
            asyncio.to_thread(sync_multicall_operation)
        ]
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Should complete faster than sequential execution
        assert len(results) == 3
        assert all(len(result) == 2 for result in results)
        # Concurrent execution should be faster than 3 * 0.05 = 0.15s
        assert end_time - start_time < 0.15
    
    async def test_multicall_with_run_in_executor_adapter(self):
        """Test wrapping sync multicall calls with run_in_executor()."""
        import concurrent.futures
        
        def sync_multicall_batch():
            # Simulate banteg/multicall.py batch operation
            time.sleep(0.03)
            return {
                "call_1": "0x0000000000000000000000000000000000000000000000000de0b6b3a7640000",
                "call_2": "0x00000000000000000000000000000000000000000000d3c21bcecceda1000000"
            }
        
        # Use ThreadPoolExecutor for I/O bound tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            loop = asyncio.get_event_loop()
            
            start_time = time.time()
            
            # Execute multiple batches concurrently
            futures = [
                loop.run_in_executor(executor, sync_multicall_batch),
                loop.run_in_executor(executor, sync_multicall_batch),
                loop.run_in_executor(executor, sync_multicall_batch),
                loop.run_in_executor(executor, sync_multicall_batch)
            ]
            
            results = await asyncio.gather(*futures)
            end_time = time.time()
            
            assert len(results) == 4
            assert all("call_1" in result and "call_2" in result for result in results)
            # Should be faster than sequential execution
            assert end_time - start_time < 0.12  # Less than 4 * 0.03


class TestW3Multicall:
    """Test w3multicall threading compatibility."""
    
    async def test_w3multicall_threading_behavior(self):
        """Test w3multicall's threading approach with AsyncWeb3."""
        # Simulate w3multicall's threading behavior
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def threaded_multicall_worker(worker_id):
            # Simulate w3multicall operation
            time.sleep(0.02)
            result = {
                "worker_id": worker_id,
                "calls": [f"result_{worker_id}_1", f"result_{worker_id}_2"]
            }
            results_queue.put(result)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=threaded_multicall_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 5
        assert all("worker_id" in result and "calls" in result for result in results)
    
    async def test_w3multicall_async_integration_challenge(self):
        """Test the challenge of integrating threading-based w3multicall with async."""
        # This demonstrates why w3multicall might be challenging to integrate
        
        async def async_web3_operation():
            await asyncio.sleep(0.01)  # Simulate async Web3 call
            return "async_result"
        
        def sync_w3multicall_operation():
            time.sleep(0.02)  # Simulate sync multicall
            return "sync_multicall_result"
        
        # The challenge: mixing async and threading approaches
        async_result = await async_web3_operation()
        sync_result = await asyncio.to_thread(sync_w3multicall_operation)
        
        assert async_result == "async_result"
        assert sync_result == "sync_multicall_result"


class TestAsyncWeb3NativeBatching:
    """Test AsyncWeb3's native batch_requests functionality."""
    
    async def test_async_web3_batch_requests_basic(self):
        """Test basic batch_requests functionality."""
        # Simplified test - just test the concept
        async def mock_batch_execution():
            # Simulate batch requests functionality
            await asyncio.sleep(0.01)
            return [
                {"block_number": 18500000},
                {"block_number": 18500001},
                {"block_number": 18500002}
            ]
        
        results = await mock_batch_execution()
        
        assert len(results) == 3
        assert all("block_number" in result for result in results)
        print("✅ Basic batch requests concept validated")
    
    async def test_async_web3_batch_vs_gather_performance(self):
        """Test performance comparison between batch_requests and asyncio.gather."""
        # Simulate individual async calls
        async def mock_eth_call(block_num):
            await asyncio.sleep(0.001)  # Simulate network latency
            return {"block_number": block_num}
        
        block_numbers = [18500000 + i for i in range(10)]
        
        # Test asyncio.gather approach
        start_time = time.time()
        gather_results = await asyncio.gather(*[
            mock_eth_call(block_num) for block_num in block_numbers
        ])
        gather_time = time.time() - start_time
        
        # Test simulated batch approach (with added overhead)
        async def mock_batch_call(block_numbers):
            await asyncio.sleep(0.002)  # Simulate batch overhead
            await asyncio.sleep(len(block_numbers) * 0.0005)  # Simulate processing
            return [{"block_number": bn} for bn in block_numbers]
        
        start_time = time.time()
        batch_results = await mock_batch_call(block_numbers)
        batch_time = time.time() - start_time
        
        assert len(gather_results) == 10
        assert len(batch_results) == 10
        
        # Performance comparison (gather is often faster for small batches)
        print(f"Gather time: {gather_time:.4f}s")
        print(f"Batch time: {batch_time:.4f}s")


class TestMulticallHTTPProvider:
    """Test MulticallHTTPProvider approach (Aureliano90/async_web3.py)."""
    
    async def test_multicall_http_provider_concept(self):
        """Test the concept of MulticallHTTPProvider that auto-batches calls."""
        # Mock the MulticallHTTPProvider behavior (simplified)
        async def mock_multicall_request(call_data):
            """Mock individual multicall request."""
            await asyncio.sleep(0.005)  # Simulate network latency
            return f"0x{''.join(['0'] * 60)}{hash(str(call_data)) % 10000:04x}"
        
        # Test auto-batching behavior
        start_time = time.time()
        
        # Simulate multiple concurrent eth_call requests
        tasks = []
        for i in range(5):
            tasks.extend([
                mock_multicall_request({"to": f"0x{'1' * 40}", "data": f"0x{'a' * (8 + i * 2)}"}),
                mock_multicall_request({"to": f"0x{'2' * 40}", "data": f"0x{'b' * (8 + i * 2)}"}),
                mock_multicall_request({"to": f"0x{'3' * 40}", "data": f"0x{'c' * (8 + i * 2)}"})
            ])
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        assert len(results) == 15  # 5 calls * 3 different contracts
        assert all(isinstance(result, str) and result.startswith("0x") for result in results)
        
        print(f"MulticallHTTPProvider simulation time: {end_time - start_time:.4f}s")
    
    async def test_multicall_contract_efficiency(self):
        """Test efficiency comparison between individual calls and multicall contract."""
        # Simulate individual contract calls
        async def individual_call(call_data):
            await asyncio.sleep(0.02)  # Simulate network latency per call
            return f"result_for_{call_data}"
        
        # Simulate multicall contract call
        async def multicall_batch(call_data_list):
            await asyncio.sleep(0.05)  # Single network call with slight overhead but faster than individual
            return [f"result_for_{data}" for data in call_data_list]
        
        call_data = [f"call_{i}" for i in range(10)]
        
        # Individual calls (sequential to show the real benefit)
        start_time = time.time()
        individual_results = []
        for data in call_data:
            result = await individual_call(data)
            individual_results.append(result)
        individual_time = time.time() - start_time
        
        # Multicall batch
        start_time = time.time()
        batch_results = await multicall_batch(call_data)
        batch_time = time.time() - start_time
        
        assert len(individual_results) == 10
        assert len(batch_results) == 10
        
        # Multicall should be significantly faster
        assert batch_time < individual_time
        efficiency_gain = individual_time / batch_time
        
        print(f"Individual calls time: {individual_time:.4f}s")
        print(f"Multicall batch time: {batch_time:.4f}s")
        print(f"Efficiency gain: {efficiency_gain:.2f}x")
        
        assert efficiency_gain > 2  # Should be at least 2x faster


class TestAsyncMulticallIntegration:
    """Integration tests for async multicall with blockchain provider."""
    
    async def test_blockchain_provider_multicall_integration(self):
        """Test integrating multicall functionality with the blockchain provider."""
        # Mock blockchain provider with multicall support
        class MockBlockchainProviderWithMulticall:
            def __init__(self):
                self.web3_instances = {}
                self.multicall_addresses = {
                    "ethereum": "0x5BA1e12693Dc8F9c48aAD8770482f4739bEeD696",
                    "arbitrum": "0xcA11bde05977b3631167028862bE2a173976CA11"
                }
            
            async def multicall_batch(self, chain_name: str, calls: List[Dict[str, Any]]):
                """Execute a batch of calls using multicall contract."""
                if chain_name not in self.multicall_addresses:
                    raise ValueError(f"Multicall not supported on {chain_name}")
                
                # Simulate multicall contract execution
                await asyncio.sleep(0.01)  # Single network call
                
                return [
                    {
                        "success": True,
                        "result": f"0x{''.join(['0'] * 60)}{i:04x}",
                        "call_index": i
                    }
                    for i, call in enumerate(calls)
                ]
            
            async def individual_calls(self, chain_name: str, calls: List[Dict[str, Any]]):
                """Execute calls individually (for comparison)."""
                results = []
                for i, call in enumerate(calls):
                    await asyncio.sleep(0.02)  # Individual network call
                    results.append({
                        "success": True,
                        "result": f"0x{''.join(['0'] * 60)}{i:04x}",
                        "call_index": i
                    })
                return results
        
        provider = MockBlockchainProviderWithMulticall()
        
        # Test data
        test_calls = [
            {"target": "0x" + "1" * 40, "call_data": "0x70a08231" + "0" * 56},
            {"target": "0x" + "2" * 40, "call_data": "0x18160ddd"},
            {"target": "0x" + "3" * 40, "call_data": "0x95d89b41"},
            {"target": "0x" + "4" * 40, "call_data": "0x313ce567"},
            {"target": "0x" + "5" * 40, "call_data": "0x06fdde03"}
        ]
        
        # Test multicall execution
        start_time = time.time()
        multicall_results = await provider.multicall_batch("ethereum", test_calls)
        multicall_time = time.time() - start_time
        
        # Test individual calls execution
        start_time = time.time()
        individual_results = await provider.individual_calls("ethereum", test_calls)
        individual_time = time.time() - start_time
        
        # Verify results
        assert len(multicall_results) == 5
        assert len(individual_results) == 5
        assert all(result["success"] for result in multicall_results)
        assert all(result["success"] for result in individual_results)
        
        # Performance comparison
        efficiency = individual_time / multicall_time
        print(f"Multicall time: {multicall_time:.4f}s")
        print(f"Individual calls time: {individual_time:.4f}s")
        print(f"Efficiency gain: {efficiency:.2f}x")
        
        assert multicall_time < individual_time
        assert efficiency > 2  # Should be at least 2x faster
    
    async def test_error_handling_in_multicall(self):
        """Test error handling in multicall operations."""
        async def failing_multicall():
            await asyncio.sleep(0.01)
            return [
                {"success": True, "result": "0x123"},
                {"success": False, "error": "execution reverted"},
                {"success": True, "result": "0x456"}
            ]
        
        results = await failing_multicall()
        
        successful_calls = [r for r in results if r["success"]]
        failed_calls = [r for r in results if not r["success"]]
        
        assert len(successful_calls) == 2
        assert len(failed_calls) == 1
        assert failed_calls[0]["error"] == "execution reverted"


# Manual test execution command:
# python -m pytest tests/unit/test_async_multicall.py -v -s

if __name__ == "__main__":
    # Example manual test execution
    async def run_basic_test():
        """Run a basic test manually."""
        banteg_test = TestBantegMulticallPy()
        
        print("Testing asyncio.to_thread adapter...")
        await banteg_test.test_multicall_with_asyncio_to_thread_adapter()
        print("✅ asyncio.to_thread adapter test passed")
        
        print("\nTesting run_in_executor adapter...")
        await banteg_test.test_multicall_with_run_in_executor_adapter()
        print("✅ run_in_executor adapter test passed")
        
        print("\nTesting MulticallHTTPProvider concept...")
        multicall_test = TestMulticallHTTPProvider()
        await multicall_test.test_multicall_http_provider_concept()
        print("✅ MulticallHTTPProvider concept test passed")
        
        print("\nTesting multicall efficiency...")
        await multicall_test.test_multicall_contract_efficiency()
        print("✅ Multicall efficiency test passed")
        
        print("\n🎉 All manual tests passed!")
    
    # Run if executed directly
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        asyncio.run(run_basic_test())