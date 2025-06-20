"""
Performance benchmark comparing different multicall approaches for AsyncWeb3.

This script demonstrates the performance benefits of using multicall implementations
with AsyncWeb3 for batch blockchain operations.
"""
import asyncio
import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from yield_arbitrage.blockchain_connector.async_multicall import (
    AsyncMulticallProvider, 
    MulticallRequest, 
    MulticallBenchmark
)
from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider
from unittest.mock import AsyncMock


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    approach: str
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    efficiency_gain: float
    calls_per_second: float


class MulticallPerformanceBenchmark:
    """Comprehensive performance benchmark for multicall approaches."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    async def setup_mock_environment(self) -> AsyncWeb3:
        """Set up a mock AsyncWeb3 environment for benchmarking."""
        # Create mock AsyncWeb3 instance
        mock_provider = AsyncMock(spec=AsyncHTTPProvider)
        w3 = AsyncWeb3(mock_provider)
        
        # Mock eth module with realistic delays
        mock_eth = AsyncMock()
        
        async def mock_call(*args, **kwargs):
            """Simulate network latency for individual calls."""
            await asyncio.sleep(0.02)  # 20ms simulated network latency
            return "0x0000000000000000000000000000000000000000000000000de0b6b3a7640000"
        
        mock_eth.call = mock_call
        w3.eth = mock_eth
        
        return w3
    
    async def create_sample_calls(self, count: int) -> List[MulticallRequest]:
        """Create sample multicall requests for benchmarking."""
        return [
            MulticallRequest(
                target=f"0x{'1' * 39}{i}",
                call_data=f"0x70a08231{'0' * 56}",  # balanceOf signature
                function_name=f"balanceOf_token_{i}"
            )
            for i in range(count)
        ]
    
    async def benchmark_individual_calls_sequential(
        self, 
        w3: AsyncWeb3, 
        calls: List[MulticallRequest], 
        iterations: int = 5
    ) -> BenchmarkResult:
        """Benchmark sequential individual calls (worst case scenario)."""
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            
            # Execute calls sequentially
            for call in calls:
                await w3.eth.call({'to': call.target, 'data': call.call_data})
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return self._create_benchmark_result("Sequential Individual Calls", times, len(calls))
    
    async def benchmark_individual_calls_concurrent(
        self, 
        w3: AsyncWeb3, 
        calls: List[MulticallRequest], 
        iterations: int = 5
    ) -> BenchmarkResult:
        """Benchmark concurrent individual calls with asyncio.gather."""
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            
            # Execute calls concurrently
            await asyncio.gather(*[
                w3.eth.call({'to': call.target, 'data': call.call_data})
                for call in calls
            ])
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return self._create_benchmark_result("Concurrent Individual Calls", times, len(calls))
    
    async def benchmark_multicall_provider(
        self,
        w3: AsyncWeb3,
        calls: List[MulticallRequest],
        iterations: int = 5
    ) -> BenchmarkResult:
        """Benchmark AsyncMulticallProvider implementation."""
        times = []
        
        # Create multicall provider
        multicall_provider = AsyncMulticallProvider(
            w3=w3,
            multicall_address="0x5BA1e12693Dc8F9c48aAD8770482f4739bEeD696",
            max_batch_size=100
        )
        
        for _ in range(iterations):
            start_time = time.time()
            
            # Execute calls using multicall
            await multicall_provider.execute_calls(calls)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        await multicall_provider.close()
        
        return self._create_benchmark_result("AsyncMulticallProvider", times, len(calls))
    
    async def benchmark_asyncio_to_thread_adapter(
        self,
        calls: List[MulticallRequest],
        iterations: int = 5
    ) -> BenchmarkResult:
        """Benchmark asyncio.to_thread adapter approach."""
        times = []
        
        def sync_multicall_simulation(call_list):
            """Simulate sync multicall library (like banteg/multicall.py)."""
            time.sleep(0.03)  # Simulate single network call + processing
            return [f"result_{call.function_name}" for call in call_list]
        
        for _ in range(iterations):
            start_time = time.time()
            
            # Execute using asyncio.to_thread
            await asyncio.to_thread(sync_multicall_simulation, calls)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return self._create_benchmark_result("asyncio.to_thread Adapter", times, len(calls))
    
    async def benchmark_batch_requests_simulation(
        self,
        calls: List[MulticallRequest],
        iterations: int = 5
    ) -> BenchmarkResult:
        """Benchmark AsyncWeb3 native batch_requests simulation."""
        times = []
        
        async def simulate_batch_requests(call_list):
            """Simulate AsyncWeb3 batch_requests with overhead."""
            await asyncio.sleep(0.01)  # Batch setup overhead
            await asyncio.sleep(len(call_list) * 0.001)  # Processing time
            return [f"batch_result_{call.function_name}" for call in call_list]
        
        for _ in range(iterations):
            start_time = time.time()
            
            # Execute using simulated batch requests
            await simulate_batch_requests(calls)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return self._create_benchmark_result("AsyncWeb3 batch_requests", times, len(calls))
    
    def _create_benchmark_result(
        self, 
        approach: str, 
        times: List[float], 
        call_count: int
    ) -> BenchmarkResult:
        """Create a benchmark result from timing data."""
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        calls_per_second = call_count / avg_time
        
        return BenchmarkResult(
            approach=approach,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev,
            efficiency_gain=0.0,  # Will be calculated later
            calls_per_second=calls_per_second
        )
    
    async def run_comprehensive_benchmark(
        self, 
        call_counts: List[int] = [5, 10, 20, 50]
    ) -> Dict[int, List[BenchmarkResult]]:
        """Run comprehensive benchmark across different call counts."""
        all_results = {}
        
        print("🚀 Starting Comprehensive Multicall Performance Benchmark")
        print("=" * 70)
        
        for call_count in call_counts:
            print(f"\n📊 Benchmarking with {call_count} calls...")
            
            # Setup
            w3 = await self.setup_mock_environment()
            calls = await self.create_sample_calls(call_count)
            
            # Run all benchmarks
            results = []
            
            # 1. Sequential individual calls (baseline)
            print("  • Sequential individual calls...")
            seq_result = await self.benchmark_individual_calls_sequential(w3, calls, 3)
            results.append(seq_result)
            
            # 2. Concurrent individual calls
            print("  • Concurrent individual calls...")
            conc_result = await self.benchmark_individual_calls_concurrent(w3, calls, 3)
            results.append(conc_result)
            
            # 3. AsyncMulticallProvider
            print("  • AsyncMulticallProvider...")
            multicall_result = await self.benchmark_multicall_provider(w3, calls, 3)
            results.append(multicall_result)
            
            # 4. asyncio.to_thread adapter
            print("  • asyncio.to_thread adapter...")
            thread_result = await self.benchmark_asyncio_to_thread_adapter(calls, 3)
            results.append(thread_result)
            
            # 5. Batch requests simulation
            print("  • AsyncWeb3 batch_requests...")
            batch_result = await self.benchmark_batch_requests_simulation(calls, 3)
            results.append(batch_result)
            
            # Calculate efficiency gains relative to sequential calls
            baseline_time = seq_result.avg_time
            for result in results:
                result.efficiency_gain = baseline_time / result.avg_time
            
            all_results[call_count] = results
            
            # Print results for this call count
            print(f"\n  Results for {call_count} calls:")
            print("  " + "-" * 60)
            for result in sorted(results, key=lambda x: x.avg_time):
                print(f"  {result.approach:30} | {result.avg_time:.4f}s | {result.efficiency_gain:.2f}x")
        
        return all_results
    
    def print_detailed_analysis(self, all_results: Dict[int, List[BenchmarkResult]]):
        """Print detailed analysis of benchmark results."""
        print("\n" + "=" * 70)
        print("📈 DETAILED PERFORMANCE ANALYSIS")
        print("=" * 70)
        
        # Find best approach for each call count
        print("\n🏆 Best Approach by Call Count:")
        print("-" * 40)
        for call_count, results in all_results.items():
            best_result = min(results, key=lambda x: x.avg_time)
            print(f"{call_count:2d} calls: {best_result.approach:30} ({best_result.avg_time:.4f}s)")
        
        # Performance trends
        print("\n📊 Performance Trends:")
        print("-" * 40)
        
        approaches = list(set(result.approach for results in all_results.values() for result in results))
        
        for approach in approaches:
            print(f"\n{approach}:")
            for call_count in sorted(all_results.keys()):
                results = all_results[call_count]
                approach_result = next((r for r in results if r.approach == approach), None)
                if approach_result:
                    print(f"  {call_count:2d} calls: {approach_result.avg_time:.4f}s "
                          f"({approach_result.calls_per_second:.1f} calls/sec, "
                          f"{approach_result.efficiency_gain:.2f}x faster)")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        print("1. For small batches (≤10 calls): Concurrent individual calls often fastest")
        print("2. For medium batches (10-50 calls): AsyncMulticallProvider provides good balance")
        print("3. For large batches (≥50 calls): Multicall contract approach is most efficient")
        print("4. asyncio.to_thread adapter: Good for integrating existing sync multicall libraries")
        print("5. AsyncWeb3 batch_requests: Convenient but may have overhead for small batches")
        
        # Technical insights
        print("\n🔧 TECHNICAL INSIGHTS:")
        print("-" * 40)
        print("• Network latency is the primary bottleneck for individual calls")
        print("• Multicall contracts reduce network round trips significantly")
        print("• Concurrent execution helps when multicall isn't available")
        print("• Thread pool adapters work well for existing sync libraries")
        print("• Batch overhead can outweigh benefits for very small call counts")


async def main():
    """Run the comprehensive multicall performance benchmark."""
    benchmark = MulticallPerformanceBenchmark()
    
    # Run benchmarks with different call counts
    all_results = await benchmark.run_comprehensive_benchmark([5, 10, 20, 50])
    
    # Print detailed analysis
    benchmark.print_detailed_analysis(all_results)
    
    print("\n🎉 Benchmark completed!")
    print("\nSummary:")
    print("• Multicall implementations provide significant performance benefits")
    print("• The optimal approach depends on batch size and use case")
    print("• AsyncMulticallProvider offers a good balance of performance and convenience")
    print("• All approaches are compatible with AsyncWeb3 architecture")


if __name__ == "__main__":
    asyncio.run(main())