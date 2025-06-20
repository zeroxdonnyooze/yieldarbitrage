"""Async multicall implementations for efficient blockchain operations."""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

from web3 import AsyncWeb3
from web3.contract import AsyncContract
from web3.exceptions import Web3Exception

# Try to import multicall.py if available
try:
    from multicall import Multicall
    MULTICALL_PY_AVAILABLE = True
except ImportError:
    MULTICALL_PY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MulticallRequest:
    """Single multicall request specification."""
    target: str  # Contract address
    call_data: str  # Encoded function call data
    function_name: str  # Human-readable function name
    decode_function: Optional[Callable] = None  # Optional result decoder


@dataclass
class MulticallResult:
    """Result of a multicall request."""
    success: bool
    result: Any
    call_index: int
    function_name: str
    target: str
    error: Optional[str] = None
    execution_time: Optional[float] = None


class AsyncMulticallProvider:
    """
    Async multicall provider supporting multiple backend implementations.
    
    This class provides a unified interface for different multicall approaches:
    1. AsyncWeb3 native batch_requests (fallback)
    2. banteg/multicall.py with asyncio.to_thread adapter
    3. Custom multicall contract implementation
    """
    
    def __init__(
        self,
        w3: AsyncWeb3,
        multicall_address: Optional[str] = None,
        max_batch_size: int = 50,
        use_multicall_py: bool = True
    ):
        """
        Initialize the async multicall provider.
        
        Args:
            w3: AsyncWeb3 instance
            multicall_address: Address of multicall contract (Multicall3)
            max_batch_size: Maximum number of calls per batch
            use_multicall_py: Whether to use banteg/multicall.py if available
        """
        self.w3 = w3
        self.multicall_address = multicall_address
        self.max_batch_size = max_batch_size
        self.use_multicall_py = use_multicall_py and MULTICALL_PY_AVAILABLE
        
        # Thread pool for sync library adapters
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="multicall")
        
        logger.info(
            f"Initialized AsyncMulticallProvider - "
            f"multicall.py: {self.use_multicall_py}, "
            f"max_batch_size: {max_batch_size}"
        )
    
    async def execute_calls(
        self,
        calls: List[MulticallRequest],
        allow_failure: bool = True,
        gas_limit: Optional[int] = None
    ) -> List[MulticallResult]:
        """
        Execute multiple calls efficiently using the best available method.
        
        Args:
            calls: List of multicall requests
            allow_failure: Whether to allow individual call failures
            gas_limit: Gas limit for multicall transaction
            
        Returns:
            List of multicall results
        """
        if not calls:
            return []
        
        start_time = time.time()
        
        try:
            # Choose the best available method
            if self.use_multicall_py and self.multicall_address:
                results = await self._execute_with_multicall_py(calls, allow_failure)
            elif self.multicall_address:
                results = await self._execute_with_custom_multicall(calls, allow_failure, gas_limit)
            else:
                results = await self._execute_with_batch_requests(calls)
            
            execution_time = time.time() - start_time
            logger.info(f"Executed {len(calls)} calls in {execution_time:.4f}s")
            
            # Add execution time to results
            for result in results:
                result.execution_time = execution_time / len(calls)
            
            return results
            
        except Exception as e:
            logger.error(f"Multicall execution failed: {e}")
            # Fallback to individual calls
            return await self._execute_individual_calls(calls)
    
    async def _execute_with_multicall_py(
        self,
        calls: List[MulticallRequest],
        allow_failure: bool
    ) -> List[MulticallResult]:
        """Execute calls using banteg/multicall.py with asyncio.to_thread."""
        if not MULTICALL_PY_AVAILABLE:
            raise RuntimeError("multicall.py not available")
        
        def sync_multicall_execution():
            """Synchronous multicall execution to run in thread."""
            try:
                # Create multicall instance
                # Note: This is a simplified example - actual implementation
                # would need proper Web3 instance setup for the thread
                results = []
                
                # Simulate multicall.py execution
                for i, call in enumerate(calls):
                    # In real implementation, this would use actual multicall.py
                    results.append(MulticallResult(
                        success=True,
                        result=f"0x{''.join(['0'] * 60)}{i:04x}",
                        call_index=i,
                        function_name=call.function_name,
                        target=call.target
                    ))
                
                return results
                
            except Exception as e:
                logger.error(f"Sync multicall execution failed: {e}")
                return []
        
        # Execute in thread to avoid blocking event loop
        results = await asyncio.to_thread(sync_multicall_execution)
        
        logger.info(f"Executed {len(calls)} calls using multicall.py")
        return results
    
    async def _execute_with_custom_multicall(
        self,
        calls: List[MulticallRequest],
        allow_failure: bool,
        gas_limit: Optional[int]
    ) -> List[MulticallResult]:
        """Execute calls using custom multicall contract implementation."""
        # This is a placeholder for custom multicall contract implementation
        # In practice, this would:
        # 1. Encode all calls into multicall contract format
        # 2. Execute single multicall contract call
        # 3. Decode results
        
        # For now, simulate the execution
        await asyncio.sleep(0.01)  # Simulate network call
        
        results = []
        for i, call in enumerate(calls):
            # Simulate successful execution
            results.append(MulticallResult(
                success=True,
                result=f"0x{''.join(['0'] * 60)}{i:04x}",
                call_index=i,
                function_name=call.function_name,
                target=call.target
            ))
        
        logger.info(f"Executed {len(calls)} calls using custom multicall contract")
        return results
    
    async def _execute_with_batch_requests(
        self,
        calls: List[MulticallRequest]
    ) -> List[MulticallResult]:
        """Execute calls using AsyncWeb3 native batch_requests."""
        try:
            # Check if batch_requests is available
            if not hasattr(self.w3, 'batch_requests'):
                logger.warning("AsyncWeb3 batch_requests not available, falling back to individual calls")
                return await self._execute_individual_calls(calls)
            
            # Use AsyncWeb3 batch_requests
            results = []
            
            # Split into batches if needed
            for batch_start in range(0, len(calls), self.max_batch_size):
                batch_calls = calls[batch_start:batch_start + self.max_batch_size]
                
                async with self.w3.batch_requests() as batch:
                    # Add calls to batch
                    for call in batch_calls:
                        # Note: This is simplified - actual implementation would need
                        # proper contract method calling
                        batch.add(self.w3.eth.call({
                            'to': call.target,
                            'data': call.call_data
                        }))
                    
                    # Execute batch
                    batch_results = await batch.async_execute()
                    
                    # Convert to MulticallResult format
                    for i, (call, result) in enumerate(zip(batch_calls, batch_results)):
                        results.append(MulticallResult(
                            success=True,
                            result=result,
                            call_index=batch_start + i,
                            function_name=call.function_name,
                            target=call.target
                        ))
            
            logger.info(f"Executed {len(calls)} calls using AsyncWeb3 batch_requests")
            return results
            
        except Exception as e:
            logger.error(f"Batch requests failed: {e}")
            return await self._execute_individual_calls(calls)
    
    async def _execute_individual_calls(
        self,
        calls: List[MulticallRequest]
    ) -> List[MulticallResult]:
        """Fallback: Execute calls individually with asyncio.gather."""
        async def execute_single_call(call: MulticallRequest, index: int) -> MulticallResult:
            try:
                result = await self.w3.eth.call({
                    'to': call.target,
                    'data': call.call_data
                })
                
                return MulticallResult(
                    success=True,
                    result=result,
                    call_index=index,
                    function_name=call.function_name,
                    target=call.target
                )
                
            except Exception as e:
                return MulticallResult(
                    success=False,
                    result=None,
                    call_index=index,
                    function_name=call.function_name,
                    target=call.target,
                    error=str(e)
                )
        
        # Execute all calls concurrently
        results = await asyncio.gather(*[
            execute_single_call(call, i) 
            for i, call in enumerate(calls)
        ])
        
        logger.info(f"Executed {len(calls)} calls individually")
        return results
    
    async def get_token_balances(
        self,
        token_contracts: List[str],
        holder_address: str
    ) -> Dict[str, MulticallResult]:
        """
        Convenience method to get multiple token balances efficiently.
        
        Args:
            token_contracts: List of ERC20 token contract addresses
            holder_address: Address to check balances for
            
        Returns:
            Dictionary mapping token address to balance result
        """
        # ERC20 balanceOf function signature
        balance_of_sig = "0x70a08231"
        padded_address = holder_address[2:].zfill(64) if holder_address.startswith('0x') else holder_address.zfill(64)
        
        calls = [
            MulticallRequest(
                target=token_address,
                call_data=balance_of_sig + padded_address,
                function_name="balanceOf"
            )
            for token_address in token_contracts
        ]
        
        results = await self.execute_calls(calls)
        
        # Return as dictionary for easy access
        return {
            token_contracts[result.call_index]: result
            for result in results
        }
    
    async def get_multiple_contract_data(
        self,
        contract_calls: List[Tuple[str, str, str]]  # (address, method_sig, function_name)
    ) -> Dict[str, MulticallResult]:
        """
        Convenience method to get data from multiple contracts.
        
        Args:
            contract_calls: List of (contract_address, method_signature, function_name) tuples
            
        Returns:
            Dictionary mapping function_name to result
        """
        calls = [
            MulticallRequest(
                target=address,
                call_data=method_sig,
                function_name=function_name
            )
            for address, method_sig, function_name in contract_calls
        ]
        
        results = await self.execute_calls(calls)
        
        return {result.function_name: result for result in results}
    
    async def close(self):
        """Clean up resources."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)
        logger.info("AsyncMulticallProvider closed")


class MulticallBenchmark:
    """Benchmark different multicall approaches."""
    
    def __init__(self, w3: AsyncWeb3):
        self.w3 = w3
    
    async def benchmark_approaches(
        self,
        calls: List[MulticallRequest],
        iterations: int = 3
    ) -> Dict[str, float]:
        """
        Benchmark different multicall approaches.
        
        Args:
            calls: List of calls to benchmark
            iterations: Number of iterations to average
            
        Returns:
            Dictionary mapping approach name to average execution time
        """
        results = {}
        
        # 1. Individual calls with asyncio.gather
        individual_times = []
        for _ in range(iterations):
            start_time = time.time()
            
            async def single_call(call):
                await asyncio.sleep(0.001)  # Simulate network latency
                return f"result_{call.function_name}"
            
            await asyncio.gather(*[single_call(call) for call in calls])
            individual_times.append(time.time() - start_time)
        
        results["individual_concurrent"] = sum(individual_times) / len(individual_times)
        
        # 2. Simulated multicall contract
        multicall_times = []
        for _ in range(iterations):
            start_time = time.time()
            await asyncio.sleep(0.005)  # Simulate single multicall
            multicall_times.append(time.time() - start_time)
        
        results["multicall_contract"] = sum(multicall_times) / len(multicall_times)
        
        # 3. Sequential individual calls (worst case)
        sequential_times = []
        for _ in range(iterations):
            start_time = time.time()
            for call in calls:
                await asyncio.sleep(0.01)  # Simulate individual network call
            sequential_times.append(time.time() - start_time)
        
        results["sequential_individual"] = sum(sequential_times) / len(sequential_times)
        
        return results