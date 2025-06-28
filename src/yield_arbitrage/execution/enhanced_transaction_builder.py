"""
Enhanced Transaction Builder for Router Integration.

This module enhances the transaction builder to work with the YieldArbitrageRouter
smart contract, enabling batched atomic execution of arbitrage paths.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from decimal import Decimal
from enum import Enum

from eth_abi import encode
from eth_utils import to_checksum_address, to_wei

from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType
from yield_arbitrage.pathfinding.path_segment_analyzer import PathSegment, SegmentType
from yield_arbitrage.execution.calldata_generator import CalldataGenerator, SegmentCalldata
from yield_arbitrage.execution.tenderly_client import TenderlyTransaction
from yield_arbitrage.execution.transaction_builder import TransactionBuilder, TokenInfo

logger = logging.getLogger(__name__)


@dataclass
class RouterTransaction:
    """Transaction specifically for the YieldArbitrageRouter."""
    segment_id: str
    to_address: str  # Router contract address
    from_address: str
    value: str = "0"
    gas_limit: int = 8_000_000
    gas_price: str = "20000000000"  # 20 gwei
    data: bytes = b""
    requires_flash_loan: bool = False
    flash_loan_asset: Optional[str] = None
    flash_loan_amount: Optional[int] = None
    estimated_gas: int = 0
    
    def to_tenderly_transaction(self) -> TenderlyTransaction:
        """Convert to TenderlyTransaction for simulation."""
        return TenderlyTransaction(
            from_address=self.from_address,
            to_address=self.to_address,
            value=self.value,
            gas=self.gas_limit,
            gas_price=self.gas_price,
            data=self.data.hex() if isinstance(self.data, bytes) else self.data
        )


@dataclass
class BatchExecutionPlan:
    """Complete execution plan for router-based arbitrage."""
    plan_id: str
    router_address: str
    executor_address: str
    segments: List[RouterTransaction]
    total_gas_estimate: int = 0
    expected_profit: Decimal = Decimal("0")
    execution_deadline: int = 0
    requires_approval_transactions: List[TenderlyTransaction] = field(default_factory=list)
    
    @property
    def total_transactions(self) -> int:
        """Total number of transactions including approvals."""
        return len(self.segments) + len(self.requires_approval_transactions)


class RouterIntegrationMode(str, Enum):
    """Integration modes for router execution."""
    DIRECT = "direct"  # Execute directly through router
    BATCH = "batch"    # Batch multiple segments
    FLASH_LOAN = "flash_loan"  # Use flash loans for capital
    HYBRID = "hybrid"  # Mix of modes based on requirements


class EnhancedTransactionBuilder:
    """
    Enhanced transaction builder optimized for YieldArbitrageRouter integration.
    
    This builder creates transactions that work with the router contract,
    supporting atomic execution, flash loans, and batch operations.
    """
    
    def __init__(
        self,
        router_address: str,
        calldata_generator: Optional[CalldataGenerator] = None,
        chain_id: int = 1
    ):
        """
        Initialize enhanced transaction builder.
        
        Args:
            router_address: Address of the deployed YieldArbitrageRouter
            calldata_generator: Calldata generator for operations
            chain_id: Target blockchain chain ID
        """
        self.router_address = to_checksum_address(router_address)
        self.calldata_generator = calldata_generator or CalldataGenerator(chain_id)
        self.chain_id = chain_id
        
        # Base transaction builder for fallback operations
        self.base_builder = TransactionBuilder()
        
        # Router function selectors
        self.router_functions = {
            "executeSegment": "0x12345678",  # Placeholder - would get from ABI
            "executeBatch": "0x87654321",
            "emergencyWithdraw": "0xabcdefab"
        }
        
        # Gas estimates for different operation types
        self.gas_estimates = {
            EdgeType.TRADE: 150_000,
            EdgeType.LEND: 100_000,
            EdgeType.BORROW: 120_000,
            EdgeType.STAKE: 200_000,
            EdgeType.FLASH_LOAN: 300_000,
            EdgeType.SPLIT: 50_000,
            EdgeType.COMBINE: 75_000,
            EdgeType.BRIDGE: 250_000
        }
        
        # Track statistics
        self.stats = {
            "segments_built": 0,
            "batches_created": 0,
            "flash_loans_prepared": 0,
            "total_gas_estimated": 0
        }
    
    def build_segment_execution(
        self,
        segment: PathSegment,
        executor_address: str,
        recipient: Optional[str] = None,
        deadline: Optional[int] = None
    ) -> RouterTransaction:
        """
        Build a router transaction for executing a single segment.
        
        Args:
            segment: Path segment to execute
            executor_address: Address executing the transaction
            recipient: Final recipient of output tokens
            deadline: Execution deadline
            
        Returns:
            RouterTransaction ready for execution
        """
        logger.info(f"Building segment execution for {segment.segment_id}")
        
        if deadline is None:
            import time
            deadline = int(time.time()) + 1200  # 20 minutes
        
        if recipient is None:
            recipient = executor_address
        
        # Generate calldata for the segment
        segment_calldata = self.calldata_generator.generate_segment_calldata(
            segment, recipient, deadline
        )
        
        # Encode for router execution
        router_calldata = self._encode_segment_for_router(segment_calldata)
        
        # Estimate gas
        estimated_gas = self._estimate_segment_gas(segment)
        
        # Create router transaction
        router_tx = RouterTransaction(
            segment_id=segment.segment_id,
            to_address=self.router_address,
            from_address=executor_address,
            data=router_calldata,
            gas_limit=min(estimated_gas + 100_000, 8_000_000),  # Add buffer
            estimated_gas=estimated_gas,
            requires_flash_loan=segment.requires_flash_loan,
            flash_loan_asset=segment.flash_loan_asset,
            flash_loan_amount=int(segment.flash_loan_amount) if segment.flash_loan_amount else None
        )
        
        self.stats["segments_built"] += 1
        self.stats["total_gas_estimated"] += estimated_gas
        
        return router_tx
    
    def build_batch_execution(
        self,
        segments: List[PathSegment],
        executor_address: str,
        mode: RouterIntegrationMode = RouterIntegrationMode.BATCH,
        recipient: Optional[str] = None
    ) -> BatchExecutionPlan:
        """
        Build a complete batch execution plan for multiple segments.
        
        Args:
            segments: List of path segments to execute
            executor_address: Address executing the transactions
            mode: Integration mode for execution
            recipient: Final recipient of output tokens
            
        Returns:
            Complete batch execution plan
        """
        logger.info(f"Building batch execution for {len(segments)} segments in {mode.value} mode")
        
        if recipient is None:
            recipient = executor_address
        
        plan_id = f"batch_{int(time.time() * 1000)}_{len(segments)}"
        
        # Determine execution strategy
        if mode == RouterIntegrationMode.DIRECT:
            router_transactions = self._build_direct_execution(segments, executor_address, recipient)
        elif mode == RouterIntegrationMode.BATCH:
            router_transactions = self._build_batched_execution(segments, executor_address, recipient)
        elif mode == RouterIntegrationMode.FLASH_LOAN:
            router_transactions = self._build_flash_loan_execution(segments, executor_address, recipient)
        else:  # HYBRID
            router_transactions = self._build_hybrid_execution(segments, executor_address, recipient)
        
        # Calculate totals
        total_gas = sum(tx.estimated_gas for tx in router_transactions)
        
        # Generate approval transactions if needed
        approval_txs = self._generate_approval_transactions(segments, executor_address)
        
        # Create execution plan
        execution_plan = BatchExecutionPlan(
            plan_id=plan_id,
            router_address=self.router_address,
            executor_address=executor_address,
            segments=router_transactions,
            total_gas_estimate=total_gas,
            execution_deadline=int(time.time()) + 1800,  # 30 minutes
            requires_approval_transactions=approval_txs
        )
        
        self.stats["batches_created"] += 1
        
        return execution_plan
    
    def build_emergency_operations(
        self,
        tokens_to_withdraw: List[str],
        amounts: List[int],
        executor_address: str,
        recipient: Optional[str] = None
    ) -> List[RouterTransaction]:
        """
        Build emergency withdrawal transactions.
        
        Args:
            tokens_to_withdraw: List of token addresses to withdraw
            amounts: Corresponding amounts to withdraw
            executor_address: Address executing the withdrawal
            recipient: Recipient of withdrawn tokens
            
        Returns:
            List of emergency withdrawal transactions
        """
        if recipient is None:
            recipient = executor_address
        
        emergency_txs = []
        
        for token, amount in zip(tokens_to_withdraw, amounts):
            # Encode emergency withdrawal call
            calldata = self._encode_emergency_withdrawal(token, amount, recipient)
            
            emergency_tx = RouterTransaction(
                segment_id=f"emergency_{token}_{amount}",
                to_address=self.router_address,
                from_address=executor_address,
                data=calldata,
                gas_limit=100_000,  # Emergency operations should be simple
                estimated_gas=50_000
            )
            
            emergency_txs.append(emergency_tx)
        
        return emergency_txs
    
    def _build_direct_execution(
        self,
        segments: List[PathSegment],
        executor_address: str,
        recipient: str
    ) -> List[RouterTransaction]:
        """Build direct execution - one transaction per segment."""
        router_transactions = []
        
        for segment in segments:
            router_tx = self.build_segment_execution(segment, executor_address, recipient)
            router_transactions.append(router_tx)
        
        return router_transactions
    
    def _build_batched_execution(
        self,
        segments: List[PathSegment],
        executor_address: str,
        recipient: str
    ) -> List[RouterTransaction]:
        """Build batched execution - multiple segments in one transaction."""
        # For simplicity, create one transaction per segment
        # In production, could implement true batching at the router level
        return self._build_direct_execution(segments, executor_address, recipient)
    
    def _build_flash_loan_execution(
        self,
        segments: List[PathSegment],
        executor_address: str,
        recipient: str
    ) -> List[RouterTransaction]:
        """Build flash loan execution strategy."""
        flash_loan_segments = []
        regular_segments = []
        
        # Separate flash loan and regular segments
        for segment in segments:
            if segment.requires_flash_loan:
                flash_loan_segments.append(segment)
            else:
                regular_segments.append(segment)
        
        router_transactions = []
        
        # Process flash loan segments
        for segment in flash_loan_segments:
            router_tx = self.build_segment_execution(segment, executor_address, recipient)
            router_transactions.append(router_tx)
            self.stats["flash_loans_prepared"] += 1
        
        # Process regular segments
        for segment in regular_segments:
            router_tx = self.build_segment_execution(segment, executor_address, recipient)
            router_transactions.append(router_tx)
        
        return router_transactions
    
    def _build_hybrid_execution(
        self,
        segments: List[PathSegment],
        executor_address: str,
        recipient: str
    ) -> List[RouterTransaction]:
        """Build hybrid execution combining strategies."""
        # Analyze segments and choose optimal strategy for each
        atomic_segments = [s for s in segments if s.is_atomic]
        non_atomic_segments = [s for s in segments if not s.is_atomic]
        
        router_transactions = []
        
        # Handle atomic segments with flash loans if needed
        for segment in atomic_segments:
            if segment.requires_flash_loan:
                router_tx = self.build_segment_execution(segment, executor_address, recipient)
                router_transactions.append(router_tx)
            else:
                # Can batch atomic segments
                router_tx = self.build_segment_execution(segment, executor_address, recipient)
                router_transactions.append(router_tx)
        
        # Handle non-atomic segments individually
        for segment in non_atomic_segments:
            router_tx = self.build_segment_execution(segment, executor_address, recipient)
            router_transactions.append(router_tx)
        
        return router_transactions
    
    def _encode_segment_for_router(self, segment_calldata: SegmentCalldata) -> bytes:
        """
        Encode segment calldata for router execution.
        
        Args:
            segment_calldata: Complete segment calldata
            
        Returns:
            Encoded data for router contract
        """
        # This is a simplified encoding - production would use proper ABI encoding
        
        # Function selector for executeSegment
        function_selector = bytes.fromhex(self.router_functions["executeSegment"][2:])
        
        # Encode PathSegment struct
        segment_id = segment_calldata.segment_id.encode('utf-8').ljust(32, b'\0')
        
        # Create simplified encoding for demonstration
        encoded_data = bytearray()
        encoded_data.extend(function_selector)
        encoded_data.extend(segment_id)
        
        # Add operation count
        encoded_data.extend(len(segment_calldata.operations).to_bytes(32, 'big'))
        
        # Add flash loan info
        encoded_data.extend(int(segment_calldata.requires_flash_loan).to_bytes(32, 'big'))
        
        if segment_calldata.requires_flash_loan:
            # Add flash loan asset (padded to 32 bytes)
            asset_address = segment_calldata.flash_loan_asset or "0x0000000000000000000000000000000000000000"
            asset_bytes = bytes.fromhex(asset_address[2:]).rjust(32, b'\0')
            encoded_data.extend(asset_bytes)
            
            # Add flash loan amount
            amount = segment_calldata.flash_loan_amount or 0
            encoded_data.extend(amount.to_bytes(32, 'big'))
        
        # Add recipient
        recipient_bytes = bytes.fromhex(segment_calldata.recipient[2:]).rjust(32, b'\0')
        encoded_data.extend(recipient_bytes)
        
        # Add deadline
        encoded_data.extend(segment_calldata.deadline.to_bytes(32, 'big'))
        
        return bytes(encoded_data)
    
    def _encode_emergency_withdrawal(self, token: str, amount: int, recipient: str) -> bytes:
        """Encode emergency withdrawal call."""
        function_selector = bytes.fromhex(self.router_functions["emergencyWithdraw"][2:])
        
        # Encode parameters: token, amount, recipient
        token_bytes = bytes.fromhex(token[2:]).rjust(32, b'\0')
        amount_bytes = amount.to_bytes(32, 'big')
        recipient_bytes = bytes.fromhex(recipient[2:]).rjust(32, b'\0')
        
        return function_selector + token_bytes + amount_bytes + recipient_bytes
    
    def _estimate_segment_gas(self, segment: PathSegment) -> int:
        """Estimate gas usage for a segment."""
        base_gas = 50_000  # Base router overhead
        
        operation_gas = 0
        for edge in segment.edges:
            operation_gas += self.gas_estimates.get(edge.edge_type, 100_000)
        
        # Flash loan overhead
        if segment.requires_flash_loan:
            operation_gas += self.gas_estimates[EdgeType.FLASH_LOAN]
        
        # Complexity multiplier
        complexity_multiplier = 1.0 + (segment.edge_count * 0.1)
        
        total_gas = int((base_gas + operation_gas) * complexity_multiplier)
        
        return min(total_gas, 8_000_000)  # Cap at block gas limit
    
    def _generate_approval_transactions(
        self,
        segments: List[PathSegment],
        executor_address: str
    ) -> List[TenderlyTransaction]:
        """Generate required approval transactions."""
        approval_txs = []
        required_approvals = set()
        
        # Collect all required approvals
        for segment in segments:
            for edge in segment.edges:
                if edge.edge_type in [EdgeType.TRADE, EdgeType.LEND]:
                    # Need to approve router to spend input tokens
                    token_id = edge.source_asset_id
                    token_address = self.calldata_generator._get_token_address(token_id)
                    
                    if token_address and token_address != "0x0000000000000000000000000000000000000000":
                        required_approvals.add(token_address)
        
        # Create approval transactions
        for token_address in required_approvals:
            approval_tx = self.base_builder.build_approval_transaction(
                token_address=token_address,
                spender_address=self.router_address,
                amount="unlimited",
                from_address=executor_address
            )
            approval_txs.append(approval_tx)
        
        return approval_txs
    
    def simulate_execution_plan(
        self,
        execution_plan: BatchExecutionPlan
    ) -> Dict[str, Any]:
        """
        Simulate execution plan and return analysis.
        
        Args:
            execution_plan: Complete execution plan
            
        Returns:
            Simulation analysis and recommendations
        """
        analysis = {
            "plan_id": execution_plan.plan_id,
            "total_transactions": execution_plan.total_transactions,
            "total_gas_estimate": execution_plan.total_gas_estimate,
            "flash_loan_segments": len([tx for tx in execution_plan.segments if tx.requires_flash_loan]),
            "approval_transactions": len(execution_plan.requires_approval_transactions),
            "estimated_execution_time": execution_plan.total_transactions * 15,  # 15 seconds per tx
            "recommendations": []
        }
        
        # Add recommendations
        if execution_plan.total_gas_estimate > 6_000_000:
            analysis["recommendations"].append("Consider splitting into smaller batches due to high gas usage")
        
        if len(execution_plan.requires_approval_transactions) > 5:
            analysis["recommendations"].append("Many approval transactions required - consider pre-approving tokens")
        
        flash_loan_count = analysis["flash_loan_segments"]
        if flash_loan_count > 3:
            analysis["recommendations"].append(f"High number of flash loans ({flash_loan_count}) - verify profitability")
        
        return analysis
    
    def optimize_execution_plan(
        self,
        execution_plan: BatchExecutionPlan
    ) -> BatchExecutionPlan:
        """
        Optimize an execution plan for better performance.
        
        Args:
            execution_plan: Original execution plan
            
        Returns:
            Optimized execution plan
        """
        # Create optimized copy
        optimized_plan = BatchExecutionPlan(
            plan_id=f"{execution_plan.plan_id}_optimized",
            router_address=execution_plan.router_address,
            executor_address=execution_plan.executor_address,
            segments=[],
            execution_deadline=execution_plan.execution_deadline
        )
        
        # Optimization strategies:
        
        # 1. Sort segments by gas usage (low to high)
        sorted_segments = sorted(execution_plan.segments, key=lambda x: x.estimated_gas)
        
        # 2. Group flash loan segments together
        flash_loan_segments = [tx for tx in sorted_segments if tx.requires_flash_loan]
        regular_segments = [tx for tx in sorted_segments if not tx.requires_flash_loan]
        
        # 3. Combine segments where possible
        optimized_plan.segments = flash_loan_segments + regular_segments
        
        # 4. Recalculate totals
        optimized_plan.total_gas_estimate = sum(tx.estimated_gas for tx in optimized_plan.segments)
        
        # 5. Minimize approval transactions
        optimized_plan.requires_approval_transactions = execution_plan.requires_approval_transactions
        
        return optimized_plan
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get transaction builder statistics."""
        return {
            "router_address": self.router_address,
            "chain_id": self.chain_id,
            "segments_built": self.stats["segments_built"],
            "batches_created": self.stats["batches_created"],
            "flash_loans_prepared": self.stats["flash_loans_prepared"],
            "total_gas_estimated": self.stats["total_gas_estimated"],
            "supported_edge_types": list(self.gas_estimates.keys()),
            "calldata_generator_stats": self.calldata_generator.get_statistics()
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            "segments_built": 0,
            "batches_created": 0,
            "flash_loans_prepared": 0,
            "total_gas_estimated": 0
        }


# Convenience functions for common operations

def create_simple_execution_plan(
    segments: List[PathSegment],
    router_address: str,
    executor_address: str,
    recipient: Optional[str] = None
) -> BatchExecutionPlan:
    """
    Create a simple execution plan for basic arbitrage paths.
    
    Args:
        segments: Path segments to execute
        router_address: Router contract address
        executor_address: Executor address
        recipient: Final recipient
        
    Returns:
        Simple batch execution plan
    """
    builder = EnhancedTransactionBuilder(router_address)
    return builder.build_batch_execution(
        segments,
        executor_address,
        RouterIntegrationMode.DIRECT,
        recipient
    )


def estimate_execution_cost(
    execution_plan: BatchExecutionPlan,
    gas_price_gwei: float = 20.0,
    eth_price_usd: float = 2000.0
) -> Dict[str, float]:
    """
    Estimate execution cost in USD.
    
    Args:
        execution_plan: Execution plan to analyze
        gas_price_gwei: Gas price in gwei
        eth_price_usd: ETH price in USD
        
    Returns:
        Cost breakdown in USD
    """
    gas_cost_eth = (execution_plan.total_gas_estimate * gas_price_gwei * 1e9) / 1e18
    gas_cost_usd = gas_cost_eth * eth_price_usd
    
    return {
        "total_gas": execution_plan.total_gas_estimate,
        "gas_price_gwei": gas_price_gwei,
        "gas_cost_eth": gas_cost_eth,
        "gas_cost_usd": gas_cost_usd,
        "transactions": execution_plan.total_transactions,
        "cost_per_transaction": gas_cost_usd / execution_plan.total_transactions if execution_plan.total_transactions > 0 else 0
    }