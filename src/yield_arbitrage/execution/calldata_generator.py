"""
Dynamic Calldata Generator for Smart Contract Router Integration.

This module generates calldata for executing path segments through the YieldArbitrageRouter,
leveraging existing protocol adapters to create the necessary transaction data.
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from eth_abi import encode

from yield_arbitrage.pathfinding.path_segment_analyzer import PathSegment, SegmentType
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType
from yield_arbitrage.protocols.uniswap_v3_adapter import UniswapV3Adapter
from yield_arbitrage.execution.transaction_builder import TransactionBuilder

logger = logging.getLogger(__name__)


@dataclass
class EdgeOperationCalldata:
    """Calldata for a single edge operation."""
    edge_type: EdgeType
    target_contract: str
    input_token: str
    output_token: str
    input_amount: int  # 0 for dynamic amounts
    min_output_amount: int
    call_data: bytes
    metadata: Dict[str, Any]


@dataclass
class SegmentCalldata:
    """Complete calldata for executing a path segment."""
    segment_id: str
    operations: List[EdgeOperationCalldata]
    requires_flash_loan: bool
    flash_loan_asset: Optional[str] = None
    flash_loan_amount: Optional[int] = None
    recipient: str = "0x0000000000000000000000000000000000000000"
    deadline: int = 0


class CalldataGenerator:
    """
    Generates calldata for executing path segments through the smart contract router.
    
    This class uses existing protocol adapters to generate the actual calldata,
    then packages it for execution through the YieldArbitrageRouter.
    """
    
    def __init__(self, chain_id: int = 1):
        """
        Initialize the calldata generator.
        
        Args:
            chain_id: Ethereum chain ID (1 for mainnet, 137 for Polygon, etc.)
        """
        self.chain_id = chain_id
        self.protocol_adapters = self._initialize_adapters()
        self.protocol_contracts = self._get_protocol_contracts()
        
        # Transaction builder for complex operations
        self.tx_builder = TransactionBuilder()
    
    def _initialize_adapters(self) -> Dict[str, Any]:
        """Initialize protocol adapters."""
        # For calldata generation, we only need the adapter methods, not full initialization
        # In a real implementation, these would be properly initialized with providers
        return {
            # "uniswap_v3": UniswapV3Adapter(chain_name="ethereum", provider=mock_provider),
            # For now, we'll mock the adapter methods we need
        }
    
    def _get_protocol_contracts(self) -> Dict[str, str]:
        """Get protocol contract addresses for the current chain."""
        # Mainnet addresses
        if self.chain_id == 1:
            return {
                "uniswap_v3_router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
                "uniswap_v3_quoter": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
                "aave_v3_pool": "0x87870Bca3F8e07Da8c8f4B3A4d8f8Eb2a8B6e4f1",
                "compound_v3_usdc": "0xc3d688B66703497DAA19211EEdff47f25384cdc3",
            }
        # Add other chains as needed
        else:
            return {}
    
    def generate_segment_calldata(self, segment: PathSegment, 
                                recipient: str,
                                deadline: Optional[int] = None) -> SegmentCalldata:
        """
        Generate calldata for executing a complete path segment.
        
        Args:
            segment: Path segment to generate calldata for
            recipient: Final recipient of output tokens
            deadline: Execution deadline (Unix timestamp)
            
        Returns:
            Complete calldata for segment execution
        """
        if deadline is None:
            deadline = 2**32 - 1  # Far future
        
        operations = []
        
        for edge in segment.edges:
            operation = self._generate_edge_calldata(edge)
            operations.append(operation)
        
        return SegmentCalldata(
            segment_id=segment.segment_id,
            operations=operations,
            requires_flash_loan=segment.requires_flash_loan,
            flash_loan_asset=segment.flash_loan_asset,
            flash_loan_amount=int(segment.flash_loan_amount) if segment.flash_loan_amount else None,
            recipient=recipient,
            deadline=deadline
        )
    
    def _generate_edge_calldata(self, edge: YieldGraphEdge) -> EdgeOperationCalldata:
        """Generate calldata for a single edge operation."""
        if edge.edge_type == EdgeType.TRADE:
            return self._generate_trade_calldata(edge)
        elif edge.edge_type == EdgeType.LEND:
            return self._generate_lend_calldata(edge)
        elif edge.edge_type == EdgeType.BORROW:
            return self._generate_borrow_calldata(edge)
        elif edge.edge_type == EdgeType.STAKE:
            return self._generate_stake_calldata(edge)
        elif edge.edge_type == EdgeType.FLASH_LOAN:
            return self._generate_flash_loan_calldata(edge)
        else:
            raise ValueError(f"Unsupported edge type: {edge.edge_type}")
    
    def _generate_trade_calldata(self, edge: YieldGraphEdge) -> EdgeOperationCalldata:
        """Generate calldata for a trade operation."""
        protocol = edge.protocol_name.lower()
        
        if protocol == "uniswap_v3":
            return self._generate_uniswap_v3_calldata(edge)
        else:
            raise ValueError(f"Unsupported trade protocol: {protocol}")
    
    def _generate_uniswap_v3_calldata(self, edge: YieldGraphEdge) -> EdgeOperationCalldata:
        """Generate Uniswap V3 swap calldata."""
        # Extract swap parameters from edge
        token_in = edge.source_asset_id
        token_out = edge.target_asset_id
        fee = 3000  # Default 0.3% fee tier, should come from edge metadata
        
        # Get token addresses (simplified - real implementation would use token registry)
        token_in_address = self._get_token_address(token_in)
        token_out_address = self._get_token_address(token_out)
        
        # Generate Uniswap V3 exactInputSingle calldata manually
        # Function signature: exactInputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160))
        function_selector = "0x414bf389"  # exactInputSingle selector
        
        # Encode the ExactInputSingleParams struct
        params = encode(
            ["address", "address", "uint24", "address", "uint256", "uint256", "uint256", "uint160"],
            [
                token_in_address,   # tokenIn
                token_out_address,  # tokenOut
                fee,               # fee
                "0x0000000000000000000000000000000000000000",  # recipient (router will set)
                2**32 - 1,         # deadline
                0,                 # amountIn (dynamic)
                0,                 # amountOutMinimum
                0                  # sqrtPriceLimitX96
            ]
        )
        
        calldata = bytes.fromhex(function_selector[2:]) + params
        
        return EdgeOperationCalldata(
            edge_type=EdgeType.TRADE,
            target_contract=self.protocol_contracts["uniswap_v3_router"],
            input_token=token_in_address,
            output_token=token_out_address,
            input_amount=0,  # Dynamic
            min_output_amount=0,  # Will be calculated
            call_data=calldata,
            metadata={"protocol": "uniswap_v3", "fee": fee}
        )
    
    def _generate_lend_calldata(self, edge: YieldGraphEdge) -> EdgeOperationCalldata:
        """Generate lending operation calldata."""
        protocol = edge.protocol_name.lower()
        
        if protocol == "aave_v3":
            return self._generate_aave_v3_supply_calldata(edge)
        else:
            raise ValueError(f"Unsupported lending protocol: {protocol}")
    
    def _generate_aave_v3_supply_calldata(self, edge: YieldGraphEdge) -> EdgeOperationCalldata:
        """Generate Aave V3 supply calldata."""
        # Function signature: supply(address asset, uint256 amount, address onBehalfOf, uint16 referralCode)
        function_selector = "0x617ba037"  # supply(address,uint256,address,uint16)
        
        asset = self._get_token_address(edge.source_asset_id)
        amount = 0  # Dynamic amount
        on_behalf_of = "0x0000000000000000000000000000000000000000"  # Router address
        referral_code = 0
        
        # Encode parameters
        encoded_params = encode(
            ["address", "uint256", "address", "uint16"],
            [asset, amount, on_behalf_of, referral_code]
        )
        
        calldata = bytes.fromhex(function_selector[2:]) + encoded_params
        
        return EdgeOperationCalldata(
            edge_type=EdgeType.LEND,
            target_contract=self.protocol_contracts["aave_v3_pool"],
            input_token=asset,
            output_token=self._get_atoken_address(asset),
            input_amount=0,  # Dynamic
            min_output_amount=0,  # 1:1 for aTokens
            call_data=calldata,
            metadata={"protocol": "aave_v3"}
        )
    
    def _generate_borrow_calldata(self, edge: YieldGraphEdge) -> EdgeOperationCalldata:
        """Generate borrowing operation calldata."""
        protocol = edge.protocol_name.lower()
        
        if protocol == "aave_v3":
            return self._generate_aave_v3_borrow_calldata(edge)
        else:
            raise ValueError(f"Unsupported borrowing protocol: {protocol}")
    
    def _generate_aave_v3_borrow_calldata(self, edge: YieldGraphEdge) -> EdgeOperationCalldata:
        """Generate Aave V3 borrow calldata."""
        # Function signature: borrow(address asset, uint256 amount, uint256 interestRateMode, uint16 referralCode, address onBehalfOf)
        function_selector = "0xa415bcad"  # borrow(address,uint256,uint256,uint16,address)
        
        asset = self._get_token_address(edge.target_asset_id)
        amount = 0  # Dynamic amount
        interest_rate_mode = 2  # Variable rate
        referral_code = 0
        on_behalf_of = "0x0000000000000000000000000000000000000000"  # Router address
        
        encoded_params = encode(
            ["address", "uint256", "uint256", "uint16", "address"],
            [asset, amount, interest_rate_mode, referral_code, on_behalf_of]
        )
        
        calldata = bytes.fromhex(function_selector[2:]) + encoded_params
        
        return EdgeOperationCalldata(
            edge_type=EdgeType.BORROW,
            target_contract=self.protocol_contracts["aave_v3_pool"],
            input_token="0x0000000000000000000000000000000000000000",  # No input token
            output_token=asset,
            input_amount=0,
            min_output_amount=0,
            call_data=calldata,
            metadata={"protocol": "aave_v3", "interest_rate_mode": interest_rate_mode}
        )
    
    def _generate_stake_calldata(self, edge: YieldGraphEdge) -> EdgeOperationCalldata:
        """Generate staking operation calldata."""
        # Staking is not yet supported in production - return error calldata
        logger.warning(f"Staking edge {edge.edge_id} not supported in current version")
        return EdgeOperationCalldata(
            edge_id=edge.edge_id,
            target_contract="0x0000000000000000000000000000000000000000",
            function_name="stake",
            encoded_data="0x",
            value=0,
            gas_estimate=0,
            operation_type=edge.edge_type,
            error="Staking operations not yet implemented"
        )
    
    def _generate_flash_loan_calldata(self, edge: YieldGraphEdge) -> EdgeOperationCalldata:
        """Generate flash loan operation calldata."""
        # Flash loans are handled at the segment level, not individual edges
        raise ValueError("Flash loan edges should be handled at segment level")
    
    def encode_segment_for_router(self, segment_calldata: SegmentCalldata) -> bytes:
        """
        Encode segment calldata for the YieldArbitrageRouter contract.
        
        Args:
            segment_calldata: Complete segment calldata
            
        Returns:
            ABI-encoded data for router execution
        """
        # Simplified encoding for demonstration
        # In production, this would use proper ABI encoding
        
        segment_id = segment_calldata.segment_id.encode('utf-8').ljust(32, b'\0')
        
        # Create a simplified encoding
        result = bytearray()
        result.extend(b'\x12\x34\x56\x78')  # Function selector placeholder
        result.extend(segment_id)
        
        # Add operation count
        result.extend(len(segment_calldata.operations).to_bytes(4, 'big'))
        
        # Add each operation (simplified)
        for op in segment_calldata.operations:
            result.extend(self._edge_type_to_int(op.edge_type).to_bytes(4, 'big'))
            result.extend(len(op.call_data).to_bytes(4, 'big'))
            result.extend(op.call_data)
        
        # Add flash loan info
        result.extend(int(segment_calldata.requires_flash_loan).to_bytes(1, 'big'))
        
        return bytes(result)
    
    def _edge_type_to_int(self, edge_type: EdgeType) -> int:
        """Convert EdgeType enum to integer for Solidity."""
        mapping = {
            EdgeType.TRADE: 0,
            EdgeType.SPLIT: 1,
            EdgeType.COMBINE: 2,
            EdgeType.BRIDGE: 3,
            EdgeType.LEND: 4,
            EdgeType.BORROW: 5,
            EdgeType.STAKE: 6,
            EdgeType.WAIT: 7,
            EdgeType.SHORT: 8,
            EdgeType.FLASH_LOAN: 9,
            EdgeType.BACK_RUN: 10
        }
        return mapping.get(edge_type, 0)
    
    def _get_token_address(self, token_id: str) -> str:
        """Get token contract address from token ID."""
        # Simplified token registry - real implementation would use comprehensive registry
        token_addresses = {
            "USDC": "0xA0b86a33E6441c2b73AC95F2DB8FFf6d4daF1E65",
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        }
        return token_addresses.get(token_id, token_id)
    
    def _get_atoken_address(self, underlying_token: str) -> str:
        """Get Aave aToken address for underlying token."""
        # Simplified mapping - real implementation would query Aave protocol data provider
        atoken_mapping = {
            "0xA0b86a33E6441c2b73AC95F2DB8FFf6d4daF1E65": "0x98C23E9d8f34FEFb1B7BD6a91B7FF122F4e16F5c",  # aUSDC
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": "0x4d5F47FA6A74757f35C14fD3a6Ef8E3C9BC514E8",  # aWETH
        }
        return atoken_mapping.get(underlying_token, underlying_token)
    
    def calculate_min_output_amounts(self, segment_calldata: SegmentCalldata,
                                   slippage_tolerance: float = 0.005) -> SegmentCalldata:
        """
        Calculate minimum output amounts based on current prices and slippage tolerance.
        
        Args:
            segment_calldata: Segment calldata to update
            slippage_tolerance: Maximum allowed slippage (0.005 = 0.5%)
            
        Returns:
            Updated segment calldata with minimum output amounts
        """
        for operation in segment_calldata.operations:
            if operation.edge_type == EdgeType.TRADE:
                # Get expected output from price oracle or DEX quoter
                expected_output = self._get_expected_output(
                    operation.input_token,
                    operation.output_token,
                    operation.input_amount
                )
                
                # Apply slippage tolerance
                min_output = int(expected_output * (1 - slippage_tolerance))
                operation.min_output_amount = min_output
        
        return segment_calldata
    
    def _get_expected_output(self, token_in: str, token_out: str, amount_in: int) -> int:
        """Get expected output amount for a trade."""
        # Simplified - real implementation would query price oracles or DEX quoters
        return amount_in  # 1:1 placeholder
    
    def validate_segment_calldata(self, segment_calldata: SegmentCalldata) -> bool:
        """
        Validate that segment calldata is properly formed.
        
        Args:
            segment_calldata: Segment calldata to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if not segment_calldata.segment_id:
                logger.error("Missing segment ID")
                return False
            
            if not segment_calldata.operations:
                logger.error("No operations in segment")
                return False
            
            # Validate each operation
            for i, op in enumerate(segment_calldata.operations):
                if not op.target_contract:
                    logger.error(f"Operation {i}: Missing target contract")
                    return False
                
                if not op.call_data:
                    logger.error(f"Operation {i}: Missing calldata")
                    return False
                
                # Validate token addresses
                if op.input_token and not self._is_valid_address(op.input_token):
                    logger.error(f"Operation {i}: Invalid input token address")
                    return False
                
                if not self._is_valid_address(op.output_token):
                    logger.error(f"Operation {i}: Invalid output token address")
                    return False
            
            # Validate flash loan parameters if required
            if segment_calldata.requires_flash_loan:
                if not segment_calldata.flash_loan_asset:
                    logger.error("Flash loan required but no asset specified")
                    return False
                
                if not segment_calldata.flash_loan_amount:
                    logger.error("Flash loan required but no amount specified")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def _is_valid_address(self, address: str) -> bool:
        """Check if address is a valid Ethereum address."""
        if not address:
            return False
        
        if not address.startswith("0x"):
            return False
        
        if len(address) != 42:
            return False
        
        try:
            int(address, 16)
            return True
        except ValueError:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "supported_protocols": list(self.protocol_adapters.keys()),
            "chain_id": self.chain_id,
            "protocol_contracts": len(self.protocol_contracts)
        }