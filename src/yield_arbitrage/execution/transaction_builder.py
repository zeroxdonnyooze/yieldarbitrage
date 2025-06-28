"""Transaction builder for DeFi operations and arbitrage path execution."""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from enum import Enum

from eth_abi import encode
from eth_utils import to_checksum_address, to_wei

from ..graph_engine.models import YieldGraphEdge, EdgeType
from .tenderly_client import TenderlyTransaction

logger = logging.getLogger(__name__)


class TokenStandard(str, Enum):
    """Token standards."""
    ERC20 = "ERC20"
    NATIVE = "NATIVE"  # ETH, MATIC, etc.


@dataclass
class TokenInfo:
    """Token information for transaction building."""
    address: str
    symbol: str
    decimals: int
    standard: TokenStandard = TokenStandard.ERC20
    
    def format_amount(self, amount: Union[float, Decimal, str]) -> int:
        """Convert amount to token units."""
        if isinstance(amount, str):
            amount = Decimal(amount)
        elif isinstance(amount, float):
            amount = Decimal(str(amount))
        
        return int(amount * (10 ** self.decimals))


@dataclass
class SwapParams:
    """Parameters for a DEX swap operation."""
    token_in: TokenInfo
    token_out: TokenInfo
    amount_in: Union[float, Decimal, str]
    amount_out_min: Union[float, Decimal, str]
    recipient: str
    deadline: Optional[int] = None
    
    # Protocol-specific parameters
    fee_tier: Optional[int] = None  # For Uniswap V3
    pool_address: Optional[str] = None
    slippage_tolerance: float = 0.005  # 0.5%


class TransactionBuilder:
    """
    Builds transactions for DeFi operations.
    
    Converts high-level edge operations into executable transactions
    that can be simulated on Tenderly.
    """
    
    def __init__(self):
        """Initialize transaction builder."""
        # Common contract addresses (Ethereum mainnet)
        self.contracts = {
            "uniswap_v2_router": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            "uniswap_v3_router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
            "uniswap_v3_quoter": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
            "sushiswap_router": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
            "curve_registry": "0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5",
            "balancer_vault": "0xBA12222222228d8Ba445958a75a0704d566BF2C8",
            "aave_pool": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
            "compound_comptroller": "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B",
        }
        
        # Common token addresses
        self.tokens = {
            "WETH": TokenInfo("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "WETH", 18),
            "USDC": TokenInfo("0xA0b86a33E6417c5c6eF57e11d8CAf7d8C0f7C8F8", "USDC", 6),
            "USDT": TokenInfo("0xdAC17F958D2ee523a2206206994597C13D831ec7", "USDT", 6),
            "DAI": TokenInfo("0x6B175474E89094C44Da98b954EedeAC495271d0F", "DAI", 18),
            "WBTC": TokenInfo("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "WBTC", 8),
        }
        
        # Function signatures for common operations
        self.function_sigs = {
            # ERC20
            "transfer": "0xa9059cbb",
            "approve": "0x095ea7b3",
            "transferFrom": "0x23b872dd",
            
            # Uniswap V2
            "swapExactTokensForTokens": "0x38ed1739",
            "swapTokensForExactTokens": "0x8803dbee",
            "swapExactETHForTokens": "0x7ff36ab5",
            "swapExactTokensForETH": "0x18cbafe5",
            
            # Uniswap V3
            "exactInputSingle": "0x414bf389",
            "exactOutputSingle": "0xdb3e2198",
            
            # Aave
            "deposit": "0xe8eda9df",
            "withdraw": "0x69328dec",
            "borrow": "0xa415bcad",
            "repay": "0x573ade81",
            
            # Compound
            "mint": "0xa0712d68",
            "redeem": "0xdb006a75",
            "borrow": "0xc5ebeaec",
            "repayBorrow": "0x0e752702",
        }
    
    def build_edge_transaction(
        self,
        edge: YieldGraphEdge,
        input_amount: Union[float, Decimal, str],
        from_address: str,
        current_block: Optional[int] = None
    ) -> TenderlyTransaction:
        """
        Build a transaction for executing an edge.
        
        Args:
            edge: The edge to execute
            input_amount: Input amount for the edge
            from_address: Address executing the transaction
            current_block: Current block number for deadline calculation
            
        Returns:
            TenderlyTransaction ready for simulation
        """
        logger.debug(f"Building transaction for edge {edge.edge_id}")
        
        if edge.edge_type == EdgeType.TRADE:
            return self._build_trade_transaction(edge, input_amount, from_address, current_block)
        elif edge.edge_type == EdgeType.LEND:
            return self._build_lend_transaction(edge, input_amount, from_address)
        elif edge.edge_type == EdgeType.BORROW:
            return self._build_borrow_transaction(edge, input_amount, from_address)
        elif edge.edge_type == EdgeType.FLASH_LOAN:
            return self._build_flash_loan_transaction(edge, input_amount, from_address)
        elif edge.edge_type == EdgeType.BRIDGE:
            return self._build_bridge_transaction(edge, input_amount, from_address)
        else:
            raise ValueError(f"Unsupported edge type: {edge.edge_type}")
    
    def build_path_transactions(
        self,
        path: List[YieldGraphEdge],
        initial_amount: Union[float, Decimal, str],
        from_address: str,
        current_block: Optional[int] = None
    ) -> List[TenderlyTransaction]:
        """
        Build a sequence of transactions for a complete arbitrage path.
        
        Args:
            path: List of edges forming the path
            initial_amount: Initial amount to start with
            from_address: Address executing the transactions
            current_block: Current block number
            
        Returns:
            List of TenderlyTransactions for the complete path
        """
        transactions = []
        current_amount = initial_amount
        
        logger.info(f"Building transaction sequence for {len(path)} edges")
        
        for i, edge in enumerate(path):
            logger.debug(f"Building transaction {i+1}/{len(path)}: {edge.edge_id}")
            
            # Build transaction for this edge
            tx = self.build_edge_transaction(edge, current_amount, from_address, current_block)
            transactions.append(tx)
            
            # Estimate output amount for next iteration
            # This is a rough estimate - actual amounts will be determined by simulation
            if hasattr(edge, 'estimated_output'):
                current_amount = edge.estimated_output
            else:
                # Use a simple heuristic
                current_amount = current_amount  # Pass through for now
        
        return transactions
    
    def _build_trade_transaction(
        self,
        edge: YieldGraphEdge,
        input_amount: Union[float, Decimal, str],
        from_address: str,
        current_block: Optional[int] = None
    ) -> TenderlyTransaction:
        """Build transaction for a DEX trade."""
        protocol = edge.protocol_name.lower()
        
        if "uniswap" in protocol:
            return self._build_uniswap_transaction(edge, input_amount, from_address, current_block)
        elif "sushiswap" in protocol:
            return self._build_sushiswap_transaction(edge, input_amount, from_address, current_block)
        elif "curve" in protocol:
            return self._build_curve_transaction(edge, input_amount, from_address)
        elif "balancer" in protocol:
            return self._build_balancer_transaction(edge, input_amount, from_address)
        else:
            # Generic DEX transaction
            return self._build_generic_dex_transaction(edge, input_amount, from_address, current_block)
    
    def _build_uniswap_transaction(
        self,
        edge: YieldGraphEdge,
        input_amount: Union[float, Decimal, str],
        from_address: str,
        current_block: Optional[int] = None
    ) -> TenderlyTransaction:
        """Build Uniswap swap transaction."""
        # Determine if V2 or V3
        is_v3 = "v3" in edge.protocol_name.lower()
        
        # Get token information
        token_in = self._get_token_info(edge.source_asset_id)
        token_out = self._get_token_info(edge.target_asset_id)
        
        if not token_in or not token_out:
            raise ValueError(f"Unknown tokens for edge {edge.edge_id}")
        
        # Format amounts
        amount_in = token_in.format_amount(input_amount)
        amount_out_min = 0  # Will be calculated based on slippage
        
        # Calculate deadline
        deadline = self._calculate_deadline(current_block)
        
        if is_v3:
            return self._build_uniswap_v3_transaction(
                token_in, token_out, amount_in, amount_out_min, from_address, deadline
            )
        else:
            return self._build_uniswap_v2_transaction(
                token_in, token_out, amount_in, amount_out_min, from_address, deadline
            )
    
    def _build_uniswap_v2_transaction(
        self,
        token_in: TokenInfo,
        token_out: TokenInfo,
        amount_in: int,
        amount_out_min: int,
        from_address: str,
        deadline: int
    ) -> TenderlyTransaction:
        """Build Uniswap V2 swap transaction."""
        router_address = self.contracts["uniswap_v2_router"]
        
        # Handle ETH/WETH specially
        if token_in.symbol == "ETH" or token_in.address == self.tokens["WETH"].address:
            if token_out.symbol in ["USDC", "USDT", "DAI"]:
                # ETH -> Token
                function_sig = self.function_sigs["swapExactETHForTokens"]
                
                # Encode parameters: amountOutMin, path[], to, deadline
                path = [self.tokens["WETH"].address, token_out.address]
                data = function_sig + encode(
                    ["uint256", "address[]", "address", "uint256"],
                    [amount_out_min, path, from_address, deadline]
                ).hex()
                
                return TenderlyTransaction(
                    from_address=from_address,
                    to_address=router_address,
                    value=str(amount_in),  # ETH value
                    data=data
                )
        
        # Standard token -> token swap
        function_sig = self.function_sigs["swapExactTokensForTokens"]
        path = [token_in.address, token_out.address]
        
        # Check if direct path exists, otherwise route through WETH
        if token_in.symbol not in ["WETH", "ETH"] and token_out.symbol not in ["WETH", "ETH"]:
            path = [token_in.address, self.tokens["WETH"].address, token_out.address]
        
        data = function_sig + encode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [amount_in, amount_out_min, path, from_address, deadline]
        ).hex()
        
        return TenderlyTransaction(
            from_address=from_address,
            to_address=router_address,
            data=data
        )
    
    def _build_uniswap_v3_transaction(
        self,
        token_in: TokenInfo,
        token_out: TokenInfo,
        amount_in: int,
        amount_out_min: int,
        from_address: str,
        deadline: int
    ) -> TenderlyTransaction:
        """Build Uniswap V3 swap transaction."""
        router_address = self.contracts["uniswap_v3_router"]
        function_sig = self.function_sigs["exactInputSingle"]
        
        # Default fee tier (0.3%)
        fee = 3000
        
        # ExactInputSingleParams struct
        params = [
            token_in.address,     # tokenIn
            token_out.address,    # tokenOut
            fee,                  # fee
            from_address,         # recipient
            deadline,             # deadline
            amount_in,            # amountIn
            amount_out_min,       # amountOutMinimum
            0                     # sqrtPriceLimitX96
        ]
        
        data = function_sig + encode(
            ["(address,address,uint24,address,uint256,uint256,uint256,uint160)"],
            [tuple(params)]
        ).hex()
        
        return TenderlyTransaction(
            from_address=from_address,
            to_address=router_address,
            data=data
        )
    
    def _build_sushiswap_transaction(
        self,
        edge: YieldGraphEdge,
        input_amount: Union[float, Decimal, str],
        from_address: str,
        current_block: Optional[int] = None
    ) -> TenderlyTransaction:
        """Build SushiSwap transaction (similar to Uniswap V2)."""
        # SushiSwap uses same interface as Uniswap V2
        token_in = self._get_token_info(edge.source_asset_id)
        token_out = self._get_token_info(edge.target_asset_id)
        
        amount_in = token_in.format_amount(input_amount)
        deadline = self._calculate_deadline(current_block)
        
        router_address = self.contracts["sushiswap_router"]
        function_sig = self.function_sigs["swapExactTokensForTokens"]
        path = [token_in.address, token_out.address]
        
        data = function_sig + encode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [amount_in, 0, path, from_address, deadline]
        ).hex()
        
        return TenderlyTransaction(
            from_address=from_address,
            to_address=router_address,
            data=data
        )
    
    def _build_curve_transaction(
        self,
        edge: YieldGraphEdge,
        input_amount: Union[float, Decimal, str],
        from_address: str
    ) -> TenderlyTransaction:
        """Build Curve swap transaction."""
        # This is a simplified implementation
        # In practice, you'd need to query the Curve registry for pool addresses
        
        # For now, return a generic transaction
        return TenderlyTransaction(
            from_address=from_address,
            to_address="0x0000000000000000000000000000000000000000",  # Placeholder
            data="0x"
        )
    
    def _build_balancer_transaction(
        self,
        edge: YieldGraphEdge,
        input_amount: Union[float, Decimal, str],
        from_address: str
    ) -> TenderlyTransaction:
        """Build Balancer swap transaction."""
        # Simplified Balancer implementation
        vault_address = self.contracts["balancer_vault"]
        
        return TenderlyTransaction(
            from_address=from_address,
            to_address=vault_address,
            data="0x"
        )
    
    def _build_generic_dex_transaction(
        self,
        edge: YieldGraphEdge,
        input_amount: Union[float, Decimal, str],
        from_address: str,
        current_block: Optional[int] = None
    ) -> TenderlyTransaction:
        """Build generic DEX transaction."""
        return TenderlyTransaction(
            from_address=from_address,
            to_address="0x0000000000000000000000000000000000000000",
            data="0x"
        )
    
    def _build_lend_transaction(
        self,
        edge: YieldGraphEdge,
        input_amount: Union[float, Decimal, str],
        from_address: str
    ) -> TenderlyTransaction:
        """Build lending transaction."""
        protocol = edge.protocol_name.lower()
        
        if "aave" in protocol:
            return self._build_aave_deposit_transaction(edge, input_amount, from_address)
        elif "compound" in protocol:
            return self._build_compound_mint_transaction(edge, input_amount, from_address)
        else:
            raise ValueError(f"Unsupported lending protocol: {edge.protocol_name}")
    
    def _build_aave_deposit_transaction(
        self,
        edge: YieldGraphEdge,
        input_amount: Union[float, Decimal, str],
        from_address: str
    ) -> TenderlyTransaction:
        """Build Aave deposit transaction."""
        pool_address = self.contracts["aave_pool"]
        token = self._get_token_info(edge.source_asset_id)
        amount = token.format_amount(input_amount)
        
        function_sig = self.function_sigs["deposit"]
        data = function_sig + encode(
            ["address", "uint256", "address", "uint16"],
            [token.address, amount, from_address, 0]  # 0 = no referral code
        ).hex()
        
        return TenderlyTransaction(
            from_address=from_address,
            to_address=pool_address,
            data=data
        )
    
    def _build_compound_mint_transaction(
        self,
        edge: YieldGraphEdge,
        input_amount: Union[float, Decimal, str],
        from_address: str
    ) -> TenderlyTransaction:
        """Build Compound mint transaction."""
        # This would need the cToken address for the specific asset
        ctoken_address = "0x0000000000000000000000000000000000000000"  # Placeholder
        
        token = self._get_token_info(edge.source_asset_id)
        amount = token.format_amount(input_amount)
        
        function_sig = self.function_sigs["mint"]
        data = function_sig + encode(["uint256"], [amount]).hex()
        
        return TenderlyTransaction(
            from_address=from_address,
            to_address=ctoken_address,
            data=data
        )
    
    def _build_borrow_transaction(
        self,
        edge: YieldGraphEdge,
        input_amount: Union[float, Decimal, str],
        from_address: str
    ) -> TenderlyTransaction:
        """Build borrowing transaction."""
        # Simplified implementation
        return TenderlyTransaction(
            from_address=from_address,
            to_address="0x0000000000000000000000000000000000000000",
            data="0x"
        )
    
    def _build_flash_loan_transaction(
        self,
        edge: YieldGraphEdge,
        input_amount: Union[float, Decimal, str],
        from_address: str
    ) -> TenderlyTransaction:
        """Build flash loan transaction."""
        # Flash loans require complex callback logic
        return TenderlyTransaction(
            from_address=from_address,
            to_address="0x0000000000000000000000000000000000000000",
            data="0x"
        )
    
    def _build_bridge_transaction(
        self,
        edge: YieldGraphEdge,
        input_amount: Union[float, Decimal, str],
        from_address: str
    ) -> TenderlyTransaction:
        """Build bridge transaction."""
        # Bridge transactions are protocol-specific
        return TenderlyTransaction(
            from_address=from_address,
            to_address="0x0000000000000000000000000000000000000000",
            data="0x"
        )
    
    def _get_token_info(self, asset_id: str) -> Optional[TokenInfo]:
        """Get token information from asset ID."""
        # Extract symbol from asset ID (e.g., "ETH_MAINNET_WETH" -> "WETH")
        parts = asset_id.split("_")
        if len(parts) >= 3:
            symbol = parts[-1]
            return self.tokens.get(symbol)
        
        return None
    
    def _calculate_deadline(self, current_block: Optional[int] = None) -> int:
        """Calculate transaction deadline."""
        import time
        
        # Default to 20 minutes from now
        return int(time.time()) + 1200
    
    def build_approval_transaction(
        self,
        token_address: str,
        spender_address: str,
        amount: Union[int, str],
        from_address: str
    ) -> TenderlyTransaction:
        """
        Build ERC20 approval transaction.
        
        Args:
            token_address: Token contract address
            spender_address: Address to approve
            amount: Amount to approve (use 2^256-1 for unlimited)
            from_address: Owner address
            
        Returns:
            TenderlyTransaction for the approval
        """
        function_sig = self.function_sigs["approve"]
        
        if isinstance(amount, str) and amount == "unlimited":
            amount = 2**256 - 1
        
        data = function_sig + encode(
            ["address", "uint256"],
            [spender_address, int(amount)]
        ).hex()
        
        return TenderlyTransaction(
            from_address=from_address,
            to_address=token_address,
            data=data
        )