"""
Transaction Analyzer for MEV Opportunity Detection.

This module analyzes blockchain transactions to determine their potential
impact on prices, liquidity, and MEV opportunities.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from decimal import Decimal
from enum import Enum

from web3 import Web3
from eth_abi import decode

logger = logging.getLogger(__name__)


class TransactionCategory(str, Enum):
    """Categories of transactions for MEV analysis."""
    DEX_TRADE = "dex_trade"             # DEX swap/trade
    ARBITRAGE = "arbitrage"             # Cross-DEX arbitrage
    LIQUIDATION = "liquidation"         # Lending protocol liquidation
    LENDING = "lending"                 # Lending/borrowing operation
    BRIDGE = "bridge"                   # Cross-chain bridge
    NFT_TRADE = "nft_trade"            # NFT marketplace trade
    DEFI_INTERACTION = "defi_interaction"  # General DeFi protocol interaction
    TOKEN_TRANSFER = "token_transfer"   # Simple token transfer
    CONTRACT_DEPLOYMENT = "contract_deployment"  # Contract deployment
    UNKNOWN = "unknown"                 # Unknown transaction type


@dataclass
class TokenFlow:
    """Represents token flow in a transaction."""
    token_address: str
    amount: Decimal
    direction: str  # 'in' or 'out'
    from_address: str
    to_address: str


@dataclass
class PoolImpact:
    """Represents impact on a liquidity pool."""
    pool_address: str
    protocol: str
    token_a: str
    token_b: str
    estimated_price_impact: float
    estimated_volume_usd: float
    liquidity_before: Optional[float] = None
    liquidity_after: Optional[float] = None


@dataclass
class TransactionImpact:
    """Analysis of a transaction's market impact."""
    transaction_hash: str
    category: TransactionCategory
    
    # Financial impact
    total_value_usd: float
    estimated_profit_usd: float = 0.0
    gas_cost_usd: float = 0.0
    
    # Token flows
    token_flows: List[TokenFlow] = field(default_factory=list)
    
    # Pool impacts
    affected_pools: List[PoolImpact] = field(default_factory=list)
    max_price_impact: float = 0.0
    
    # MEV analysis
    creates_arbitrage_opportunity: bool = False
    sandwich_vulnerable: bool = False
    liquidation_opportunity: bool = False
    
    # Execution timing
    time_sensitivity: float = 0.0  # 0-1 scale
    block_deadline: Optional[int] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class TransactionAnalyzer:
    """
    Analyzes blockchain transactions to determine their impact and MEV potential.
    
    This analyzer decodes transaction data, identifies protocol interactions,
    and calculates potential market impact and MEV opportunities.
    """
    
    def __init__(self, chain_id: int = 1):
        """Initialize transaction analyzer for specific chain."""
        self.chain_id = chain_id
        
        # Protocol contracts and function signatures
        self.protocol_contracts = self._load_protocol_contracts()
        self.function_signatures = self._load_function_signatures()
        
        # Price oracles and liquidity sources
        self.price_oracles = {}
        self.liquidity_sources = {}
        
        # Analysis configuration
        self.min_impact_threshold = 0.001  # 0.1% minimum impact
        self.large_trade_threshold = 100000  # $100k USD
        
    def _load_protocol_contracts(self) -> Dict[str, Dict[str, Any]]:
        """Load known protocol contracts and their details."""
        # Ethereum mainnet contracts
        ethereum_contracts = {
            # Uniswap V2
            "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": {
                "protocol": "uniswap_v2",
                "name": "UniswapV2Router02",
                "type": "router"
            },
            "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f": {
                "protocol": "uniswap_v2",
                "name": "UniswapV2Factory",
                "type": "factory"
            },
            
            # Uniswap V3
            "0xe592427a0aece92de3edee1f18e0157c05861564": {
                "protocol": "uniswap_v3",
                "name": "SwapRouter",
                "type": "router"
            },
            "0x1f98431c8ad98523631ae4a59f267346ea31f984": {
                "protocol": "uniswap_v3", 
                "name": "UniswapV3Factory",
                "type": "factory"
            },
            
            # Sushiswap
            "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f": {
                "protocol": "sushiswap",
                "name": "SushiswapRouter",
                "type": "router"
            },
            
            # 1inch
            "0x1111111254eeb25477b68fb85ed929f73a960582": {
                "protocol": "1inch",
                "name": "1inchV5Router",
                "type": "aggregator"
            },
            
            # Aave V3
            "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2": {
                "protocol": "aave_v3",
                "name": "AaveV3Pool",
                "type": "lending"
            },
            
            # Compound V3
            "0xc3d688b66703497daa19ca5fd3e5ddd1e7de9bef": {
                "protocol": "compound_v3",
                "name": "CompoundV3Comet",
                "type": "lending"
            }
        }
        
        return {
            1: ethereum_contracts,  # Ethereum mainnet
            # Add other chains as needed
        }
    
    def _load_function_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Load function signatures for transaction decoding."""
        return {
            # Uniswap V2 Router
            "0x38ed1739": {
                "name": "swapExactTokensForTokens",
                "protocol": "uniswap_v2",
                "category": TransactionCategory.DEX_TRADE,
                "mev_risk": 0.8
            },
            "0x7ff36ab5": {
                "name": "swapExactETHForTokens", 
                "protocol": "uniswap_v2",
                "category": TransactionCategory.DEX_TRADE,
                "mev_risk": 0.8
            },
            "0x18cbafe5": {
                "name": "swapExactTokensForETH",
                "protocol": "uniswap_v2", 
                "category": TransactionCategory.DEX_TRADE,
                "mev_risk": 0.8
            },
            
            # Uniswap V3 Router
            "0x414bf389": {
                "name": "exactInputSingle",
                "protocol": "uniswap_v3",
                "category": TransactionCategory.DEX_TRADE,
                "mev_risk": 0.7
            },
            "0xc04b8d59": {
                "name": "exactInput",
                "protocol": "uniswap_v3",
                "category": TransactionCategory.DEX_TRADE,
                "mev_risk": 0.7
            },
            
            # ERC20 Transfers
            "0xa9059cbb": {
                "name": "transfer",
                "protocol": "erc20",
                "category": TransactionCategory.TOKEN_TRANSFER,
                "mev_risk": 0.1
            },
            "0x23b872dd": {
                "name": "transferFrom",
                "protocol": "erc20",
                "category": TransactionCategory.TOKEN_TRANSFER,
                "mev_risk": 0.1
            },
            
            # Aave
            "0x617ba037": {
                "name": "supply",
                "protocol": "aave",
                "category": TransactionCategory.LENDING,
                "mev_risk": 0.3
            },
            "0x69328dec": {
                "name": "withdraw",
                "protocol": "aave", 
                "category": TransactionCategory.LENDING,
                "mev_risk": 0.3
            },
            "0x573ade81": {
                "name": "borrow",
                "protocol": "aave",
                "category": TransactionCategory.LENDING,
                "mev_risk": 0.4
            },
            "0x563dd613": {
                "name": "liquidationCall",
                "protocol": "aave",
                "category": TransactionCategory.LIQUIDATION,
                "mev_risk": 0.9
            },
            
            # 1inch Aggregator
            "0x12aa3caf": {
                "name": "swap",
                "protocol": "1inch",
                "category": TransactionCategory.ARBITRAGE,
                "mev_risk": 0.6
            }
        }
    
    async def analyze_transaction(self, tx_data: Dict[str, Any]) -> TransactionImpact:
        """
        Analyze a transaction and determine its market impact.
        
        Args:
            tx_data: Raw transaction data from Web3
            
        Returns:
            TransactionImpact analysis
        """
        tx_hash = tx_data.get('hash', '')
        logger.debug(f"Analyzing transaction: {tx_hash}")
        
        # Initialize impact analysis
        impact = TransactionImpact(
            transaction_hash=tx_hash,
            category=TransactionCategory.UNKNOWN,
            total_value_usd=0.0
        )
        
        try:
            # Basic transaction analysis
            await self._analyze_basic_properties(tx_data, impact)
            
            # Decode and analyze function call
            await self._analyze_function_call(tx_data, impact)
            
            # Analyze token flows
            await self._analyze_token_flows(tx_data, impact)
            
            # Calculate pool impacts
            await self._calculate_pool_impacts(tx_data, impact)
            
            # Assess MEV opportunities
            await self._assess_mev_opportunities(impact)
            
            # Calculate time sensitivity
            impact.time_sensitivity = self._calculate_time_sensitivity(impact)
            
        except Exception as e:
            logger.error(f"Error analyzing transaction {tx_hash}: {e}")
            impact.metadata["analysis_error"] = str(e)
        
        return impact
    
    async def _analyze_basic_properties(self, tx_data: Dict[str, Any], impact: TransactionImpact):
        """Analyze basic transaction properties."""
        # Calculate gas cost
        gas_price = tx_data.get('gasPrice', 0)
        gas_limit = tx_data.get('gas', 0)
        impact.gas_cost_usd = (gas_price * gas_limit) / 1e18 * 2000  # Assume $2000 ETH
        
        # Calculate transaction value
        value_wei = tx_data.get('value', 0)
        impact.total_value_usd = value_wei / 1e18 * 2000  # Assume $2000 ETH
        
        # Determine if contract interaction
        to_address = tx_data.get('to', '').lower()
        input_data = tx_data.get('input', '0x')
        
        if input_data and input_data != '0x':
            # Contract interaction
            impact.metadata["is_contract_interaction"] = True
            
            # Check if known protocol
            chain_contracts = self.protocol_contracts.get(self.chain_id, {})
            if to_address in chain_contracts:
                protocol_info = chain_contracts[to_address]
                impact.metadata["protocol"] = protocol_info["protocol"]
                impact.metadata["contract_type"] = protocol_info["type"]
        else:
            # Simple transfer
            impact.category = TransactionCategory.TOKEN_TRANSFER
            impact.metadata["is_contract_interaction"] = False
    
    async def _analyze_function_call(self, tx_data: Dict[str, Any], impact: TransactionImpact):
        """Analyze function call to determine transaction type."""
        input_data = tx_data.get('input', '0x')
        
        if not input_data or input_data == '0x':
            return
        
        # Extract function selector (first 4 bytes)
        function_selector = input_data[:10]  # 0x + 8 hex chars
        
        if function_selector in self.function_signatures:
            sig_info = self.function_signatures[function_selector]
            
            impact.category = sig_info["category"]
            impact.metadata["function_name"] = sig_info["name"]
            impact.metadata["protocol"] = sig_info["protocol"]
            impact.metadata["base_mev_risk"] = sig_info["mev_risk"]
            
            # Decode function parameters if possible
            try:
                decoded_params = await self._decode_function_params(
                    function_selector, input_data[10:]
                )
                impact.metadata["decoded_params"] = decoded_params
            except Exception as e:
                logger.debug(f"Could not decode function params: {e}")
    
    async def _decode_function_params(self, function_selector: str, param_data: str) -> Dict[str, Any]:
        """Decode function parameters for known functions."""
        # This is a simplified decoder - in production would use full ABI
        decoded = {}
        
        try:
            if function_selector == "0x38ed1739":  # swapExactTokensForTokens
                # Simplified decoding - would use proper ABI in production
                decoded["function"] = "swapExactTokensForTokens"
                decoded["estimated_params"] = "amount_in, amount_out_min, path, to, deadline"
            
            elif function_selector == "0xa9059cbb":  # ERC20 transfer
                if len(param_data) >= 128:  # 64 chars each for address and amount
                    to_address = "0x" + param_data[24:64]
                    amount_hex = param_data[64:128]
                    amount = int(amount_hex, 16)
                    
                    decoded["to"] = to_address
                    decoded["amount"] = amount
            
        except Exception as e:
            logger.debug(f"Error decoding params for {function_selector}: {e}")
        
        return decoded
    
    async def _analyze_token_flows(self, tx_data: Dict[str, Any], impact: TransactionImpact):
        """Analyze token flows in the transaction."""
        # This would analyze logs to determine actual token transfers
        # For now, we'll use simplified analysis based on function calls
        
        function_name = impact.metadata.get("function_name", "")
        
        if "swap" in function_name.lower():
            # DEX swap - estimate token flows
            if impact.total_value_usd > self.large_trade_threshold:
                impact.metadata["large_trade"] = True
                impact.max_price_impact = min(0.05, impact.total_value_usd / 1000000)  # Rough estimate
        
        elif function_name == "transfer":
            # Token transfer
            decoded_params = impact.metadata.get("decoded_params", {})
            if "amount" in decoded_params:
                amount = decoded_params["amount"]
                # Estimate USD value (would need token price data in production)
                impact.total_value_usd = max(impact.total_value_usd, amount / 1e18 * 100)  # Rough estimate
    
    async def _calculate_pool_impacts(self, tx_data: Dict[str, Any], impact: TransactionImpact):
        """Calculate impact on liquidity pools."""
        if impact.category == TransactionCategory.DEX_TRADE:
            protocol = impact.metadata.get("protocol", "unknown")
            
            # Create estimated pool impact
            pool_impact = PoolImpact(
                pool_address="unknown",  # Would determine from logs in production
                protocol=protocol,
                token_a="unknown",
                token_b="unknown", 
                estimated_price_impact=impact.max_price_impact,
                estimated_volume_usd=impact.total_value_usd
            )
            
            impact.affected_pools.append(pool_impact)
    
    async def _assess_mev_opportunities(self, impact: TransactionImpact):
        """Assess MEV opportunities created by the transaction."""
        base_mev_risk = impact.metadata.get("base_mev_risk", 0.0)
        
        # Large trades create more MEV opportunities
        if impact.total_value_usd > self.large_trade_threshold:
            base_mev_risk += 0.2
        
        # High price impact creates arbitrage opportunities
        if impact.max_price_impact > 0.01:  # >1% price impact
            impact.creates_arbitrage_opportunity = True
            base_mev_risk += 0.3
        
        # Vulnerable to sandwich attacks
        if impact.category == TransactionCategory.DEX_TRADE and impact.max_price_impact > 0.005:
            impact.sandwich_vulnerable = True
            base_mev_risk += 0.2
        
        # Liquidation opportunities
        if impact.category == TransactionCategory.LIQUIDATION:
            impact.liquidation_opportunity = True
            base_mev_risk = max(base_mev_risk, 0.9)
        
        impact.metadata["mev_risk_score"] = min(1.0, base_mev_risk)
    
    def _calculate_time_sensitivity(self, impact: TransactionImpact) -> float:
        """Calculate time sensitivity of the opportunity."""
        base_sensitivity = 0.0
        
        # DEX trades are time sensitive
        if impact.category == TransactionCategory.DEX_TRADE:
            base_sensitivity = 0.6
        
        # Arbitrage is highly time sensitive
        if impact.creates_arbitrage_opportunity:
            base_sensitivity = max(base_sensitivity, 0.8)
        
        # Liquidations are extremely time sensitive
        if impact.liquidation_opportunity:
            base_sensitivity = 0.9
        
        # Large trades are more time sensitive
        if impact.total_value_usd > self.large_trade_threshold:
            base_sensitivity += 0.2
        
        return min(1.0, base_sensitivity)
    
    def get_high_impact_transactions(
        self, 
        impacts: List[TransactionImpact],
        min_value_usd: float = 10000,
        min_mev_risk: float = 0.5
    ) -> List[TransactionImpact]:
        """Filter transactions with high MEV potential."""
        return [
            impact for impact in impacts
            if (impact.total_value_usd >= min_value_usd and 
                impact.metadata.get("mev_risk_score", 0) >= min_mev_risk)
        ]
    
    def get_arbitrage_opportunities(self, impacts: List[TransactionImpact]) -> List[TransactionImpact]:
        """Filter transactions that create arbitrage opportunities."""
        return [
            impact for impact in impacts
            if impact.creates_arbitrage_opportunity
        ]
    
    def get_sandwich_targets(self, impacts: List[TransactionImpact]) -> List[TransactionImpact]:
        """Filter transactions vulnerable to sandwich attacks."""
        return [
            impact for impact in impacts
            if impact.sandwich_vulnerable
        ]


# Convenience functions

def create_transaction_analyzer(chain_id: int = 1) -> TransactionAnalyzer:
    """Create a transaction analyzer for the specified chain."""
    return TransactionAnalyzer(chain_id)


async def analyze_transaction_batch(
    analyzer: TransactionAnalyzer,
    transactions: List[Dict[str, Any]]
) -> List[TransactionImpact]:
    """Analyze a batch of transactions."""
    impacts = []
    
    for tx_data in transactions:
        try:
            impact = await analyzer.analyze_transaction(tx_data)
            impacts.append(impact)
        except Exception as e:
            logger.error(f"Error analyzing transaction {tx_data.get('hash', 'unknown')}: {e}")
    
    return impacts