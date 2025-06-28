"""Production protocol registry with real mainnet DeFi protocol configurations."""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ProtocolCategory(str, Enum):
    """Categories of DeFi protocols."""
    DEX_SPOT = "dex_spot"           # Spot trading DEXs
    DEX_PERP = "dex_perp"           # Perpetual/derivatives DEXs
    LENDING = "lending"             # Lending protocols
    YIELD_FARMING = "yield_farming" # Yield farming protocols
    LIQUID_STAKING = "liquid_staking" # Liquid staking protocols
    CDP = "cdp"                     # Collateralized debt positions
    AGGREGATOR = "aggregator"       # DEX aggregators
    BRIDGE = "bridge"               # Cross-chain bridges


@dataclass
class ContractInfo:
    """Information about a protocol contract."""
    address: str
    name: str
    abi_name: str  # Reference to ABI in abi_manager
    deployment_block: Optional[int] = None
    is_proxy: bool = False
    implementation_address: Optional[str] = None


@dataclass
class ProtocolConfig:
    """Production configuration for a DeFi protocol."""
    protocol_id: str
    name: str
    category: ProtocolCategory
    description: str
    website: str
    docs_url: str
    
    # Chain support
    supported_chains: List[str]
    
    # Contract configurations per chain
    contracts: Dict[str, Dict[str, ContractInfo]]  # chain -> contract_name -> info
    
    # Protocol-specific settings
    settings: Dict[str, Any]
    
    # Integration metadata
    adapter_class_name: str
    
    # Risk parameters
    risk_level: str = "medium"  # "low", "medium", "high"
    tvl_usd: Optional[float] = None
    audit_status: Optional[str] = None
    
    # Operational flags
    is_enabled: bool = True
    supports_flash_loans: bool = False
    supports_batch_operations: bool = False
    
    # Metadata
    last_updated: Optional[str] = None


class ProductionProtocolRegistry:
    """
    Production registry of real DeFi protocols with mainnet configurations.
    
    This registry contains verified, production-ready protocol configurations
    including real contract addresses, ABIs, and operational parameters.
    """
    
    def __init__(self):
        """Initialize the production protocol registry."""
        self.protocols: Dict[str, ProtocolConfig] = {}
        self._initialize_mainnet_protocols()
    
    def _initialize_mainnet_protocols(self) -> None:
        """Initialize registry with real mainnet protocol configurations."""
        
        # Uniswap V3 - Highest liquidity DEX
        self.protocols["uniswap_v3"] = ProtocolConfig(
            protocol_id="uniswap_v3",
            name="Uniswap V3",
            category=ProtocolCategory.DEX_SPOT,
            description="Concentrated liquidity automated market maker",
            website="https://uniswap.org",
            docs_url="https://docs.uniswap.org",
            supported_chains=["ethereum", "arbitrum", "base"],
            contracts={
                "ethereum": {
                    "factory": ContractInfo(
                        address="0x1F98431c8aD98523631AE4a59f267346ea31F984",
                        name="UniswapV3Factory",
                        abi_name="uniswap_v3_factory",
                        deployment_block=12369621
                    ),
                    "swap_router": ContractInfo(
                        address="0xE592427A0AEce92De3Edee1F18E0157C05861564",
                        name="SwapRouter",
                        abi_name="uniswap_v3_router",
                        deployment_block=12369621
                    ),
                    "quoter": ContractInfo(
                        address="0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
                        name="Quoter",
                        abi_name="uniswap_v3_quoter",
                        deployment_block=12369621
                    ),
                    "nft_manager": ContractInfo(
                        address="0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
                        name="NonfungiblePositionManager",
                        abi_name="uniswap_v3_nft_manager",
                        deployment_block=12369621
                    )
                },
                "arbitrum": {
                    "factory": ContractInfo(
                        address="0x1F98431c8aD98523631AE4a59f267346ea31F984",
                        name="UniswapV3Factory",
                        abi_name="uniswap_v3_factory"
                    ),
                    "swap_router": ContractInfo(
                        address="0xE592427A0AEce92De3Edee1F18E0157C05861564",
                        name="SwapRouter",
                        abi_name="uniswap_v3_router"
                    )
                },
                "base": {
                    "factory": ContractInfo(
                        address="0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
                        name="UniswapV3Factory", 
                        abi_name="uniswap_v3_factory"
                    ),
                    "swap_router": ContractInfo(
                        address="0x2626664c2603336E57B271c5C0b26F421741e481",
                        name="SwapRouter",
                        abi_name="uniswap_v3_router"
                    )
                }
            },
            settings={
                "fee_tiers": [100, 500, 3000, 10000],  # 0.01%, 0.05%, 0.3%, 1%
                "tick_spacing": {100: 1, 500: 10, 3000: 60, 10000: 200},
                "max_tick": 887272,
                "min_tick": -887272
            },
            risk_level="low",
            tvl_usd=3_500_000_000,  # ~$3.5B TVL
            audit_status="audited_multiple",
            supports_flash_loans=False,
            supports_batch_operations=True,
            adapter_class_name="UniswapV3Adapter",
            last_updated="2024-06-22"
        )
        
        # Uniswap V2 - Original AMM
        self.protocols["uniswap_v2"] = ProtocolConfig(
            protocol_id="uniswap_v2",
            name="Uniswap V2",
            category=ProtocolCategory.DEX_SPOT,
            description="Constant product automated market maker",
            website="https://uniswap.org",
            docs_url="https://docs.uniswap.org/contracts/v2/overview",
            supported_chains=["ethereum"],
            contracts={
                "ethereum": {
                    "factory": ContractInfo(
                        address="0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
                        name="UniswapV2Factory",
                        abi_name="uniswap_v2_factory",
                        deployment_block=10000835
                    ),
                    "router": ContractInfo(
                        address="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                        name="UniswapV2Router02",
                        abi_name="uniswap_v2_router",
                        deployment_block=10207858
                    )
                }
            },
            settings={
                "fee": 0.003,  # 0.3%
                "min_liquidity": 1000
            },
            risk_level="low",
            tvl_usd=1_200_000_000,  # ~$1.2B TVL
            audit_status="audited_multiple",
            supports_flash_loans=False,
            supports_batch_operations=False,
            adapter_class_name="UniswapV2Adapter"
        )
        
        # Aave V3 - Leading lending protocol
        self.protocols["aave_v3"] = ProtocolConfig(
            protocol_id="aave_v3",
            name="Aave V3",
            category=ProtocolCategory.LENDING,
            description="Decentralized lending and borrowing protocol",
            website="https://aave.com",
            docs_url="https://docs.aave.com/developers/",
            supported_chains=["ethereum", "arbitrum", "base"],
            contracts={
                "ethereum": {
                    "pool": ContractInfo(
                        address="0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",
                        name="Pool",
                        abi_name="aave_v3_pool",
                        deployment_block=16291127
                    ),
                    "pool_data_provider": ContractInfo(
                        address="0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3",
                        name="AaveProtocolDataProvider",
                        abi_name="aave_v3_data_provider"
                    ),
                    "price_oracle": ContractInfo(
                        address="0x54586bE62E3c3580375aE3723C145253060Ca0C2",
                        name="AaveOracle",
                        abi_name="aave_v3_oracle"
                    )
                },
                "arbitrum": {
                    "pool": ContractInfo(
                        address="0x794a61358D6845594F94dc1DB02A252b5b4814aD",
                        name="Pool",
                        abi_name="aave_v3_pool"
                    )
                },
                "base": {
                    "pool": ContractInfo(
                        address="0xA238Dd80C259a72e81d7e4664a9801593F98d1c5",
                        name="Pool",
                        abi_name="aave_v3_pool"
                    )
                }
            },
            settings={
                "ltv_ratios": {"WETH": 0.80, "WBTC": 0.70, "USDC": 0.77},
                "liquidation_thresholds": {"WETH": 0.825, "WBTC": 0.75, "USDC": 0.80},
                "reserve_factors": {"WETH": 0.15, "WBTC": 0.20, "USDC": 0.10}
            },
            risk_level="low",
            tvl_usd=11_000_000_000,  # ~$11B TVL
            audit_status="audited_multiple",
            supports_flash_loans=True,
            supports_batch_operations=True,
            adapter_class_name="AaveV3Adapter"
        )
        
        # Curve Finance - Stable asset DEX
        self.protocols["curve"] = ProtocolConfig(
            protocol_id="curve",
            name="Curve Finance",
            category=ProtocolCategory.DEX_SPOT,
            description="Decentralized exchange for stable assets",
            website="https://curve.fi",
            docs_url="https://curve.readthedocs.io",
            supported_chains=["ethereum"],
            contracts={
                "ethereum": {
                    "registry": ContractInfo(
                        address="0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5",
                        name="CurveRegistry",
                        abi_name="curve_registry"
                    ),
                    "factory": ContractInfo(
                        address="0xB9fC157394Af804a3578134A6585C0dc9cc990d4",
                        name="CurveFactory",
                        abi_name="curve_factory"
                    ),
                    "3pool": ContractInfo(
                        address="0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7",
                        name="3Pool",
                        abi_name="curve_3pool",
                        deployment_block=9456293
                    )
                }
            },
            settings={
                "fee_range": [0.0001, 0.01],  # 0.01% to 1%
                "A_factor_range": [1, 2000],
                "stable_pairs": ["USDC/USDT", "DAI/USDC", "USDT/DAI"]
            },
            risk_level="low",
            tvl_usd=1_800_000_000,  # ~$1.8B TVL
            audit_status="audited_multiple",
            supports_flash_loans=False,
            supports_batch_operations=True,
            adapter_class_name="CurveAdapter"
        )
        
        # Balancer V2 - Multi-asset AMM
        self.protocols["balancer_v2"] = ProtocolConfig(
            protocol_id="balancer_v2",
            name="Balancer V2",
            category=ProtocolCategory.DEX_SPOT,
            description="Multi-asset automated market maker",
            website="https://balancer.fi",
            docs_url="https://docs.balancer.fi",
            supported_chains=["ethereum", "arbitrum"],
            contracts={
                "ethereum": {
                    "vault": ContractInfo(
                        address="0xBA12222222228d8Ba445958a75a0704d566BF2C8",
                        name="Vault",
                        abi_name="balancer_v2_vault",
                        deployment_block=12272146
                    ),
                    "weighted_pool_factory": ContractInfo(
                        address="0x8E9aa87E45f953751CE7b4093f5C00Fb8cdb3Cfc",
                        name="WeightedPoolFactory",
                        abi_name="balancer_v2_weighted_factory"
                    )
                },
                "arbitrum": {
                    "vault": ContractInfo(
                        address="0xBA12222222228d8Ba445958a75a0704d566BF2C8",
                        name="Vault",
                        abi_name="balancer_v2_vault"
                    )
                }
            },
            settings={
                "pool_types": ["weighted", "stable", "liquidity_bootstrapping"],
                "fee_range": [0.0001, 0.10],  # 0.01% to 10%
                "max_tokens_per_pool": 8
            },
            risk_level="medium",
            tvl_usd=900_000_000,  # ~$900M TVL
            audit_status="audited",
            supports_flash_loans=True,
            supports_batch_operations=True,
            adapter_class_name="BalancerV2Adapter"
        )
        
        # SushiSwap - Uniswap V2 fork
        self.protocols["sushiswap"] = ProtocolConfig(
            protocol_id="sushiswap",
            name="SushiSwap",
            category=ProtocolCategory.DEX_SPOT,
            description="Community-driven DEX and DeFi platform",
            website="https://sushi.com",
            docs_url="https://docs.sushi.com",
            supported_chains=["ethereum", "arbitrum"],
            contracts={
                "ethereum": {
                    "factory": ContractInfo(
                        address="0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac",
                        name="SushiswapFactory",
                        abi_name="sushiswap_factory",
                        deployment_block=10794229
                    ),
                    "router": ContractInfo(
                        address="0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
                        name="SushiswapRouter",
                        abi_name="sushiswap_router"
                    )
                },
                "arbitrum": {
                    "factory": ContractInfo(
                        address="0xc35DADB65012eC5796536bD9864eD8773aBc74C4",
                        name="SushiswapFactory",
                        abi_name="sushiswap_factory"
                    ),
                    "router": ContractInfo(
                        address="0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506",
                        name="SushiswapRouter",
                        abi_name="sushiswap_router"
                    )
                }
            },
            settings={
                "fee": 0.003,  # 0.3%
                "min_liquidity": 1000
            },
            risk_level="medium",
            tvl_usd=600_000_000,  # ~$600M TVL
            audit_status="audited",
            supports_flash_loans=False,
            supports_batch_operations=False,
            adapter_class_name="SushiswapAdapter"
        )
        
        logger.info(f"Initialized production registry with {len(self.protocols)} protocols")
    
    def get_protocol(self, protocol_id: str) -> Optional[ProtocolConfig]:
        """Get protocol configuration by ID."""
        return self.protocols.get(protocol_id)
    
    def get_protocols_by_category(self, category: ProtocolCategory) -> List[ProtocolConfig]:
        """Get all protocols in a specific category."""
        return [p for p in self.protocols.values() if p.category == category]
    
    def get_protocols_by_chain(self, chain_name: str) -> List[ProtocolConfig]:
        """Get all protocols supporting a specific chain."""
        return [p for p in self.protocols.values() if chain_name in p.supported_chains]
    
    def get_enabled_protocols(self) -> List[ProtocolConfig]:
        """Get all enabled protocols."""
        return [p for p in self.protocols.values() if p.is_enabled]
    
    def get_flash_loan_protocols(self) -> List[ProtocolConfig]:
        """Get protocols that support flash loans."""
        return [p for p in self.protocols.values() if p.supports_flash_loans and p.is_enabled]
    
    def get_contract_address(self, protocol_id: str, chain_name: str, contract_name: str) -> Optional[str]:
        """Get contract address for a specific protocol, chain, and contract."""
        protocol = self.get_protocol(protocol_id)
        if not protocol:
            return None
        
        chain_contracts = protocol.contracts.get(chain_name, {})
        contract_info = chain_contracts.get(contract_name)
        
        return contract_info.address if contract_info else None
    
    def get_all_contract_addresses(self, chain_name: str) -> Dict[str, Dict[str, str]]:
        """Get all contract addresses for a specific chain."""
        result = {}
        
        for protocol_id, protocol in self.protocols.items():
            if chain_name in protocol.supported_chains:
                chain_contracts = protocol.contracts.get(chain_name, {})
                if chain_contracts:
                    result[protocol_id] = {
                        name: info.address for name, info in chain_contracts.items()
                    }
        
        return result
    
    def validate_protocol_config(self, protocol_id: str) -> Dict[str, Any]:
        """Validate a protocol configuration."""
        protocol = self.get_protocol(protocol_id)
        if not protocol:
            return {"valid": False, "error": f"Protocol {protocol_id} not found"}
        
        validation_result = {
            "valid": True,
            "protocol_id": protocol_id,
            "warnings": [],
            "errors": []
        }
        
        # Check required fields
        if not protocol.contracts:
            validation_result["errors"].append("No contracts configured")
        
        # Check contract addresses
        for chain_name, contracts in protocol.contracts.items():
            for contract_name, contract_info in contracts.items():
                if not contract_info.address:
                    validation_result["errors"].append(
                        f"Missing address for {contract_name} on {chain_name}"
                    )
                elif not contract_info.address.startswith("0x"):
                    validation_result["errors"].append(
                        f"Invalid address format for {contract_name} on {chain_name}"
                    )
        
        # Check if any chains have missing contracts
        if protocol.supported_chains:
            configured_chains = set(protocol.contracts.keys())
            supported_chains = set(protocol.supported_chains)
            missing_chains = supported_chains - configured_chains
            
            if missing_chains:
                validation_result["warnings"].append(
                    f"Missing contract configs for chains: {', '.join(missing_chains)}"
                )
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
        return validation_result
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the protocol registry."""
        total_protocols = len(self.protocols)
        enabled_protocols = len(self.get_enabled_protocols())
        
        category_counts = {}
        chain_counts = {}
        
        for protocol in self.protocols.values():
            # Count by category
            category = protocol.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count by chain
            for chain in protocol.supported_chains:
                chain_counts[chain] = chain_counts.get(chain, 0) + 1
        
        return {
            "total_protocols": total_protocols,
            "enabled_protocols": enabled_protocols,
            "disabled_protocols": total_protocols - enabled_protocols,
            "categories": category_counts,
            "chain_support": chain_counts,
            "flash_loan_protocols": len(self.get_flash_loan_protocols()),
            "total_tvl_usd": sum(p.tvl_usd or 0 for p in self.protocols.values()),
            "risk_levels": {
                "low": len([p for p in self.protocols.values() if p.risk_level == "low"]),
                "medium": len([p for p in self.protocols.values() if p.risk_level == "medium"]),
                "high": len([p for p in self.protocols.values() if p.risk_level == "high"])
            }
        }


# Global production registry instance
production_registry = ProductionProtocolRegistry()