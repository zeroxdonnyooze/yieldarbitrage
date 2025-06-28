"""
Flash Loan Provider interface and implementations for yield arbitrage.

This module provides a unified interface for discovering and integrating flash loan
opportunities from various providers (Aave, dYdX, Balancer, etc.) into the graph engine.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from decimal import Decimal
from enum import Enum

logger = logging.getLogger(__name__)


class FlashLoanProvider(str, Enum):
    """Supported flash loan providers."""
    AAVE_V3 = "aave_v3"
    AAVE_V2 = "aave_v2"
    DYDX = "dydx"
    BALANCER = "balancer"
    COMPOUND = "compound"
    UNISWAP_V3 = "uniswap_v3"


@dataclass
class FlashLoanTerms:
    """Terms and conditions for a flash loan."""
    provider: FlashLoanProvider
    asset: str
    max_amount: Decimal
    fee_rate: Decimal  # As decimal (0.0009 = 0.09%)
    fixed_fee: Decimal  # Fixed fee amount
    min_amount: Decimal
    gas_estimate: int
    requires_collateral: bool = False
    time_limit_blocks: Optional[int] = None  # Must repay within N blocks
    contract_address: str = ""
    callback_gas_limit: int = 3_000_000


@dataclass
class FlashLoanOpportunity:
    """Represents a flash loan opportunity for a specific asset."""
    asset_id: str
    available_providers: List[FlashLoanTerms]
    best_terms: FlashLoanTerms  # Provider with lowest cost
    total_max_liquidity: Decimal


class FlashLoanDiscovery:
    """
    Discovers and manages flash loan opportunities across multiple providers.
    
    This class provides the graph engine with information about available
    flash loans that can enable capital-efficient arbitrage strategies.
    """
    
    def __init__(self, chain_name: str = "ethereum"):
        """
        Initialize flash loan discovery.
        
        Args:
            chain_name: Blockchain network name
        """
        self.chain_name = chain_name
        self.provider_configs = self._initialize_provider_configs()
        self.cached_opportunities: Dict[str, FlashLoanOpportunity] = {}
        
        # Statistics
        self.stats = {
            "opportunities_discovered": 0,
            "providers_checked": 0,
            "total_liquidity_discovered": Decimal('0')
        }
    
    def _initialize_provider_configs(self) -> Dict[FlashLoanProvider, Dict]:
        """Initialize provider-specific configurations."""
        if self.chain_name == "ethereum":
            return {
                FlashLoanProvider.AAVE_V3: {
                    "pool_address": "0x87870Bca3F8e07Da8c8f4B3A4d8f8Eb2a8B6e4f1",
                    "fee_rate": Decimal("0.0009"),  # 0.09%
                    "supported_assets": {
                        "USDC": "0xA0b86a33E6441c2b73AC95F2DB8FFf6d4daF1E65",
                        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                        "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                        "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                        "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
                    }
                },
                FlashLoanProvider.BALANCER: {
                    "vault_address": "0xBA12222222228d8Ba445958a75a0704d566BF2C8",
                    "fee_rate": Decimal("0"),  # No fees on Balancer flash loans
                    "supported_assets": {
                        "USDC": "0xA0b86a33E6441c2b73AC95F2DB8FFf6d4daF1E65",
                        "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                        "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
                    }
                },
                FlashLoanProvider.UNISWAP_V3: {
                    "factory_address": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
                    "fee_rate": Decimal("0"),  # Pay pool fees only
                    "supported_assets": {
                        "USDC": "0xA0b86a33E6441c2b73AC95F2DB8FFf6d4daF1E65",
                        "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
                    }
                }
            }
        else:
            # Other chains
            return {}
    
    def discover_flash_loan_opportunities(self, 
                                        target_assets: Optional[Set[str]] = None,
                                        min_liquidity: Decimal = Decimal('1000')) -> List[FlashLoanOpportunity]:
        """
        Discover available flash loan opportunities.
        
        Args:
            target_assets: Specific assets to check (None for all)
            min_liquidity: Minimum liquidity required
            
        Returns:
            List of flash loan opportunities
        """
        opportunities = []
        
        # Get all assets to check
        all_assets = set()
        for provider_config in self.provider_configs.values():
            all_assets.update(provider_config.get("supported_assets", {}).keys())
        
        assets_to_check = target_assets or all_assets
        
        for asset in assets_to_check:
            opportunity = self._discover_asset_opportunities(asset, min_liquidity)
            if opportunity and opportunity.total_max_liquidity >= min_liquidity:
                opportunities.append(opportunity)
                self.cached_opportunities[asset] = opportunity
        
        self.stats["opportunities_discovered"] = len(opportunities)
        
        logger.info(f"Discovered {len(opportunities)} flash loan opportunities")
        return opportunities
    
    def _discover_asset_opportunities(self, asset: str, min_liquidity: Decimal) -> Optional[FlashLoanOpportunity]:
        """Discover flash loan opportunities for a specific asset."""
        available_providers = []
        total_liquidity = Decimal('0')
        
        for provider, config in self.provider_configs.items():
            if asset in config.get("supported_assets", {}):
                terms = self._get_provider_terms(provider, asset, config)
                if terms and terms.max_amount >= min_liquidity:
                    available_providers.append(terms)
                    total_liquidity += terms.max_amount
                    self.stats["providers_checked"] += 1
        
        if not available_providers:
            return None
        
        # Find best terms (lowest total cost)
        best_terms = min(available_providers, 
                        key=lambda t: t.fee_rate + (t.fixed_fee / max(t.max_amount, Decimal('1'))))
        
        self.stats["total_liquidity_discovered"] += total_liquidity
        
        return FlashLoanOpportunity(
            asset_id=asset,
            available_providers=available_providers,
            best_terms=best_terms,
            total_max_liquidity=total_liquidity
        )
    
    def _get_provider_terms(self, provider: FlashLoanProvider, 
                          asset: str, config: Dict) -> Optional[FlashLoanTerms]:
        """Get terms for a specific provider and asset."""
        try:
            if provider == FlashLoanProvider.AAVE_V3:
                return self._get_aave_v3_terms(asset, config)
            elif provider == FlashLoanProvider.BALANCER:
                return self._get_balancer_terms(asset, config)
            elif provider == FlashLoanProvider.UNISWAP_V3:
                return self._get_uniswap_v3_terms(asset, config)
            else:
                logger.warning(f"Unsupported provider: {provider}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting terms for {provider} {asset}: {e}")
            return None
    
    def _get_aave_v3_terms(self, asset: str, config: Dict) -> FlashLoanTerms:
        """Get Aave V3 flash loan terms."""
        # In real implementation, would query Aave contracts for actual liquidity
        max_liquidity = self._estimate_aave_liquidity(asset)
        
        return FlashLoanTerms(
            provider=FlashLoanProvider.AAVE_V3,
            asset=asset,
            max_amount=max_liquidity,
            fee_rate=config["fee_rate"],
            fixed_fee=Decimal('0'),
            min_amount=Decimal('1'),
            gas_estimate=150_000,
            requires_collateral=False,
            contract_address=config["pool_address"]
        )
    
    def _get_balancer_terms(self, asset: str, config: Dict) -> FlashLoanTerms:
        """Get Balancer flash loan terms."""
        max_liquidity = self._estimate_balancer_liquidity(asset)
        
        return FlashLoanTerms(
            provider=FlashLoanProvider.BALANCER,
            asset=asset,
            max_amount=max_liquidity,
            fee_rate=config["fee_rate"],  # Usually 0
            fixed_fee=Decimal('0'),
            min_amount=Decimal('1'),
            gas_estimate=200_000,
            requires_collateral=False,
            contract_address=config["vault_address"]
        )
    
    def _get_uniswap_v3_terms(self, asset: str, config: Dict) -> FlashLoanTerms:
        """Get Uniswap V3 flash loan terms."""
        max_liquidity = self._estimate_uniswap_v3_liquidity(asset)
        
        return FlashLoanTerms(
            provider=FlashLoanProvider.UNISWAP_V3,
            asset=asset,
            max_amount=max_liquidity,
            fee_rate=Decimal('0.0001'),  # Approximate pool fee
            fixed_fee=Decimal('0'),
            min_amount=Decimal('1'),
            gas_estimate=100_000,
            requires_collateral=False,
            contract_address=config["factory_address"]
        )
    
    def _estimate_aave_liquidity(self, asset: str) -> Decimal:
        """Estimate available Aave liquidity for asset."""
        # Simplified estimation - real implementation would query contracts
        estimates = {
            "USDC": Decimal('100_000_000'),  # $100M
            "USDT": Decimal('80_000_000'),   # $80M
            "DAI": Decimal('50_000_000'),    # $50M
            "WETH": Decimal('200_000'),      # 200K ETH
            "WBTC": Decimal('5_000')         # 5K BTC
        }
        return estimates.get(asset, Decimal('1_000'))
    
    def _estimate_balancer_liquidity(self, asset: str) -> Decimal:
        """Estimate available Balancer liquidity for asset."""
        estimates = {
            "USDC": Decimal('20_000_000'),   # $20M
            "DAI": Decimal('15_000_000'),    # $15M
            "WETH": Decimal('50_000')        # 50K ETH
        }
        return estimates.get(asset, Decimal('500'))
    
    def _estimate_uniswap_v3_liquidity(self, asset: str) -> Decimal:
        """Estimate available Uniswap V3 liquidity for asset."""
        estimates = {
            "USDC": Decimal('30_000_000'),   # $30M
            "WETH": Decimal('80_000')        # 80K ETH
        }
        return estimates.get(asset, Decimal('1_000'))
    
    def get_best_flash_loan_for_amount(self, asset: str, 
                                     required_amount: Decimal) -> Optional[FlashLoanTerms]:
        """
        Get the best flash loan provider for a specific amount.
        
        Args:
            asset: Asset to borrow
            required_amount: Amount needed
            
        Returns:
            Best flash loan terms or None if not available
        """
        if asset in self.cached_opportunities:
            opportunity = self.cached_opportunities[asset]
        else:
            opportunities = self.discover_flash_loan_opportunities({asset})
            if not opportunities:
                return None
            opportunity = opportunities[0]
        
        # Filter providers that can supply the required amount
        suitable_providers = [
            terms for terms in opportunity.available_providers
            if terms.max_amount >= required_amount
        ]
        
        if not suitable_providers:
            return None
        
        # Return provider with lowest total cost for this amount
        def calculate_cost(terms: FlashLoanTerms) -> Decimal:
            return (required_amount * terms.fee_rate) + terms.fixed_fee
        
        return min(suitable_providers, key=calculate_cost)
    
    def calculate_flash_loan_cost(self, terms: FlashLoanTerms, amount: Decimal) -> Decimal:
        """Calculate total cost of a flash loan."""
        return (amount * terms.fee_rate) + terms.fixed_fee
    
    def is_flash_loan_profitable(self, terms: FlashLoanTerms, 
                                amount: Decimal, expected_profit: Decimal) -> bool:
        """Check if a flash loan would be profitable given expected returns."""
        cost = self.calculate_flash_loan_cost(terms, amount)
        return expected_profit > cost
    
    def get_supported_assets(self) -> Set[str]:
        """Get all assets that support flash loans."""
        all_assets = set()
        for provider_config in self.provider_configs.values():
            all_assets.update(provider_config.get("supported_assets", {}).keys())
        return all_assets
    
    def get_statistics(self) -> Dict:
        """Get discovery statistics."""
        return {
            **self.stats,
            "cached_opportunities": len(self.cached_opportunities),
            "providers_configured": len(self.provider_configs),
            "supported_assets": len(self.get_supported_assets())
        }
    
    def clear_cache(self):
        """Clear cached opportunities (e.g., when liquidity changes)."""
        self.cached_opportunities.clear()


class FlashLoanEdgeGenerator:
    """
    Generates flash loan edges for the graph engine.
    
    This class creates virtual edges that represent flash loan opportunities,
    allowing the pathfinding algorithms to consider flash loan strategies.
    """
    
    def __init__(self, flash_loan_discovery: FlashLoanDiscovery):
        """
        Initialize flash loan edge generator.
        
        Args:
            flash_loan_discovery: Flash loan discovery instance
        """
        self.discovery = flash_loan_discovery
        
    def generate_flash_loan_edges(self, 
                                 opportunities: List[FlashLoanOpportunity]) -> List[Dict]:
        """
        Generate flash loan edges from discovered opportunities.
        
        Args:
            opportunities: Flash loan opportunities
            
        Returns:
            List of edge data for graph engine integration
        """
        edges = []
        
        for opportunity in opportunities:
            # Create primary flash loan edge (asset -> flash_asset)
            primary_edge = self._create_flash_loan_edge(
                source_asset=opportunity.asset_id,
                target_asset=f"FLASH_{opportunity.asset_id}",
                terms=opportunity.best_terms,
                edge_id=f"flash_loan_{opportunity.asset_id}"
            )
            edges.append(primary_edge)
            
            # Create repayment edge (flash_asset -> asset)
            repay_edge = self._create_repayment_edge(
                source_asset=f"FLASH_{opportunity.asset_id}",
                target_asset=opportunity.asset_id,
                terms=opportunity.best_terms,
                edge_id=f"repay_{opportunity.asset_id}"
            )
            edges.append(repay_edge)
        
        logger.info(f"Generated {len(edges)} flash loan edges")
        return edges
    
    def _create_flash_loan_edge(self, source_asset: str, target_asset: str,
                               terms: FlashLoanTerms, edge_id: str) -> Dict:
        """Create a flash loan edge."""
        return {
            "edge_id": edge_id,
            "source_asset_id": source_asset,
            "target_asset_id": target_asset,
            "edge_type": "FLASH_LOAN",
            "protocol_name": terms.provider.value,
            "chain_name": self.discovery.chain_name,
            "execution_properties": {
                "supports_synchronous": True,
                "requires_time_delay": None,
                "requires_bridge": False,
                "requires_capital_holding": False,
                "gas_estimate": terms.gas_estimate,
                "mev_sensitivity": 0.1,  # Low MEV risk for flash loans
                "min_liquidity_required": float(terms.max_amount)
            },
            "constraints": {
                "min_input_amount": float(terms.min_amount),
                "max_input_amount": float(terms.max_amount),
                "fee_rate": float(terms.fee_rate),
                "fixed_fee": float(terms.fixed_fee)
            },
            "state": {
                "is_active": True,
                "last_updated": None,
                "liquidity_available": float(terms.max_amount),
                "current_rate": float(terms.fee_rate)
            },
            "metadata": {
                "provider": terms.provider.value,
                "contract_address": terms.contract_address,
                "requires_collateral": terms.requires_collateral,
                "callback_gas_limit": terms.callback_gas_limit
            }
        }
    
    def _create_repayment_edge(self, source_asset: str, target_asset: str,
                              terms: FlashLoanTerms, edge_id: str) -> Dict:
        """Create a repayment edge for flash loans."""
        return {
            "edge_id": edge_id,
            "source_asset_id": source_asset,
            "target_asset_id": target_asset,
            "edge_type": "FLASH_REPAY",
            "protocol_name": terms.provider.value,
            "chain_name": self.discovery.chain_name,
            "execution_properties": {
                "supports_synchronous": True,
                "requires_time_delay": None,
                "requires_bridge": False,
                "requires_capital_holding": False,
                "gas_estimate": 50_000,  # Lower gas for repayment
                "mev_sensitivity": 0.0,  # No MEV risk for repayment
                "min_liquidity_required": 0.0
            },
            "constraints": {
                "min_input_amount": float(terms.min_amount),
                "max_input_amount": float(terms.max_amount),
                "fee_rate": float(terms.fee_rate),
                "fixed_fee": float(terms.fixed_fee)
            },
            "state": {
                "is_active": True,
                "last_updated": None,
                "liquidity_available": float(terms.max_amount),
                "current_rate": float(terms.fee_rate)
            },
            "metadata": {
                "provider": terms.provider.value,
                "contract_address": terms.contract_address,
                "is_repayment": True
            }
        }
    
    def update_flash_loan_edges(self, existing_edges: List[Dict]) -> List[Dict]:
        """
        Update existing flash loan edges with current terms.
        
        Args:
            existing_edges: Current flash loan edges
            
        Returns:
            Updated edge list
        """
        # Refresh opportunities
        opportunities = self.discovery.discover_flash_loan_opportunities()
        
        # Create lookup for new terms
        asset_terms = {
            opp.asset_id: opp.best_terms 
            for opp in opportunities
        }
        
        updated_edges = []
        for edge in existing_edges:
            if edge.get("edge_type") in ["FLASH_LOAN", "FLASH_REPAY"]:
                # Extract asset from edge
                asset = edge["source_asset_id"]
                if asset.startswith("FLASH_"):
                    asset = asset[6:]  # Remove "FLASH_" prefix
                
                # Update with new terms if available
                if asset in asset_terms:
                    terms = asset_terms[asset]
                    edge["constraints"]["fee_rate"] = float(terms.fee_rate)
                    edge["constraints"]["max_input_amount"] = float(terms.max_amount)
                    edge["state"]["liquidity_available"] = float(terms.max_amount)
                    edge["state"]["current_rate"] = float(terms.fee_rate)
                    
            updated_edges.append(edge)
        
        return updated_edges