"""
Flash Loan Graph Engine Integration.

This module integrates flash loan discovery and edge generation into the 
main yield arbitrage graph engine, enabling pathfinding algorithms to 
discover and utilize flash loan opportunities.
"""
import logging
from typing import Dict, List, Optional, Set, Tuple
from decimal import Decimal

from yield_arbitrage.protocols.flash_loan_provider import (
    FlashLoanDiscovery, FlashLoanEdgeGenerator, FlashLoanOpportunity, FlashLoanProvider
)
from yield_arbitrage.graph_engine.models import (
    YieldGraphEdge, EdgeType, EdgeExecutionProperties, EdgeConstraints, EdgeState
)

logger = logging.getLogger(__name__)


class FlashLoanGraphIntegrator:
    """
    Integrates flash loan capabilities into the yield arbitrage graph engine.
    
    This class manages the discovery of flash loan opportunities and their
    integration as virtual edges in the graph, enabling capital-efficient
    arbitrage strategies.
    """
    
    def __init__(self, chain_name: str = "ethereum"):
        """
        Initialize flash loan graph integrator.
        
        Args:
            chain_name: Blockchain network name
        """
        self.chain_name = chain_name
        self.discovery = FlashLoanDiscovery(chain_name)
        self.edge_generator = FlashLoanEdgeGenerator(self.discovery)
        
        # Track flash loan edges in the graph
        self.flash_loan_edges: Dict[str, YieldGraphEdge] = {}
        self.virtual_assets: Set[str] = set()  # FLASH_* assets
        
        # Configuration
        self.min_flash_loan_amount = Decimal('1000')  # $1K minimum
        self.max_flash_loan_utilization = 0.8  # Use max 80% of available liquidity
        
        # Statistics
        self.stats = {
            "flash_loan_edges_created": 0,
            "virtual_assets_created": 0,
            "opportunities_integrated": 0,
            "total_flash_loan_capacity": Decimal('0')
        }
    
    def discover_and_integrate_flash_loans(self, 
                                         target_assets: Optional[Set[str]] = None,
                                         existing_edges: Optional[List[YieldGraphEdge]] = None) -> List[YieldGraphEdge]:
        """
        Discover flash loan opportunities and integrate them as graph edges.
        
        Args:
            target_assets: Specific assets to find flash loans for
            existing_edges: Existing edges to consider for integration
            
        Returns:
            List of new flash loan edges
        """
        logger.info(f"Discovering flash loan opportunities for {len(target_assets) if target_assets else 'all'} assets")
        
        try:
            # Discover opportunities
            opportunities = self.discovery.discover_flash_loan_opportunities(
                target_assets=target_assets,
                min_liquidity=self.min_flash_loan_amount
            )
        except Exception as e:
            logger.error(f"Failed to discover flash loan opportunities: {e}")
            return []
        
        # Generate edges from opportunities
        new_edges = []
        for opportunity in opportunities:
            edges = self._integrate_opportunity(opportunity, existing_edges)
            new_edges.extend(edges)
        
        self.stats["opportunities_integrated"] = len(opportunities)
        self.stats["flash_loan_edges_created"] = len(new_edges)
        
        logger.info(f"Integrated {len(opportunities)} flash loan opportunities as {len(new_edges)} edges")
        
        return new_edges
    
    def _integrate_opportunity(self, opportunity: FlashLoanOpportunity,
                             existing_edges: Optional[List[YieldGraphEdge]] = None) -> List[YieldGraphEdge]:
        """Integrate a single flash loan opportunity into the graph."""
        edges = []
        
        # Create virtual flash loan asset
        flash_asset_id = f"FLASH_{opportunity.asset_id}"
        self.virtual_assets.add(flash_asset_id)
        self.stats["virtual_assets_created"] += 1
        
        # Create flash loan edge (asset -> flash_asset)
        flash_edge = self._create_flash_loan_edge(opportunity)
        edges.append(flash_edge)
        self.flash_loan_edges[flash_edge.edge_id] = flash_edge
        
        # Create repayment edge (flash_asset -> asset)
        repay_edge = self._create_repayment_edge(opportunity)
        edges.append(repay_edge)
        self.flash_loan_edges[repay_edge.edge_id] = repay_edge
        
        # Update capacity tracking
        self.stats["total_flash_loan_capacity"] += opportunity.total_max_liquidity
        
        # Check for synergies with existing edges
        if existing_edges:
            synergy_edges = self._find_flash_loan_synergies(opportunity, existing_edges)
            edges.extend(synergy_edges)
        
        return edges
    
    def _create_flash_loan_edge(self, opportunity: FlashLoanOpportunity) -> YieldGraphEdge:
        """Create a flash loan edge from an opportunity."""
        terms = opportunity.best_terms
        
        # Calculate effective liquidity (use only a portion to avoid failures)
        effective_liquidity = float(terms.max_amount * Decimal(str(self.max_flash_loan_utilization)))
        
        edge = YieldGraphEdge(
            edge_id=f"flash_loan_{opportunity.asset_id}_{terms.provider.value}",
            source_asset_id=opportunity.asset_id,
            target_asset_id=f"FLASH_{opportunity.asset_id}",
            edge_type=EdgeType.FLASH_LOAN,
            protocol_name=terms.provider.value,
            chain_name=self.chain_name,
            execution_properties=EdgeExecutionProperties(
                supports_synchronous=True,
                requires_time_delay=None,
                requires_bridge=False,
                requires_capital_holding=False,
                gas_estimate=terms.gas_estimate,
                mev_sensitivity=0.1,  # Low MEV sensitivity
                min_liquidity_required=effective_liquidity
            ),
            constraints=EdgeConstraints(
                min_input_amount=float(terms.min_amount),
                max_input_amount=effective_liquidity
            ),
            state=EdgeState(
                liquidity_usd=effective_liquidity,
                conversion_rate=1.0,  # 1:1 for flash loans
                gas_cost_usd=5.0,  # Estimate $5 gas cost
                confidence_score=0.95  # High confidence in flash loans
            )
        )
        
        # Add flash loan specific metadata
        edge.metadata = {
            "provider": terms.provider.value,
            "fee_rate": float(terms.fee_rate),
            "fixed_fee": float(terms.fixed_fee),
            "contract_address": terms.contract_address,
            "callback_gas_limit": terms.callback_gas_limit,
            "max_available_liquidity": float(terms.max_amount),
            "requires_collateral": terms.requires_collateral
        }
        
        return edge
    
    def _create_repayment_edge(self, opportunity: FlashLoanOpportunity) -> YieldGraphEdge:
        """Create a flash loan repayment edge."""
        terms = opportunity.best_terms
        effective_liquidity = float(terms.max_amount * Decimal(str(self.max_flash_loan_utilization)))
        
        edge = YieldGraphEdge(
            edge_id=f"repay_{opportunity.asset_id}_{terms.provider.value}",
            source_asset_id=f"FLASH_{opportunity.asset_id}",
            target_asset_id=opportunity.asset_id,
            edge_type=EdgeType.FLASH_REPAY,
            protocol_name=terms.provider.value,
            chain_name=self.chain_name,
            execution_properties=EdgeExecutionProperties(
                supports_synchronous=True,
                requires_time_delay=None,
                requires_bridge=False,
                requires_capital_holding=False,
                gas_estimate=50_000,  # Lower gas for repayment
                mev_sensitivity=0.0,  # No MEV risk
                min_liquidity_required=0.0
            ),
            constraints=EdgeConstraints(
                min_input_amount=float(terms.min_amount),
                max_input_amount=effective_liquidity
            ),
            state=EdgeState(
                liquidity_usd=effective_liquidity,
                conversion_rate=1.0 + float(terms.fee_rate),  # Include fee
                gas_cost_usd=2.0,  # Lower gas cost for repayment
                confidence_score=0.99  # Very high confidence in repayment
            )
        )
        
        # Add repayment specific metadata
        edge.metadata = {
            "provider": terms.provider.value,
            "fee_rate": float(terms.fee_rate),
            "fixed_fee": float(terms.fixed_fee),
            "is_repayment": True,
            "requires_flash_loan_context": True
        }
        
        return edge
    
    def _find_flash_loan_synergies(self, opportunity: FlashLoanOpportunity,
                                 existing_edges: List[YieldGraphEdge]) -> List[YieldGraphEdge]:
        """Find synergies between flash loans and existing edges."""
        synergy_edges = []
        
        # Look for high-capital arbitrage opportunities
        high_capital_edges = [
            edge for edge in existing_edges
            if (edge.constraints.min_input_amount > 10000 and  # $10K+ trades
                edge.source_asset_id == opportunity.asset_id)
        ]
        
        # Create enhanced versions of high-capital edges for flash loan context
        for edge in high_capital_edges:
            if self._would_benefit_from_flash_loan(edge, opportunity):
                enhanced_edge = self._create_flash_enhanced_edge(edge, opportunity)
                if enhanced_edge:
                    synergy_edges.append(enhanced_edge)
        
        return synergy_edges
    
    def _would_benefit_from_flash_loan(self, edge: YieldGraphEdge,
                                     opportunity: FlashLoanOpportunity) -> bool:
        """Check if an edge would benefit from flash loan capital."""
        # Benefits if:
        # 1. Edge requires significant capital
        # 2. Flash loan cost is lower than expected profit margin
        # 3. Edge has high confidence score
        
        required_capital = edge.constraints.min_input_amount
        flash_loan_cost = float(opportunity.best_terms.fee_rate * Decimal(str(required_capital)))
        
        # Estimate profit margin (simplified)
        potential_profit = required_capital * 0.01  # Assume 1% profit opportunity
        
        return (required_capital >= 5000 and  # $5K minimum
                flash_loan_cost < potential_profit * 0.5 and  # Cost < 50% of profit
                edge.state.confidence_score > 0.7)
    
    def _create_flash_enhanced_edge(self, original_edge: YieldGraphEdge,
                                   opportunity: FlashLoanOpportunity) -> Optional[YieldGraphEdge]:
        """Create an enhanced version of an edge for flash loan context."""
        try:
            # Create new edge with flash loan context
            enhanced_edge = YieldGraphEdge(
                edge_id=f"flash_enhanced_{original_edge.edge_id}",
                source_asset_id=f"FLASH_{opportunity.asset_id}",
                target_asset_id=original_edge.target_asset_id,
                edge_type=original_edge.edge_type,
                protocol_name=original_edge.protocol_name,
                chain_name=original_edge.chain_name,
                execution_properties=original_edge.execution_properties.model_copy(),
                constraints=original_edge.constraints.model_copy(),
                state=original_edge.state.model_copy()
            )
            
            # Adjust constraints for flash loan context
            max_flash_amount = float(opportunity.best_terms.max_amount * Decimal(str(self.max_flash_loan_utilization)))
            enhanced_edge.constraints.max_input_amount = min(
                original_edge.constraints.max_input_amount,
                max_flash_amount
            )
            
            # Add flash loan context metadata
            enhanced_edge.metadata = {
                **original_edge.metadata,
                "flash_loan_enhanced": True,
                "flash_loan_provider": opportunity.best_terms.provider.value,
                "original_edge_id": original_edge.edge_id
            }
            
            return enhanced_edge
            
        except Exception as e:
            logger.error(f"Failed to create flash enhanced edge: {e}")
            return None
    
    def get_flash_loan_capacity(self, asset: str) -> Optional[Decimal]:
        """Get total flash loan capacity for an asset."""
        if asset in self.discovery.cached_opportunities:
            return self.discovery.cached_opportunities[asset].total_max_liquidity
        
        # Try to discover if not cached
        try:
            opportunities = self.discovery.discover_flash_loan_opportunities({asset})
            if opportunities:
                return opportunities[0].total_max_liquidity
        except Exception:
            pass
        
        return None
    
    def is_virtual_asset(self, asset_id: str) -> bool:
        """Check if an asset is a virtual flash loan asset."""
        return asset_id in self.virtual_assets
    
    def get_required_repayment_amount(self, flash_asset: str, borrowed_amount: Decimal) -> Decimal:
        """Calculate required repayment amount for a flash loan."""
        # Extract original asset from flash asset
        if not flash_asset.startswith("FLASH_"):
            raise ValueError(f"Not a flash loan asset: {flash_asset}")
        
        original_asset = flash_asset[6:]  # Remove "FLASH_" prefix
        
        # Find the opportunity
        if original_asset in self.discovery.cached_opportunities:
            opportunity = self.discovery.cached_opportunities[original_asset]
            cost = self.discovery.calculate_flash_loan_cost(opportunity.best_terms, borrowed_amount)
            return borrowed_amount + cost
        
        # Fallback to conservative estimate
        return borrowed_amount * Decimal('1.001')  # 0.1% fee
    
    def validate_flash_loan_path(self, edges: List[YieldGraphEdge]) -> bool:
        """
        Validate that a path with flash loans is properly structured.
        
        Args:
            edges: List of edges forming a path
            
        Returns:
            True if flash loan path is valid
        """
        flash_loan_edges = [e for e in edges if e.edge_type == EdgeType.FLASH_LOAN]
        repayment_edges = [e for e in edges if e.edge_type == EdgeType.FLASH_REPAY]
        
        # Must have matching flash loan and repayment edges
        if len(flash_loan_edges) != len(repayment_edges):
            return False
        
        # Each flash loan must have corresponding repayment
        for flash_edge in flash_loan_edges:
            flash_asset = flash_edge.target_asset_id
            original_asset = flash_asset[6:]  # Remove "FLASH_"
            
            # Find matching repayment
            matching_repay = None
            for repay_edge in repayment_edges:
                if (repay_edge.source_asset_id == flash_asset and
                    repay_edge.target_asset_id == original_asset):
                    matching_repay = repay_edge
                    break
            
            if not matching_repay:
                logger.warning(f"No matching repayment for flash loan {flash_edge.edge_id}")
                return False
        
        return True
    
    def update_flash_loan_liquidity(self):
        """Update flash loan liquidity information."""
        logger.info("Updating flash loan liquidity...")
        
        # Clear cache and rediscover
        self.discovery.clear_cache()
        
        # Update existing edges
        updated_edges = []
        for edge in self.flash_loan_edges.values():
            if edge.edge_type in [EdgeType.FLASH_LOAN, EdgeType.FLASH_REPAY]:
                # Extract asset from edge
                asset = edge.source_asset_id
                if asset.startswith("FLASH_"):
                    asset = asset[6:]
                
                # Get updated terms
                opportunities = self.discovery.discover_flash_loan_opportunities({asset})
                if opportunities:
                    opportunity = opportunities[0]
                    
                    # Update edge with new liquidity
                    effective_liquidity = float(opportunity.best_terms.max_amount * Decimal(str(self.max_flash_loan_utilization)))
                    edge.constraints.max_input_amount = effective_liquidity
                    edge.state.liquidity_usd = effective_liquidity
                    
                    # Update metadata
                    edge.metadata.update({
                        "fee_rate": float(opportunity.best_terms.fee_rate),
                        "max_available_liquidity": float(opportunity.best_terms.max_amount)
                    })
        
        logger.info("Flash loan liquidity updated")
    
    def get_statistics(self) -> Dict:
        """Get flash loan integration statistics."""
        discovery_stats = self.discovery.get_statistics()
        
        return {
            **self.stats,
            **discovery_stats,
            "flash_loan_edges_active": len(self.flash_loan_edges),
            "virtual_assets_active": len(self.virtual_assets)
        }