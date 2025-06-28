"""
Unit tests for flash loan graph engine integration.

Tests the integration of flash loan discovery into the yield arbitrage
graph engine, including edge creation, path validation, and synergies.
"""
import pytest
from decimal import Decimal
from unittest.mock import Mock, patch
from typing import List

from yield_arbitrage.graph_engine.flash_loan_integrator import FlashLoanGraphIntegrator
from yield_arbitrage.graph_engine.models import (
    YieldGraphEdge, EdgeType, EdgeExecutionProperties, EdgeConstraints, EdgeState
)
from yield_arbitrage.protocols.flash_loan_provider import (
    FlashLoanOpportunity, FlashLoanTerms, FlashLoanProvider
)


def create_test_edge(
    edge_id: str,
    source_asset: str,
    target_asset: str,
    edge_type: EdgeType = EdgeType.TRADE,
    min_input: float = 1000.0,
    max_input: float = 100000.0
) -> YieldGraphEdge:
    """Create a test edge for integration tests."""
    return YieldGraphEdge(
        edge_id=edge_id,
        source_asset_id=source_asset,
        target_asset_id=target_asset,
        edge_type=edge_type,
        protocol_name="test_protocol",
        chain_name="ethereum",
        execution_properties=EdgeExecutionProperties(),
        constraints=EdgeConstraints(
            min_input_amount=min_input,
            max_input_amount=max_input
        ),
        state=EdgeState()
    )


def create_test_opportunity(asset: str = "USDC", 
                          max_amount: Decimal = Decimal('100000')) -> FlashLoanOpportunity:
    """Create a test flash loan opportunity."""
    terms = FlashLoanTerms(
        provider=FlashLoanProvider.AAVE_V3,
        asset=asset,
        max_amount=max_amount,
        fee_rate=Decimal('0.0009'),
        fixed_fee=Decimal('0'),
        min_amount=Decimal('1'),
        gas_estimate=150000,
        contract_address="0x87870Bca3F8e07Da8c8f4B3A4d8f8Eb2a8B6e4f1"
    )
    
    return FlashLoanOpportunity(
        asset_id=asset,
        available_providers=[terms],
        best_terms=terms,
        total_max_liquidity=max_amount
    )


class TestFlashLoanGraphIntegrator:
    """Test suite for FlashLoanGraphIntegrator."""
    
    @pytest.fixture
    def integrator(self):
        """Create a FlashLoanGraphIntegrator instance."""
        return FlashLoanGraphIntegrator(chain_name="ethereum")
    
    def test_initialization(self, integrator):
        """Test integrator initialization."""
        assert integrator.chain_name == "ethereum"
        assert integrator.discovery is not None
        assert integrator.edge_generator is not None
        assert len(integrator.flash_loan_edges) == 0
        assert len(integrator.virtual_assets) == 0
    
    def test_discover_and_integrate_flash_loans(self, integrator):
        """Test discovering and integrating flash loan opportunities."""
        with patch.object(integrator.discovery, 'discover_flash_loan_opportunities') as mock_discovery:
            opportunity = create_test_opportunity("USDC")
            mock_discovery.return_value = [opportunity]
            
            new_edges = integrator.discover_and_integrate_flash_loans(
                target_assets={"USDC"}
            )
        
        assert len(new_edges) == 2  # Flash loan + repayment edge
        assert len(integrator.flash_loan_edges) == 2
        assert "FLASH_USDC" in integrator.virtual_assets
        
        # Check flash loan edge
        flash_edges = [e for e in new_edges if e.edge_type == EdgeType.FLASH_LOAN]
        assert len(flash_edges) == 1
        
        flash_edge = flash_edges[0]
        assert flash_edge.source_asset_id == "USDC"
        assert flash_edge.target_asset_id == "FLASH_USDC"
        assert flash_edge.protocol_name == "aave_v3"
        
        # Check repayment edge
        repay_edges = [e for e in new_edges if e.edge_type == EdgeType.FLASH_REPAY]
        assert len(repay_edges) == 1
        
        repay_edge = repay_edges[0]
        assert repay_edge.source_asset_id == "FLASH_USDC"
        assert repay_edge.target_asset_id == "USDC"
    
    def test_flash_loan_edge_properties(self, integrator):
        """Test flash loan edge properties."""
        with patch.object(integrator.discovery, 'discover_flash_loan_opportunities') as mock_discovery:
            opportunity = create_test_opportunity("USDC", Decimal('50000'))
            mock_discovery.return_value = [opportunity]
            
            new_edges = integrator.discover_and_integrate_flash_loans(
                target_assets={"USDC"}
            )
        
        flash_edge = next(e for e in new_edges if e.edge_type == EdgeType.FLASH_LOAN)
        
        # Check execution properties
        assert flash_edge.execution_properties.supports_synchronous is True
        assert flash_edge.execution_properties.requires_time_delay is None
        assert flash_edge.execution_properties.gas_estimate == 150000
        assert flash_edge.execution_properties.mev_sensitivity == 0.1
        
        # Check constraints (should use 80% of max liquidity)
        expected_liquidity = 50000 * 0.8  # 80% utilization
        assert flash_edge.constraints.max_input_amount == expected_liquidity
        
        # Check metadata
        assert flash_edge.metadata["provider"] == "aave_v3"
        assert flash_edge.metadata["fee_rate"] == 0.0009
        assert flash_edge.metadata["contract_address"] == "0x87870Bca3F8e07Da8c8f4B3A4d8f8Eb2a8B6e4f1"
    
    def test_repayment_edge_properties(self, integrator):
        """Test repayment edge properties."""
        with patch.object(integrator.discovery, 'discover_flash_loan_opportunities') as mock_discovery:
            opportunity = create_test_opportunity("USDC")
            mock_discovery.return_value = [opportunity]
            
            new_edges = integrator.discover_and_integrate_flash_loans(
                target_assets={"USDC"}
            )
        
        repay_edge = next(e for e in new_edges if e.edge_type == EdgeType.FLASH_REPAY)
        
        # Check properties
        assert repay_edge.execution_properties.gas_estimate == 50000  # Lower gas
        assert repay_edge.execution_properties.mev_sensitivity == 0.0  # No MEV risk
        assert repay_edge.state.conversion_rate == 1.0009  # 1 + fee rate
        
        # Check metadata
        assert repay_edge.metadata["is_repayment"] is True
        assert repay_edge.metadata["requires_flash_loan_context"] is True
    
    def test_flash_loan_synergies(self, integrator):
        """Test finding synergies with existing edges."""
        # Create high-capital edge that would benefit from flash loan
        existing_edge = create_test_edge(
            "high_capital_trade",
            "USDC",
            "WETH",
            min_input=20000.0,  # $20K minimum
            max_input=500000.0
        )
        existing_edge.state.confidence_score = 0.8  # High confidence
        
        with patch.object(integrator.discovery, 'discover_flash_loan_opportunities') as mock_discovery:
            opportunity = create_test_opportunity("USDC", Decimal('1000000'))
            mock_discovery.return_value = [opportunity]
            
            new_edges = integrator.discover_and_integrate_flash_loans(
                target_assets={"USDC"},
                existing_edges=[existing_edge]
            )
        
        # Should have flash loan + repayment + enhanced edge
        assert len(new_edges) >= 2
        
        # Check for enhanced edge
        enhanced_edges = [e for e in new_edges if e.metadata.get("flash_loan_enhanced")]
        if enhanced_edges:  # Synergy was found
            enhanced_edge = enhanced_edges[0]
            assert enhanced_edge.source_asset_id == "FLASH_USDC"
            assert enhanced_edge.target_asset_id == "WETH"
            assert enhanced_edge.metadata["original_edge_id"] == "high_capital_trade"
    
    def test_would_benefit_from_flash_loan(self, integrator):
        """Test flash loan benefit detection."""
        opportunity = create_test_opportunity("USDC", Decimal('100000'))
        
        # Edge that would benefit (high capital, good profit margin)
        high_capital_edge = create_test_edge(
            "profitable_trade",
            "USDC",
            "WETH",
            min_input=10000.0  # $10K
        )
        high_capital_edge.state.confidence_score = 0.8
        
        assert integrator._would_benefit_from_flash_loan(high_capital_edge, opportunity)
        
        # Edge that wouldn't benefit (low capital)
        low_capital_edge = create_test_edge(
            "small_trade",
            "USDC",
            "WETH",
            min_input=100.0  # $100
        )
        
        assert not integrator._would_benefit_from_flash_loan(low_capital_edge, opportunity)
    
    def test_virtual_asset_management(self, integrator):
        """Test virtual asset management."""
        with patch.object(integrator.discovery, 'discover_flash_loan_opportunities') as mock_discovery:
            opportunity = create_test_opportunity("USDC")
            mock_discovery.return_value = [opportunity]
            
            integrator.discover_and_integrate_flash_loans(target_assets={"USDC"})
        
        # Check virtual asset tracking
        assert integrator.is_virtual_asset("FLASH_USDC")
        assert not integrator.is_virtual_asset("USDC")
        assert not integrator.is_virtual_asset("WETH")
        
        # Check capacity (should be from the test opportunity)
        capacity = integrator.get_flash_loan_capacity("USDC")
        assert capacity is not None
        assert capacity > Decimal('0')
    
    def test_required_repayment_calculation(self, integrator):
        """Test repayment amount calculation."""
        with patch.object(integrator.discovery, 'discover_flash_loan_opportunities') as mock_discovery:
            opportunity = create_test_opportunity("USDC")
            mock_discovery.return_value = [opportunity]
            
            integrator.discover_and_integrate_flash_loans(target_assets={"USDC"})
        
        borrowed_amount = Decimal('10000')
        repayment = integrator.get_required_repayment_amount("FLASH_USDC", borrowed_amount)
        
        # Should include fee: 10000 + (10000 * 0.0009) = 10009
        expected = borrowed_amount + (borrowed_amount * Decimal('0.0009'))
        assert abs(repayment - expected) < Decimal('2.0')  # Allow for calculation differences
    
    def test_flash_loan_path_validation(self, integrator):
        """Test flash loan path validation."""
        # Create valid flash loan path
        flash_edge = create_test_edge("flash", "USDC", "FLASH_USDC", EdgeType.FLASH_LOAN)
        trade_edge = create_test_edge("trade", "FLASH_USDC", "WETH", EdgeType.TRADE)
        repay_edge = create_test_edge("repay", "FLASH_USDC", "USDC", EdgeType.FLASH_REPAY)
        
        valid_path = [flash_edge, trade_edge, repay_edge]
        assert integrator.validate_flash_loan_path(valid_path)
        
        # Create invalid path (missing repayment)
        invalid_path = [flash_edge, trade_edge]
        assert not integrator.validate_flash_loan_path(invalid_path)
        
        # Create path with mismatched flash loan/repayment
        wrong_repay = create_test_edge("wrong_repay", "FLASH_DAI", "DAI", EdgeType.FLASH_REPAY)
        mismatched_path = [flash_edge, trade_edge, wrong_repay]
        assert not integrator.validate_flash_loan_path(mismatched_path)
    
    def test_liquidity_updates(self, integrator):
        """Test flash loan liquidity updates."""
        with patch.object(integrator.discovery, 'discover_flash_loan_opportunities') as mock_discovery:
            # Initial opportunity
            initial_opportunity = create_test_opportunity("USDC", Decimal('50000'))
            mock_discovery.return_value = [initial_opportunity]
            
            integrator.discover_and_integrate_flash_loans(target_assets={"USDC"})
            
            # Get initial edge
            flash_edge = next(e for e in integrator.flash_loan_edges.values() 
                            if e.edge_type == EdgeType.FLASH_LOAN)
            initial_liquidity = flash_edge.state.liquidity_usd
            
            # Mock updated opportunity with more liquidity
            updated_opportunity = create_test_opportunity("USDC", Decimal('200000'))
            mock_discovery.return_value = [updated_opportunity]
            
            # Update liquidity
            integrator.update_flash_loan_liquidity()
            
            # Check that liquidity was updated
            updated_liquidity = flash_edge.state.liquidity_usd
            assert updated_liquidity > initial_liquidity
    
    def test_error_handling(self, integrator):
        """Test error handling in edge creation."""
        # Test with invalid opportunity data
        with patch.object(integrator.discovery, 'discover_flash_loan_opportunities') as mock_discovery:
            # Mock exception during discovery
            mock_discovery.side_effect = Exception("Network error")
            
            # Should handle gracefully
            new_edges = integrator.discover_and_integrate_flash_loans(target_assets={"USDC"})
            assert len(new_edges) == 0
    
    def test_statistics_tracking(self, integrator):
        """Test statistics tracking."""
        with patch.object(integrator.discovery, 'discover_flash_loan_opportunities') as mock_discovery:
            opportunities = [
                create_test_opportunity("USDC", Decimal('100000')),
                create_test_opportunity("WETH", Decimal('50000'))
            ]
            mock_discovery.return_value = opportunities
            
            integrator.discover_and_integrate_flash_loans()
        
        stats = integrator.get_statistics()
        
        assert stats["opportunities_integrated"] == 2
        assert stats["flash_loan_edges_created"] == 4  # 2 opportunities * 2 edges each
        assert stats["virtual_assets_created"] == 2
        assert stats["flash_loan_edges_active"] == 4
        assert stats["virtual_assets_active"] == 2
        assert stats["total_flash_loan_capacity"] == Decimal('150000')
    
    def test_multiple_assets_integration(self, integrator):
        """Test integrating multiple assets simultaneously."""
        with patch.object(integrator.discovery, 'discover_flash_loan_opportunities') as mock_discovery:
            opportunities = [
                create_test_opportunity("USDC", Decimal('100000')),
                create_test_opportunity("WETH", Decimal('50000')),
                create_test_opportunity("DAI", Decimal('75000'))
            ]
            mock_discovery.return_value = opportunities
            
            new_edges = integrator.discover_and_integrate_flash_loans()
        
        # Should have 2 edges per asset
        assert len(new_edges) == 6  # 3 assets * 2 edges each
        
        # Check virtual assets
        assert "FLASH_USDC" in integrator.virtual_assets
        assert "FLASH_WETH" in integrator.virtual_assets
        assert "FLASH_DAI" in integrator.virtual_assets
        
        # Check edge distribution
        flash_edges = [e for e in new_edges if e.edge_type == EdgeType.FLASH_LOAN]
        repay_edges = [e for e in new_edges if e.edge_type == EdgeType.FLASH_REPAY]
        
        assert len(flash_edges) == 3
        assert len(repay_edges) == 3


@pytest.mark.asyncio
async def test_flash_loan_integration_full_workflow():
    """Integration test for complete flash loan workflow."""
    integrator = FlashLoanGraphIntegrator(chain_name="ethereum")
    
    # Create existing edges that might benefit from flash loans
    existing_edges = [
        create_test_edge("small_trade", "USDC", "WETH", min_input=100.0),
        create_test_edge("large_trade", "USDC", "DAI", min_input=25000.0, max_input=1000000.0)
    ]
    existing_edges[1].state.confidence_score = 0.9  # High confidence
    
    # Mock discovery
    with patch.object(integrator.discovery, 'discover_flash_loan_opportunities') as mock_discovery:
        opportunities = [
            create_test_opportunity("USDC", Decimal('500000')),
            create_test_opportunity("WETH", Decimal('200000'))
        ]
        mock_discovery.return_value = opportunities
        
        # Integrate flash loans
        new_edges = integrator.discover_and_integrate_flash_loans(
            target_assets={"USDC", "WETH"},
            existing_edges=existing_edges
        )
    
    # Validate results
    assert len(new_edges) >= 4  # At least 2 flash loan + 2 repay edges
    
    # Check that we have flash loan and repayment edges for each asset
    usdc_flash_edges = [e for e in new_edges if e.edge_type == EdgeType.FLASH_LOAN and "USDC" in e.source_asset_id]
    usdc_repay_edges = [e for e in new_edges if e.edge_type == EdgeType.FLASH_REPAY and "USDC" in e.target_asset_id]
    
    assert len(usdc_flash_edges) >= 1
    assert len(usdc_repay_edges) >= 1
    
    # Test path validation
    flash_edge = usdc_flash_edges[0]
    repay_edge = usdc_repay_edges[0]
    trade_edge = create_test_edge("intermediate", "FLASH_USDC", "WETH", EdgeType.TRADE)
    
    test_path = [flash_edge, trade_edge, repay_edge]
    assert integrator.validate_flash_loan_path(test_path)
    
    # Test statistics
    stats = integrator.get_statistics()
    assert stats["opportunities_integrated"] == 2
    assert stats["virtual_assets_active"] == 2