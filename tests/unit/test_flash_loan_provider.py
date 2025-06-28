"""
Unit tests for flash loan provider and discovery functionality.

Tests flash loan discovery, terms calculation, and provider comparison
across different protocols and chains.
"""
import pytest
from decimal import Decimal
from unittest.mock import Mock, patch

from yield_arbitrage.protocols.flash_loan_provider import (
    FlashLoanProvider, FlashLoanTerms, FlashLoanOpportunity, FlashLoanDiscovery,
    FlashLoanEdgeGenerator
)


class TestFlashLoanDiscovery:
    """Test suite for FlashLoanDiscovery."""
    
    @pytest.fixture
    def discovery(self):
        """Create a FlashLoanDiscovery instance."""
        return FlashLoanDiscovery(chain_name="ethereum")
    
    def test_initialization(self, discovery):
        """Test discovery initialization."""
        assert discovery.chain_name == "ethereum"
        assert len(discovery.provider_configs) > 0
        assert FlashLoanProvider.AAVE_V3 in discovery.provider_configs
        assert FlashLoanProvider.BALANCER in discovery.provider_configs
    
    def test_supported_assets(self, discovery):
        """Test getting supported assets."""
        assets = discovery.get_supported_assets()
        
        assert "USDC" in assets
        assert "WETH" in assets
        assert "DAI" in assets
        assert len(assets) > 0
    
    def test_discover_opportunities_single_asset(self, discovery):
        """Test discovering opportunities for a single asset."""
        opportunities = discovery.discover_flash_loan_opportunities(
            target_assets={"USDC"},
            min_liquidity=Decimal('1000')
        )
        
        assert len(opportunities) > 0
        opportunity = opportunities[0]
        
        assert opportunity.asset_id == "USDC"
        assert len(opportunity.available_providers) > 0
        assert opportunity.best_terms is not None
        assert opportunity.total_max_liquidity > Decimal('1000')
    
    def test_discover_opportunities_all_assets(self, discovery):
        """Test discovering opportunities for all assets."""
        opportunities = discovery.discover_flash_loan_opportunities(
            min_liquidity=Decimal('1000')
        )
        
        assert len(opportunities) > 0
        
        # Check that all opportunities have valid data
        for opp in opportunities:
            assert opp.asset_id
            assert len(opp.available_providers) > 0
            assert opp.best_terms.provider in FlashLoanProvider
            assert opp.total_max_liquidity > Decimal('0')
    
    def test_provider_terms_aave_v3(self, discovery):
        """Test Aave V3 terms calculation."""
        config = discovery.provider_configs[FlashLoanProvider.AAVE_V3]
        terms = discovery._get_aave_v3_terms("USDC", config)
        
        assert terms.provider == FlashLoanProvider.AAVE_V3
        assert terms.asset == "USDC"
        assert terms.fee_rate == Decimal("0.0009")  # 0.09%
        assert terms.max_amount > Decimal('0')
        assert terms.gas_estimate > 0
        assert not terms.requires_collateral
    
    def test_provider_terms_balancer(self, discovery):
        """Test Balancer terms calculation."""
        config = discovery.provider_configs[FlashLoanProvider.BALANCER]
        terms = discovery._get_balancer_terms("USDC", config)
        
        assert terms.provider == FlashLoanProvider.BALANCER
        assert terms.asset == "USDC"
        assert terms.fee_rate == Decimal("0")  # No fees
        assert terms.max_amount > Decimal('0')
        assert terms.gas_estimate > 0
    
    def test_best_flash_loan_selection(self, discovery):
        """Test selection of best flash loan provider."""
        opportunities = discovery.discover_flash_loan_opportunities(
            target_assets={"USDC"}
        )
        
        opportunity = opportunities[0]
        best_terms = opportunity.best_terms
        
        # Best terms should have lowest total cost
        for terms in opportunity.available_providers:
            test_amount = Decimal('10000')
            best_cost = (test_amount * best_terms.fee_rate) + best_terms.fixed_fee
            current_cost = (test_amount * terms.fee_rate) + terms.fixed_fee
            
            # Best should be <= current (allowing for ties)
            assert best_cost <= current_cost
    
    def test_get_best_flash_loan_for_amount(self, discovery):
        """Test getting best flash loan for specific amount."""
        # Test with amount that should be available
        terms = discovery.get_best_flash_loan_for_amount("USDC", Decimal('10000'))
        
        assert terms is not None
        assert terms.max_amount >= Decimal('10000')
        assert terms.provider in FlashLoanProvider
    
    def test_get_best_flash_loan_insufficient_liquidity(self, discovery):
        """Test getting flash loan when amount exceeds liquidity."""
        # Test with very large amount
        terms = discovery.get_best_flash_loan_for_amount("USDC", Decimal('1000000000'))
        
        # Should return None or terms with sufficient liquidity
        if terms:
            assert terms.max_amount >= Decimal('1000000000')
    
    def test_flash_loan_cost_calculation(self, discovery):
        """Test flash loan cost calculation."""
        terms = FlashLoanTerms(
            provider=FlashLoanProvider.AAVE_V3,
            asset="USDC",
            max_amount=Decimal('100000'),
            fee_rate=Decimal('0.001'),  # 0.1%
            fixed_fee=Decimal('10'),
            min_amount=Decimal('1'),
            gas_estimate=150000
        )
        
        amount = Decimal('50000')
        cost = discovery.calculate_flash_loan_cost(terms, amount)
        
        expected_cost = (amount * terms.fee_rate) + terms.fixed_fee
        assert cost == expected_cost
        assert cost == Decimal('60')  # 50 + 10
    
    def test_flash_loan_profitability_check(self, discovery):
        """Test profitability check."""
        terms = FlashLoanTerms(
            provider=FlashLoanProvider.AAVE_V3,
            asset="USDC",
            max_amount=Decimal('100000'),
            fee_rate=Decimal('0.001'),  # 0.1%
            fixed_fee=Decimal('10'),
            min_amount=Decimal('1'),
            gas_estimate=150000
        )
        
        amount = Decimal('10000')
        
        # Profitable scenario
        high_profit = Decimal('200')  # $200 profit vs ~$20 cost
        assert discovery.is_flash_loan_profitable(terms, amount, high_profit)
        
        # Unprofitable scenario
        low_profit = Decimal('5')   # $5 profit vs ~$20 cost
        assert not discovery.is_flash_loan_profitable(terms, amount, low_profit)
    
    def test_statistics_tracking(self, discovery):
        """Test statistics tracking."""
        # Run discovery to generate stats
        discovery.discover_flash_loan_opportunities(target_assets={"USDC", "WETH"})
        
        stats = discovery.get_statistics()
        
        assert "opportunities_discovered" in stats
        assert "providers_checked" in stats
        assert "total_liquidity_discovered" in stats
        assert "cached_opportunities" in stats
        assert "providers_configured" in stats
        assert "supported_assets" in stats
        
        assert stats["opportunities_discovered"] > 0
        assert stats["providers_checked"] > 0
    
    def test_cache_management(self, discovery):
        """Test opportunity caching."""
        # Discover opportunities
        opportunities = discovery.discover_flash_loan_opportunities(target_assets={"USDC"})
        
        assert len(discovery.cached_opportunities) > 0
        assert "USDC" in discovery.cached_opportunities
        
        # Clear cache
        discovery.clear_cache()
        assert len(discovery.cached_opportunities) == 0
    
    def test_different_chain_initialization(self):
        """Test initialization for different chains."""
        polygon_discovery = FlashLoanDiscovery(chain_name="polygon")
        
        assert polygon_discovery.chain_name == "polygon"
        # Different chains might have different or no providers
        assert isinstance(polygon_discovery.provider_configs, dict)


class TestFlashLoanEdgeGenerator:
    """Test suite for FlashLoanEdgeGenerator."""
    
    @pytest.fixture
    def discovery(self):
        """Create a FlashLoanDiscovery instance."""
        return FlashLoanDiscovery(chain_name="ethereum")
    
    @pytest.fixture
    def generator(self, discovery):
        """Create a FlashLoanEdgeGenerator instance."""
        return FlashLoanEdgeGenerator(discovery)
    
    @pytest.fixture
    def sample_opportunity(self):
        """Create a sample flash loan opportunity."""
        terms = FlashLoanTerms(
            provider=FlashLoanProvider.AAVE_V3,
            asset="USDC",
            max_amount=Decimal('100000'),
            fee_rate=Decimal('0.0009'),
            fixed_fee=Decimal('0'),
            min_amount=Decimal('1'),
            gas_estimate=150000,
            contract_address="0x87870Bca3F8e07Da8c8f4B3A4d8f8Eb2a8B6e4f1"
        )
        
        return FlashLoanOpportunity(
            asset_id="USDC",
            available_providers=[terms],
            best_terms=terms,
            total_max_liquidity=Decimal('100000')
        )
    
    def test_generate_flash_loan_edges(self, generator, sample_opportunity):
        """Test flash loan edge generation."""
        opportunities = [sample_opportunity]
        edges = generator.generate_flash_loan_edges(opportunities)
        
        assert len(edges) == 2  # Flash loan + repayment edge
        
        # Check flash loan edge
        flash_edge = next(e for e in edges if e["edge_type"] == "FLASH_LOAN")
        assert flash_edge["source_asset_id"] == "USDC"
        assert flash_edge["target_asset_id"] == "FLASH_USDC"
        assert flash_edge["protocol_name"] == "aave_v3"
        
        # Check repayment edge
        repay_edge = next(e for e in edges if e["edge_type"] == "FLASH_REPAY")
        assert repay_edge["source_asset_id"] == "FLASH_USDC"
        assert repay_edge["target_asset_id"] == "USDC"
        assert repay_edge["protocol_name"] == "aave_v3"
    
    def test_flash_loan_edge_properties(self, generator, sample_opportunity):
        """Test flash loan edge properties."""
        opportunities = [sample_opportunity]
        edges = generator.generate_flash_loan_edges(opportunities)
        
        flash_edge = next(e for e in edges if e["edge_type"] == "FLASH_LOAN")
        
        # Check execution properties
        props = flash_edge["execution_properties"]
        assert props["supports_synchronous"] is True
        assert props["requires_time_delay"] is None
        assert props["gas_estimate"] == 150000
        assert props["mev_sensitivity"] == 0.1
        
        # Check constraints
        constraints = flash_edge["constraints"]
        assert constraints["fee_rate"] == 0.0009
        assert constraints["max_input_amount"] == 100000.0
        
        # Check metadata
        metadata = flash_edge["metadata"]
        assert metadata["provider"] == "aave_v3"
        assert metadata["contract_address"] == "0x87870Bca3F8e07Da8c8f4B3A4d8f8Eb2a8B6e4f1"
    
    def test_repayment_edge_properties(self, generator, sample_opportunity):
        """Test repayment edge properties."""
        opportunities = [sample_opportunity]
        edges = generator.generate_flash_loan_edges(opportunities)
        
        repay_edge = next(e for e in edges if e["edge_type"] == "FLASH_REPAY")
        
        # Check execution properties
        props = repay_edge["execution_properties"]
        assert props["supports_synchronous"] is True
        assert props["mev_sensitivity"] == 0.0  # No MEV risk
        assert props["gas_estimate"] == 50000  # Lower gas
        
        # Check metadata
        metadata = repay_edge["metadata"]
        assert metadata["is_repayment"] is True
        assert metadata["provider"] == "aave_v3"
    
    def test_edge_update_functionality(self, generator, sample_opportunity):
        """Test updating existing flash loan edges."""
        # Create initial edges
        opportunities = [sample_opportunity]
        initial_edges = generator.generate_flash_loan_edges(opportunities)
        
        # Mock discovery to return updated terms
        updated_terms = FlashLoanTerms(
            provider=FlashLoanProvider.AAVE_V3,
            asset="USDC",
            max_amount=Decimal('200000'),  # Increased liquidity
            fee_rate=Decimal('0.0005'),    # Lower fee
            fixed_fee=Decimal('0'),
            min_amount=Decimal('1'),
            gas_estimate=150000,
            contract_address="0x87870Bca3F8e07Da8c8f4B3A4d8f8Eb2a8B6e4f1"
        )
        
        updated_opportunity = FlashLoanOpportunity(
            asset_id="USDC",
            available_providers=[updated_terms],
            best_terms=updated_terms,
            total_max_liquidity=Decimal('200000')
        )
        
        with patch.object(generator.discovery, 'discover_flash_loan_opportunities') as mock_discovery:
            mock_discovery.return_value = [updated_opportunity]
            
            updated_edges = generator.update_flash_loan_edges(initial_edges)
        
        # Check that edges were updated
        flash_edge = next(e for e in updated_edges if e["edge_type"] == "FLASH_LOAN")
        assert flash_edge["constraints"]["fee_rate"] == 0.0005
        assert flash_edge["constraints"]["max_input_amount"] == 200000.0


@pytest.mark.asyncio
async def test_flash_loan_integration():
    """Integration test for flash loan discovery and edge generation."""
    discovery = FlashLoanDiscovery(chain_name="ethereum")
    generator = FlashLoanEdgeGenerator(discovery)
    
    # Discover opportunities
    opportunities = discovery.discover_flash_loan_opportunities(
        target_assets={"USDC", "WETH"},
        min_liquidity=Decimal('1000')
    )
    
    assert len(opportunities) > 0
    
    # Generate edges
    edges = generator.generate_flash_loan_edges(opportunities)
    
    # Should have 2 edges per opportunity (flash loan + repay)
    assert len(edges) == len(opportunities) * 2
    
    # Validate edge structure
    for edge in edges:
        assert "edge_id" in edge
        assert "edge_type" in edge
        assert edge["edge_type"] in ["FLASH_LOAN", "FLASH_REPAY"]
        assert "execution_properties" in edge
        assert "constraints" in edge
        assert "metadata" in edge
    
    # Check statistics
    stats = discovery.get_statistics()
    assert stats["opportunities_discovered"] == len(opportunities)
    assert stats["providers_checked"] > 0