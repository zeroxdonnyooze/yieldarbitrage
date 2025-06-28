"""
Unit tests for MEV Opportunity Detection.

Tests the MEV opportunity detection components including opportunity models,
transaction analysis, and back-run edge integration.
"""
import pytest
import asyncio
import time
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from yield_arbitrage.mev_detection.opportunity_models import (
    BackRunOpportunity, SandwichOpportunity, ArbitrageOpportunity,
    MEVOpportunityType, OpportunityStatus, OpportunityQueue
)
from yield_arbitrage.mev_detection.transaction_analyzer import (
    TransactionAnalyzer, TransactionCategory, TransactionImpact
)
from yield_arbitrage.mev_detection.opportunity_detector import (
    MEVOpportunityDetector, OpportunityDetectionConfig, DetectedOpportunity
)
from yield_arbitrage.graph_engine.models import BackRunEdge, EdgeType


class TestOpportunityModels:
    """Test MEV opportunity data models."""
    
    def test_back_run_opportunity_creation(self):
        """Test BackRunOpportunity creation and validation."""
        opportunity = BackRunOpportunity(
            opportunity_id="test_backrun_001",
            target_transaction_hash="0x1234567890abcdef",
            estimated_profit_usd=100.0,
            confidence_score=0.8,
            required_capital_usd=1000.0,
            max_gas_price=int(200e9),
            execution_deadline=time.time() + 300,
            chain_id=1,
            source_asset="WETH",
            target_asset="USDC",
            optimal_amount=2.5,
            expected_price_movement=0.015,
            pool_address="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
        )
        
        assert opportunity.opportunity_type == MEVOpportunityType.BACK_RUN
        assert opportunity.estimated_profit_usd == 100.0
        assert opportunity.source_asset == "WETH"
        assert opportunity.target_asset == "USDC"
        assert not opportunity.is_expired()
        assert opportunity.profit_to_capital_ratio() == 0.1
    
    def test_opportunity_expiration(self):
        """Test opportunity expiration logic."""
        # Create expired opportunity
        opportunity = BackRunOpportunity(
            opportunity_id="test_expired",
            target_transaction_hash="0x1234567890abcdef",
            estimated_profit_usd=50.0,
            confidence_score=0.7,
            required_capital_usd=500.0,
            max_gas_price=int(150e9),
            execution_deadline=time.time() + 300,
            chain_id=1,
            source_asset="WETH",
            target_asset="USDC",
            optimal_amount=1.0,
            expected_price_movement=0.01,
            pool_address="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
            expires_at=time.time() - 60  # Expired 1 minute ago
        )
        
        assert opportunity.is_expired()
        assert opportunity.time_to_expiry() == 0
    
    def test_opportunity_queue_prioritization(self):
        """Test opportunity queue prioritization by profit."""
        queue = OpportunityQueue(max_size=3)
        
        # Create opportunities with different profits
        opportunities = [
            BackRunOpportunity(
                opportunity_id=f"test_{i}",
                target_transaction_hash=f"0x{i:040x}",
                estimated_profit_usd=profit,
                confidence_score=0.7,
                required_capital_usd=profit * 10,
                max_gas_price=int(150e9),
                execution_deadline=time.time() + 300,
                chain_id=1,
                source_asset="WETH",
                target_asset="USDC",
                optimal_amount=1.0,
                expected_price_movement=0.01,
                pool_address="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
            )
            for i, profit in enumerate([50.0, 150.0, 100.0])
        ]
        
        # Add to queue
        for opp in opportunities:
            queue.add_opportunity(opp)
        
        # Should get highest profit first
        first = queue.get_next_opportunity()
        assert first.estimated_profit_usd == 150.0
        
        second = queue.get_next_opportunity()
        assert second.estimated_profit_usd == 100.0
        
        third = queue.get_next_opportunity()
        assert third.estimated_profit_usd == 50.0
    
    def test_queue_size_limit(self):
        """Test opportunity queue size limits."""
        queue = OpportunityQueue(max_size=2)
        
        # Add 3 opportunities
        opportunities = [
            BackRunOpportunity(
                opportunity_id=f"test_{i}",
                target_transaction_hash=f"0x{i:040x}",
                estimated_profit_usd=profit,
                confidence_score=0.7,
                required_capital_usd=1000.0,
                max_gas_price=int(150e9),
                execution_deadline=time.time() + 300,
                chain_id=1,
                source_asset="WETH",
                target_asset="USDC",
                optimal_amount=1.0,
                expected_price_movement=0.01,
                pool_address="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
            )
            for i, profit in enumerate([50.0, 150.0, 100.0])
        ]
        
        results = [queue.add_opportunity(opp) for opp in opportunities]
        
        # Should accept first two, reject lowest profit
        assert all(results[:2])  # First two added
        assert len(queue.opportunities) == 2
        
        # Should contain highest profit opportunities
        profits = [opp.estimated_profit_usd for opp in queue.opportunities]
        assert 150.0 in profits
        assert 100.0 in profits
        assert 50.0 not in profits


class TestTransactionAnalyzer:
    """Test transaction analyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create transaction analyzer instance."""
        return TransactionAnalyzer(chain_id=1)
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.chain_id == 1
        assert len(analyzer.function_signatures) > 0
        assert len(analyzer.protocol_contracts) > 0
    
    @pytest.mark.asyncio
    async def test_dex_trade_analysis(self, analyzer):
        """Test analysis of DEX trade transaction."""
        # Mock Uniswap V2 swap transaction
        tx_data = {
            'hash': '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
            'to': '0x7a250d5630b4cf539739df2c5dacb4c659f2488d',  # Uniswap V2 Router
            'from': '0xabcdefabcdefabcdefabcdefabcdefabcdefabcd',
            'value': int(10 * 1e18),  # 10 ETH
            'gasPrice': int(100e9),   # 100 gwei
            'gas': 300000,
            'input': '0x38ed1739' + '0' * 200,  # swapExactTokensForTokens
            'nonce': 42
        }
        
        impact = await analyzer.analyze_transaction(tx_data)
        
        assert impact.transaction_hash == tx_data['hash']
        assert impact.category == TransactionCategory.DEX_TRADE
        assert impact.total_value_usd > 0
        assert impact.metadata.get('function_name') == 'swapExactTokensForTokens'
        assert impact.metadata.get('protocol') == 'uniswap_v2'
        assert impact.metadata.get('base_mev_risk') == 0.8
    
    @pytest.mark.asyncio
    async def test_large_trade_impact(self, analyzer):
        """Test analysis of large trade with high MEV risk."""
        # Mock large value transaction
        tx_data = {
            'hash': '0x2345678901bcdef02345678901bcdef02345678901bcdef02345678901bcdef0',
            'to': '0x7a250d5630b4cf539739df2c5dacb4c659f2488d',
            'from': '0xabcdefabcdefabcdefabcdefabcdefabcdefabcd',
            'value': int(200 * 1e18),  # 200 ETH - large trade
            'gasPrice': int(150e9),    # High gas price
            'gas': 500000,
            'input': '0x38ed1739' + '0' * 200,
            'nonce': 123
        }
        
        impact = await analyzer.analyze_transaction(tx_data)
        
        assert impact.total_value_usd >= analyzer.large_trade_threshold
        assert impact.metadata.get('large_trade') is True
        assert impact.max_price_impact > 0
        assert impact.metadata.get('mev_risk_score', 0) > 0.8
    
    @pytest.mark.asyncio
    async def test_token_transfer_analysis(self, analyzer):
        """Test analysis of simple token transfer."""
        # Mock ERC20 transfer
        tx_data = {
            'hash': '0x3456789012cdef003456789012cdef003456789012cdef003456789012cdef00',
            'to': '0xa0b86a33e6441b9435b654f6d26cc98b6e1d0a3a',  # Random token
            'from': '0xabcdefabcdefabcdefabcdefabcdefabcdefabcd',
            'value': 0,
            'gasPrice': int(50e9),
            'gas': 21000,
            'input': '0xa9059cbb' + '0' * 128,  # transfer function
            'nonce': 456
        }
        
        impact = await analyzer.analyze_transaction(tx_data)
        
        assert impact.category == TransactionCategory.TOKEN_TRANSFER
        assert impact.metadata.get('function_name') == 'transfer'
        assert impact.metadata.get('base_mev_risk') == 0.1


class TestOpportunityDetector:
    """Test MEV opportunity detector."""
    
    @pytest.fixture
    def config(self):
        """Create detector configuration."""
        return OpportunityDetectionConfig(
            min_profit_usd=25.0,
            min_confidence_score=0.5,
            detect_back_runs=True,
            detect_sandwich_attacks=True,
            detect_arbitrage=True
        )
    
    @pytest.fixture
    def detector(self, config):
        """Create opportunity detector instance."""
        return MEVOpportunityDetector(config, chain_id=1)
    
    def test_detector_initialization(self, detector, config):
        """Test detector initialization."""
        assert detector.config == config
        assert detector.chain_id == 1
        assert not detector.is_running
        assert len(detector.detected_opportunities) == 0
    
    @pytest.mark.asyncio
    async def test_back_run_detection(self, detector):
        """Test back-run opportunity detection."""
        # Create mock transaction impact with arbitrage opportunity
        impact = TransactionImpact(
            transaction_hash="0x1234567890abcdef",
            category=TransactionCategory.DEX_TRADE,
            total_value_usd=50000.0,
            max_price_impact=0.02,
            creates_arbitrage_opportunity=True,
            affected_pools=[
                Mock(
                    pool_address="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
                    protocol="uniswap_v3"
                )
            ]
        )
        impact.metadata = {"mev_risk_score": 0.8}
        
        # Detect opportunities
        opportunities = await detector._detect_opportunities(impact)
        
        # Should detect back-run opportunity
        back_runs = [op for op in opportunities if op.opportunity_type == MEVOpportunityType.BACK_RUN]
        assert len(back_runs) > 0
        
        back_run = back_runs[0]
        assert isinstance(back_run, BackRunOpportunity)
        assert back_run.estimated_profit_usd > 0
        assert back_run.target_transaction_hash == impact.transaction_hash
    
    @pytest.mark.asyncio
    async def test_sandwich_detection(self, detector):
        """Test sandwich opportunity detection."""
        # Create mock vulnerable transaction
        impact = TransactionImpact(
            transaction_hash="0x2345678901bcdef0",
            category=TransactionCategory.DEX_TRADE,
            total_value_usd=25000.0,
            max_price_impact=0.008,  # Sufficient for sandwich
            sandwich_vulnerable=True
        )
        impact.metadata = {"mev_risk_score": 0.7}
        
        opportunities = await detector._detect_opportunities(impact)
        
        # Should detect sandwich opportunity
        sandwiches = [op for op in opportunities if op.opportunity_type == MEVOpportunityType.SANDWICH]
        assert len(sandwiches) > 0
        
        sandwich = sandwiches[0]
        assert isinstance(sandwich, SandwichOpportunity)
        assert sandwich.estimated_profit_usd > 0
        assert sandwich.victim_slippage_tolerance > 0
    
    @pytest.mark.asyncio
    async def test_opportunity_validation(self, detector):
        """Test opportunity validation logic."""
        # Create low-profit opportunity (should be rejected)
        low_profit_opportunity = BackRunOpportunity(
            opportunity_id="low_profit_test",
            target_transaction_hash="0x1234567890abcdef",
            estimated_profit_usd=10.0,  # Below minimum
            confidence_score=0.8,
            required_capital_usd=1000.0,
            max_gas_price=int(150e9),
            execution_deadline=time.time() + 300,
            chain_id=1,
            source_asset="WETH",
            target_asset="USDC",
            optimal_amount=1.0,
            expected_price_movement=0.01,
            pool_address="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
        )
        
        detected_low = DetectedOpportunity(
            opportunity=low_profit_opportunity,
            detection_confidence=0.8,
            detection_timestamp=time.time(),
            source_transaction=Mock()
        )
        
        # Should reject low profit
        assert not await detector._validate_opportunity(detected_low)
        
        # Create valid opportunity
        valid_opportunity = BackRunOpportunity(
            opportunity_id="valid_test",
            target_transaction_hash="0x1234567890abcdef",
            estimated_profit_usd=100.0,  # Above minimum
            confidence_score=0.8,
            required_capital_usd=2000.0,
            max_gas_price=int(150e9),
            execution_deadline=time.time() + 300,
            chain_id=1,
            source_asset="WETH",
            target_asset="USDC",
            optimal_amount=2.0,
            expected_price_movement=0.015,
            pool_address="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
        )
        
        detected_valid = DetectedOpportunity(
            opportunity=valid_opportunity,
            detection_confidence=0.8,
            detection_timestamp=time.time(),
            source_transaction=Mock()
        )
        
        # Should accept valid opportunity
        assert await detector._validate_opportunity(detected_valid)


class TestBackRunEdgeIntegration:
    """Test integration with BackRunEdge from graph engine."""
    
    def test_back_run_edge_creation_from_opportunity(self):
        """Test creating BackRunEdge from opportunity."""
        opportunity = BackRunOpportunity(
            opportunity_id="integration_test",
            target_transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            estimated_profit_usd=150.0,
            confidence_score=0.85,
            required_capital_usd=5000.0,
            max_gas_price=int(180e9),
            execution_deadline=time.time() + 180,
            chain_id=1,
            source_asset="WETH",
            target_asset="USDC",
            optimal_amount=3.0,
            expected_price_movement=0.018,
            pool_address="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
        )
        
        # Create corresponding BackRunEdge
        edge = BackRunEdge(
            chain_name="ethereum",
            target_transaction=opportunity.target_transaction_hash,
            source_asset=opportunity.source_asset,
            target_asset=opportunity.target_asset,
            expected_profit=opportunity.estimated_profit_usd
        )
        
        assert edge.edge_type == EdgeType.BACK_RUN
        assert edge.protocol_name == "MEV_BACKRUN"
        assert edge.target_transaction == opportunity.target_transaction_hash
        assert edge.expected_profit == opportunity.estimated_profit_usd
        assert edge.execution_properties.mev_sensitivity == 0.0  # No frontrun risk
        assert edge.execution_properties.gas_estimate == 200000
    
    def test_back_run_edge_calculation(self):
        """Test BackRunEdge output calculation."""
        edge = BackRunEdge(
            chain_name="ethereum",
            target_transaction="0x1234567890abcdef",
            source_asset="WETH",
            target_asset="USDC",
            expected_profit=200.0
        )
        
        # Test calculation with valid input
        result = edge.calculate_output(5.0)  # 5 ETH input
        
        assert result["output_amount"] > 5.0  # Should include profit
        assert result["expected_profit_usd"] == 200.0
        assert result["target_transaction"] == "0x1234567890abcdef"
        assert result["effective_rate"] > 1.0  # Profitable rate
        assert result["confidence"] == 0.7  # Expected confidence for back-runs
    
    def test_back_run_edge_validation(self):
        """Test BackRunEdge input validation."""
        edge = BackRunEdge(
            chain_name="ethereum",
            target_transaction="0x1234567890abcdef",
            source_asset="WETH",
            target_asset="USDC",
            expected_profit=100.0
        )
        
        # Test invalid inputs
        zero_result = edge.calculate_output(0.0)
        assert zero_result["output_amount"] == 0.0
        assert "error" in zero_result
        
        negative_result = edge.calculate_output(-1.0)
        assert negative_result["output_amount"] == 0.0
        assert "error" in negative_result


if __name__ == "__main__":
    # Run basic test
    print("🧪 Testing MEV Opportunity Detection")
    print("=" * 50)
    
    # Test opportunity creation
    opportunity = BackRunOpportunity(
        opportunity_id="test_001",
        target_transaction_hash="0x1234567890abcdef",
        estimated_profit_usd=100.0,
        confidence_score=0.8,
        required_capital_usd=1000.0,
        max_gas_price=int(200e9),
        execution_deadline=time.time() + 300,
        chain_id=1,
        source_asset="WETH",
        target_asset="USDC",
        optimal_amount=2.5,
        expected_price_movement=0.015,
        pool_address="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
    )
    
    print(f"✅ BackRunOpportunity created:")
    print(f"   - Opportunity Type: {opportunity.opportunity_type}")
    print(f"   - Estimated Profit: ${opportunity.estimated_profit_usd}")
    print(f"   - Confidence Score: {opportunity.confidence_score}")
    print(f"   - Is Expired: {opportunity.is_expired()}")
    
    # Test BackRunEdge integration
    edge = BackRunEdge(
        chain_name="ethereum",
        target_transaction=opportunity.target_transaction_hash,
        source_asset=opportunity.source_asset,
        target_asset=opportunity.target_asset,
        expected_profit=opportunity.estimated_profit_usd
    )
    
    print(f"✅ BackRunEdge created:")
    print(f"   - Edge Type: {edge.edge_type}")
    print(f"   - Expected Profit: ${edge.expected_profit}")
    print(f"   - MEV Sensitivity: {edge.execution_properties.mev_sensitivity}")
    
    print("\\n✅ MEV opportunity detection test passed!")