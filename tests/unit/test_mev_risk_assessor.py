"""
Unit tests for MEV Risk Assessor.

Tests the MEV risk assessment functionality for edges and paths,
including risk calculations and execution recommendations.
"""
import pytest
from unittest.mock import Mock
from decimal import Decimal

from yield_arbitrage.mev_protection.mev_risk_assessor import (
    MEVRiskAssessor, MEVRiskLevel, EdgeMEVAnalysis, PathMEVAnalysis,
    calculate_edge_mev_risk, assess_path_mev_risk
)
from yield_arbitrage.graph_engine.models import (
    YieldGraphEdge, EdgeType, EdgeState, EdgeExecutionProperties, EdgeConstraints
)
from yield_arbitrage.pathfinding.path_segment_analyzer import PathSegment, SegmentType


class TestMEVRiskAssessor:
    """Test MEV risk assessment functionality."""
    
    @pytest.fixture
    def mev_assessor(self):
        """Create MEV risk assessor instance."""
        return MEVRiskAssessor()
    
    @pytest.fixture
    def sample_trade_edge(self):
        """Create a sample trade edge for testing."""
        return YieldGraphEdge(
            edge_id="uniswap_v3_weth_usdc",
            source_asset_id="WETH",
            target_asset_id="USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v3",
            chain_name="ethereum",
            state=EdgeState(
                conversion_rate=2000.0,
                liquidity_usd=1_000_000.0,
                gas_cost_usd=20.0
            ),
            execution_properties=EdgeExecutionProperties(
                mev_sensitivity=0.7,
                max_slippage_allowed=0.01,
                supports_private_mempool=True
            )
        )
    
    @pytest.fixture
    def sample_lend_edge(self):
        """Create a sample lending edge for testing."""
        return YieldGraphEdge(
            edge_id="aave_usdc_lend",
            source_asset_id="USDC",
            target_asset_id="aUSDC",
            edge_type=EdgeType.LEND,
            protocol_name="aave",
            chain_name="ethereum",
            state=EdgeState(
                conversion_rate=1.0,
                liquidity_usd=10_000_000.0,
                gas_cost_usd=15.0
            ),
            execution_properties=EdgeExecutionProperties(
                mev_sensitivity=0.3,
                supports_private_mempool=True
            )
        )
    
    @pytest.fixture
    def sample_flash_loan_edge(self):
        """Create a sample flash loan edge."""
        return YieldGraphEdge(
            edge_id="aave_flash_loan_usdc",
            source_asset_id="USDC",
            target_asset_id="USDC",
            edge_type=EdgeType.FLASH_LOAN,
            protocol_name="aave",
            chain_name="ethereum",
            state=EdgeState(
                conversion_rate=0.9991,  # Flash loan fee
                liquidity_usd=50_000_000.0
            ),
            execution_properties=EdgeExecutionProperties(
                mev_sensitivity=0.8,
                supports_private_mempool=True
            )
        )


def test_edge_risk_assessment_trade(mev_assessor, sample_trade_edge):
    """Test MEV risk assessment for a trade edge."""
    amount_usd = 100_000
    
    analysis = mev_assessor.assess_edge_risk(sample_trade_edge, amount_usd)
    
    # Verify analysis structure
    assert isinstance(analysis, EdgeMEVAnalysis)
    assert analysis.edge_id == sample_trade_edge.edge_id
    assert 0 <= analysis.final_risk_score <= 1.0
    
    # Trade edges should have significant risk
    assert analysis.base_risk_score == 0.7  # From edge properties
    assert analysis.sandwich_risk > 0  # Large trade should have sandwich risk
    assert analysis.frontrun_risk > 0  # Trades are frontrunnable
    
    # Check risk level categorization
    assert analysis.risk_level in MEVRiskLevel
    
    # Should recommend private execution for high-value trades
    assert analysis.recommended_execution != "public"
    
    # Should have MEV loss estimate
    assert analysis.estimated_mev_loss_bps > 0


def test_edge_risk_assessment_lending(mev_assessor, sample_lend_edge):
    """Test MEV risk assessment for a lending edge."""
    amount_usd = 50_000
    
    analysis = mev_assessor.assess_edge_risk(sample_lend_edge, amount_usd)
    
    # Lending should have lower risk than trading
    assert analysis.final_risk_score < 0.5
    assert analysis.risk_level in [MEVRiskLevel.MINIMAL, MEVRiskLevel.LOW]
    
    # Lower sandwich and frontrun risks for lending
    assert analysis.sandwich_risk == 0  # No sandwich for lending
    assert analysis.frontrun_risk < 0.3


def test_edge_risk_size_impact(mev_assessor, sample_trade_edge):
    """Test that trade size impacts MEV risk."""
    # Small trade
    small_analysis = mev_assessor.assess_edge_risk(sample_trade_edge, 1_000)
    
    # Large trade
    large_analysis = mev_assessor.assess_edge_risk(sample_trade_edge, 1_000_000)
    
    # Large trades should have higher risk
    assert large_analysis.final_risk_score > small_analysis.final_risk_score
    assert large_analysis.size_modifier > small_analysis.size_modifier
    assert large_analysis.liquidity_risk > small_analysis.liquidity_risk


def test_chain_risk_modifiers(mev_assessor, sample_trade_edge):
    """Test chain-specific risk modifiers."""
    amount_usd = 100_000
    
    # Test Ethereum (high MEV)
    eth_edge = sample_trade_edge
    eth_analysis = mev_assessor.assess_edge_risk(eth_edge, amount_usd)
    
    # Test Arbitrum (lower MEV)
    arb_edge = sample_trade_edge.model_copy()
    arb_edge.chain_name = "arbitrum"
    arb_analysis = mev_assessor.assess_edge_risk(arb_edge, amount_usd)
    
    # Ethereum should have higher risk due to chain modifier
    assert eth_analysis.chain_modifier > arb_analysis.chain_modifier
    assert eth_analysis.final_risk_score > arb_analysis.final_risk_score


def test_protocol_risk_modifiers(mev_assessor):
    """Test protocol-specific risk modifiers."""
    amount_usd = 50_000
    
    # Create edges for different protocols
    uniswap_edge = YieldGraphEdge(
        edge_id="test_uniswap",
        source_asset_id="WETH",
        target_asset_id="USDC",
        edge_type=EdgeType.TRADE,
        protocol_name="uniswap_v2",
        chain_name="ethereum"
    )
    
    curve_edge = YieldGraphEdge(
        edge_id="test_curve",
        source_asset_id="USDC",
        target_asset_id="USDT",
        edge_type=EdgeType.TRADE,
        protocol_name="curve",
        chain_name="ethereum"
    )
    
    # Assess both
    uniswap_analysis = mev_assessor.assess_edge_risk(uniswap_edge, amount_usd)
    curve_analysis = mev_assessor.assess_edge_risk(curve_edge, amount_usd)
    
    # Uniswap should have higher risk modifier than Curve
    assert uniswap_analysis.protocol_modifier > curve_analysis.protocol_modifier


def test_path_risk_assessment(mev_assessor, sample_trade_edge, sample_lend_edge):
    """Test MEV risk assessment for a complete path."""
    path = [sample_trade_edge, sample_lend_edge, sample_trade_edge]
    amount_usd = 100_000
    
    path_analysis = mev_assessor.assess_path_risk(path, amount_usd)
    
    # Verify path analysis structure
    assert isinstance(path_analysis, PathMEVAnalysis)
    assert path_analysis.total_edges == 3
    assert len(path_analysis.edge_analyses) == 3
    
    # Path metrics
    assert path_analysis.max_edge_risk > 0
    assert path_analysis.average_edge_risk > 0
    assert path_analysis.compounded_risk >= path_analysis.max_edge_risk
    
    # Should identify critical edges (high-risk trades)
    assert len(path_analysis.critical_edges) > 0
    assert sample_trade_edge.edge_id in path_analysis.critical_edges
    
    # Execution recommendations
    assert path_analysis.recommended_execution_method != "public"
    assert path_analysis.requires_atomic_execution  # Due to high-risk edges


def test_path_compounded_risk(mev_assessor, sample_trade_edge):
    """Test that multiple high-risk edges compound risk."""
    # Single high-risk edge
    single_path = [sample_trade_edge]
    single_analysis = mev_assessor.assess_path_risk(single_path, 100_000)
    
    # Multiple high-risk edges
    multi_path = [sample_trade_edge, sample_trade_edge, sample_trade_edge]
    multi_analysis = mev_assessor.assess_path_risk(multi_path, 100_000)
    
    # Compounded risk should be higher
    assert multi_analysis.compounded_risk > single_analysis.compounded_risk
    assert multi_analysis.overall_risk_level.value >= single_analysis.overall_risk_level.value


def test_segment_risk_analysis(mev_assessor, sample_trade_edge, sample_lend_edge):
    """Test risk analysis by path segments."""
    path = [sample_trade_edge, sample_lend_edge]
    
    # Create segments
    segment1 = PathSegment(
        segment_id="high_risk_segment",
        segment_type=SegmentType.ATOMIC,
        edges=[sample_trade_edge],
        start_index=0,
        end_index=1
    )
    
    segment2 = PathSegment(
        segment_id="low_risk_segment",
        segment_type=SegmentType.ATOMIC,
        edges=[sample_lend_edge],
        start_index=1,
        end_index=2
    )
    
    segments = [segment1, segment2]
    
    # Analyze with segments
    path_analysis = mev_assessor.assess_path_risk(path, 100_000, segments)
    
    # Should have segment risk analysis
    assert len(path_analysis.segment_risks) == 2
    assert path_analysis.highest_risk_segment == "high_risk_segment"
    assert path_analysis.segment_risks["high_risk_segment"] > path_analysis.segment_risks["low_risk_segment"]


def test_execution_strategy_generation(mev_assessor, sample_trade_edge, sample_flash_loan_edge):
    """Test execution strategy generation based on risk."""
    # Low risk path
    low_risk_path = [sample_lend_edge]
    low_risk_analysis = mev_assessor.assess_path_risk(low_risk_path, 10_000)
    
    assert low_risk_analysis.execution_strategy["method"] in ["public", "public_protected"]
    assert not low_risk_analysis.execution_strategy["use_flashbots"]
    
    # High risk path
    high_risk_path = [sample_flash_loan_edge, sample_trade_edge, sample_trade_edge]
    high_risk_analysis = mev_assessor.assess_path_risk(high_risk_path, 500_000)
    
    assert high_risk_analysis.execution_strategy["method"] in ["flashbots", "private_relay"]
    assert high_risk_analysis.execution_strategy["atomic_required"]
    assert high_risk_analysis.execution_strategy["priority_fee_multiplier"] > 1.0


def test_mitigation_suggestions(mev_assessor, sample_trade_edge):
    """Test that appropriate mitigation suggestions are generated."""
    # High-risk large trade
    analysis = mev_assessor.assess_edge_risk(sample_trade_edge, 1_000_000)
    
    # Should have mitigation suggestions
    assert len(analysis.mitigation_suggestions) > 0
    
    # Check for relevant suggestions
    suggestions_text = " ".join(analysis.mitigation_suggestions)
    assert "private mempool" in suggestions_text.lower() or "flashbots" in suggestions_text.lower()


def test_convenience_functions():
    """Test convenience functions for quick risk assessment."""
    # Create test edge
    edge = YieldGraphEdge(
        edge_id="test_edge",
        source_asset_id="WETH",
        target_asset_id="USDC",
        edge_type=EdgeType.TRADE,
        protocol_name="uniswap_v3",
        chain_name="ethereum",
        execution_properties=EdgeExecutionProperties(mev_sensitivity=0.6)
    )
    
    # Test edge risk calculation
    risk_score = calculate_edge_mev_risk(edge, 50_000)
    assert 0 <= risk_score <= 1.0
    
    # Test path risk assessment
    path = [edge, edge]
    path_analysis = assess_path_mev_risk(path, 50_000)
    assert isinstance(path_analysis, PathMEVAnalysis)
    assert path_analysis.total_edges == 2


def test_risk_categorization_thresholds(mev_assessor):
    """Test risk level categorization."""
    test_scores = [
        (0.1, MEVRiskLevel.MINIMAL),
        (0.3, MEVRiskLevel.LOW),
        (0.5, MEVRiskLevel.MEDIUM),
        (0.7, MEVRiskLevel.HIGH),
        (0.9, MEVRiskLevel.CRITICAL)
    ]
    
    for score, expected_level in test_scores:
        level = mev_assessor._categorize_risk_level(score)
        assert level == expected_level


def test_mev_loss_estimation(mev_assessor, sample_trade_edge):
    """Test MEV loss estimation in basis points."""
    # Small trade should have relatively higher loss
    small_analysis = mev_assessor.assess_edge_risk(sample_trade_edge, 10_000)
    
    # Large trade should have relatively lower loss
    large_analysis = mev_assessor.assess_edge_risk(sample_trade_edge, 10_000_000)
    
    # Both should have loss estimates
    assert small_analysis.estimated_mev_loss_bps > 0
    assert large_analysis.estimated_mev_loss_bps > 0
    
    # Large trades typically have lower relative MEV loss
    assert large_analysis.estimated_mev_loss_bps < small_analysis.estimated_mev_loss_bps


if __name__ == "__main__":
    # Run basic test
    print("🧪 Testing MEV Risk Assessor")
    print("=" * 40)
    
    assessor = MEVRiskAssessor()
    
    # Create test edge
    test_edge = YieldGraphEdge(
        edge_id="test_trade",
        source_asset_id="WETH",
        target_asset_id="USDC",
        edge_type=EdgeType.TRADE,
        protocol_name="uniswap_v3",
        chain_name="ethereum",
        state=EdgeState(liquidity_usd=1_000_000),
        execution_properties=EdgeExecutionProperties(mev_sensitivity=0.7)
    )
    
    # Test edge analysis
    analysis = assessor.assess_edge_risk(test_edge, 100_000)
    print(f"✅ Edge MEV Analysis:")
    print(f"   - Risk Score: {analysis.final_risk_score:.2f}")
    print(f"   - Risk Level: {analysis.risk_level.value}")
    print(f"   - Execution: {analysis.recommended_execution}")
    print(f"   - Est. Loss: {analysis.estimated_mev_loss_bps:.1f} bps")
    
    # Test path analysis
    path = [test_edge, test_edge]
    path_analysis = assessor.assess_path_risk(path, 100_000)
    print(f"\n✅ Path MEV Analysis:")
    print(f"   - Compounded Risk: {path_analysis.compounded_risk:.2f}")
    print(f"   - Overall Level: {path_analysis.overall_risk_level.value}")
    print(f"   - Execution: {path_analysis.recommended_execution_method}")
    print(f"   - Atomic Required: {path_analysis.requires_atomic_execution}")
    
    print("\n✅ MEV Risk Assessor test passed!")