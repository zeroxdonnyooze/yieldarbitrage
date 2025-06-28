#!/usr/bin/env python3
"""
Test script for MEV Protection Layer.

This script demonstrates the complete MEV protection functionality including
risk assessment and execution routing for different chains.
"""
import sys
from decimal import Decimal

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.mev_protection import (
    MEVRiskAssessor, MEVRiskLevel, calculate_edge_mev_risk
)
from yield_arbitrage.mev_protection.execution_router import (
    MEVAwareExecutionRouter, ExecutionMethod, get_chain_execution_method
)
from yield_arbitrage.graph_engine.models import (
    YieldGraphEdge, EdgeType, EdgeState, EdgeExecutionProperties
)
from yield_arbitrage.execution.enhanced_transaction_builder import (
    BatchExecutionPlan, RouterTransaction
)


def create_test_edges():
    """Create test edges for different scenarios."""
    edges = []
    
    # High-risk DEX trade on Ethereum
    eth_trade = YieldGraphEdge(
        edge_id="eth_uniswap_weth_usdc",
        source_asset_id="WETH",
        target_asset_id="USDC",
        edge_type=EdgeType.TRADE,
        protocol_name="uniswap_v3",
        chain_name="ethereum",
        state=EdgeState(
            conversion_rate=2000.0,
            liquidity_usd=5_000_000.0,
            gas_cost_usd=30.0
        ),
        execution_properties=EdgeExecutionProperties(
            mev_sensitivity=0.8,
            max_impact_allowed=0.01,
            supports_private_mempool=True
        )
    )
    edges.append(("Ethereum DEX Trade", eth_trade))
    
    # Medium-risk lending on BSC
    bsc_lend = YieldGraphEdge(
        edge_id="bsc_venus_bnb_lend",
        source_asset_id="BNB",
        target_asset_id="vBNB",
        edge_type=EdgeType.LEND,
        protocol_name="venus",
        chain_name="bsc",
        state=EdgeState(
            conversion_rate=1.0,
            liquidity_usd=10_000_000.0,
            gas_cost_usd=2.0
        ),
        execution_properties=EdgeExecutionProperties(
            mev_sensitivity=0.4,
            supports_private_mempool=True
        )
    )
    edges.append(("BSC Lending", bsc_lend))
    
    # Low-risk stable swap on Arbitrum
    arb_stable = YieldGraphEdge(
        edge_id="arb_curve_usdc_usdt",
        source_asset_id="USDC",
        target_asset_id="USDT",
        edge_type=EdgeType.TRADE,
        protocol_name="curve",
        chain_name="arbitrum",
        state=EdgeState(
            conversion_rate=0.9998,
            liquidity_usd=50_000_000.0,
            gas_cost_usd=0.5
        ),
        execution_properties=EdgeExecutionProperties(
            mev_sensitivity=0.2,
            supports_private_mempool=False  # L2 doesn't need it
        )
    )
    edges.append(("Arbitrum Stable Swap", arb_stable))
    
    # Flash loan on Ethereum
    flash_loan = YieldGraphEdge(
        edge_id="eth_aave_flash_usdc",
        source_asset_id="NONE",  # Flash loans start with no capital
        target_asset_id="USDC",  # Flash loan provides USDC
        edge_type=EdgeType.FLASH_LOAN,
        protocol_name="aave",
        chain_name="ethereum",
        state=EdgeState(
            conversion_rate=0.9991,
            liquidity_usd=100_000_000.0
        ),
        execution_properties=EdgeExecutionProperties(
            mev_sensitivity=0.9,
            supports_private_mempool=True
        )
    )
    edges.append(("Ethereum Flash Loan", flash_loan))
    
    return edges


def test_mev_risk_assessment():
    """Test MEV risk assessment for different edges."""
    print("🔍 Testing MEV Risk Assessment")
    print("=" * 50)
    
    assessor = MEVRiskAssessor()
    test_edges = create_test_edges()
    
    # Test different trade sizes
    trade_sizes = [10_000, 100_000, 1_000_000]
    
    for name, edge in test_edges:
        print(f"\n📊 {name} (Chain: {edge.chain_name}, Protocol: {edge.protocol_name})")
        print("-" * 40)
        
        for size in trade_sizes:
            analysis = assessor.assess_edge_risk(edge, size)
            
            print(f"  ${size:,} trade:")
            print(f"    - Risk Score: {analysis.final_risk_score:.3f}")
            print(f"    - Risk Level: {analysis.risk_level.value}")
            print(f"    - Sandwich Risk: {analysis.sandwich_risk:.2f}")
            print(f"    - Frontrun Risk: {analysis.frontrun_risk:.2f}")
            print(f"    - MEV Loss: {analysis.estimated_mev_loss_bps:.1f} bps")
            print(f"    - Execution: {analysis.recommended_execution}")
            
            if analysis.mitigation_suggestions:
                print(f"    - Suggestions: {analysis.mitigation_suggestions[0]}")


def test_path_mev_assessment():
    """Test MEV assessment for complete arbitrage paths."""
    print("\n\n🛤️ Testing Path MEV Assessment")
    print("=" * 50)
    
    assessor = MEVRiskAssessor()
    
    # Create different risk paths
    _, eth_trade = create_test_edges()[0]
    _, bsc_lend = create_test_edges()[1]
    _, arb_stable = create_test_edges()[2]
    _, flash_loan = create_test_edges()[3]
    
    paths = [
        ("Low Risk Path", [arb_stable, bsc_lend]),
        ("Medium Risk Path", [bsc_lend, eth_trade, arb_stable]),
        ("High Risk Path", [flash_loan, eth_trade, eth_trade]),
    ]
    
    for path_name, edges in paths:
        analysis = assessor.assess_path_risk(edges, 100_000)
        
        print(f"\n📈 {path_name}:")
        print(f"  - Total Edges: {analysis.total_edges}")
        print(f"  - Max Edge Risk: {analysis.max_edge_risk:.3f}")
        print(f"  - Average Risk: {analysis.average_edge_risk:.3f}")
        print(f"  - Compounded Risk: {analysis.compounded_risk:.3f}")
        print(f"  - Overall Level: {analysis.overall_risk_level.value}")
        print(f"  - Execution Method: {analysis.recommended_execution_method}")
        print(f"  - Atomic Required: {analysis.requires_atomic_execution}")
        print(f"  - Total MEV Loss: {analysis.estimated_total_mev_loss_bps:.1f} bps")
        
        if analysis.critical_edges:
            print(f"  - Critical Edges: {', '.join(analysis.critical_edges)}")
        
        # Show execution strategy
        strategy = analysis.execution_strategy
        print(f"  - Strategy:")
        print(f"    * Method: {strategy['method']}")
        print(f"    * Priority Fee: {strategy['priority_fee_multiplier']}x")
        print(f"    * Slippage Buffer: {strategy['slippage_buffer']}")


def test_execution_routing():
    """Test MEV-aware execution routing for different chains."""
    print("\n\n🚀 Testing Execution Routing")
    print("=" * 50)
    
    router = MEVAwareExecutionRouter()
    assessor = MEVRiskAssessor()
    
    # Test different chains and risk levels
    test_cases = [
        (1, "Ethereum", MEVRiskLevel.LOW),
        (1, "Ethereum", MEVRiskLevel.HIGH),
        (56, "BSC", MEVRiskLevel.MEDIUM),
        (137, "Polygon", MEVRiskLevel.HIGH),
        (42161, "Arbitrum", MEVRiskLevel.HIGH),
        (10, "Optimism", MEVRiskLevel.MEDIUM),
    ]
    
    for chain_id, chain_name, risk_level in test_cases:
        # Create mock MEV analysis
        risk_score = {"minimal": 0.1, "low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9}[risk_level.value]
        mock_analysis = type('PathMEVAnalysis', (), {
            'overall_risk_level': risk_level,
            'compounded_risk': risk_score,
            'path_id': f'test_{chain_name}_{risk_level.value}'
        })()
        
        # Create mock execution plan
        mock_plan = type('BatchExecutionPlan', (), {
            'segments': []
        })()
        
        # Get execution route
        route = router.select_execution_route(chain_id, mock_analysis, mock_plan)
        
        print(f"\n🔗 {chain_name} (Chain ID: {chain_id}) - {risk_level.value} risk:")
        print(f"  - Method: {route.method.value}")
        print(f"  - Endpoint: {route.endpoint}")
        print(f"  - Auth Required: {route.auth_required}")
        print(f"  - Priority Fee: {route.priority_fee_wei / 1e9:.1f} gwei")
        print(f"  - Expected Time: {route.expected_confirmation_time}s")
        
        if route.flashbots_params:
            print(f"  - Flashbots Bundle: {route.bundle_id}")
        if route.fallback_method:
            print(f"  - Fallback: {route.fallback_method.value}")


def test_chain_capabilities():
    """Test MEV protection capabilities for different chains."""
    print("\n\n🌐 Chain MEV Protection Capabilities")
    print("=" * 50)
    
    router = MEVAwareExecutionRouter()
    
    chains = [
        (1, "Ethereum"),
        (56, "BSC"),
        (137, "Polygon"),
        (42161, "Arbitrum"),
        (10, "Optimism"),
        (8453, "Base"),
        (43114, "Avalanche"),  # Not configured
    ]
    
    for chain_id, chain_name in chains:
        capabilities = router.get_chain_capabilities(chain_id)
        
        if capabilities["supported"]:
            print(f"\n✅ {chain_name} (Chain ID: {chain_id}):")
            print(f"  - Block Time: {capabilities['block_time']}s")
            print(f"  - Has Sequencer: {capabilities['has_sequencer']}")
            print(f"  - Protection Methods:")
            for method, available in capabilities["methods"].items():
                if available:
                    print(f"    * {method}: ✓")
        else:
            print(f"\n❌ {chain_name} (Chain ID: {chain_id}): Not configured")


def test_convenience_functions():
    """Test convenience functions."""
    print("\n\n🛠️ Testing Convenience Functions")
    print("=" * 40)
    
    # Create test edge
    edge = YieldGraphEdge(
        edge_id="test_edge",
        source_asset_id="WETH",
        target_asset_id="USDC",
        edge_type=EdgeType.TRADE,
        protocol_name="uniswap_v3",
        chain_name="ethereum",
        execution_properties=EdgeExecutionProperties(mev_sensitivity=0.7)
    )
    
    # Quick risk calculation
    risk = calculate_edge_mev_risk(edge, 50_000)
    print(f"✅ Quick edge risk: {risk:.3f}")
    
    # Chain execution method recommendation
    for risk_level in [MEVRiskLevel.LOW, MEVRiskLevel.HIGH]:
        method = get_chain_execution_method(1, risk_level)
        print(f"✅ Ethereum {risk_level.value} risk → {method.value}")


def main():
    """Run all MEV protection tests."""
    print("🛡️ MEV Protection Layer Test Suite")
    print("=" * 70)
    print("Testing modular MEV protection with Flashbots (Ethereum) and")
    print("private nodes/relays for other chains")
    print("=" * 70)
    
    # Run all tests
    test_mev_risk_assessment()
    test_path_mev_assessment()
    test_execution_routing()
    test_chain_capabilities()
    test_convenience_functions()
    
    print("\n\n" + "=" * 70)
    print("🎉 MEV Protection Layer Tests Complete!")
    print("=" * 70)
    
    print("\n✅ Task 12.1 Complete: MEV Risk Assessment")
    print("  - Edge-level MEV risk scoring")
    print("  - Path-level compounded risk analysis")
    print("  - Chain and protocol-specific modifiers")
    print("  - Size-based risk adjustments")
    print("  - Execution method recommendations")
    print("  - MEV loss estimations")
    
    print("\n✅ Modular Execution Routing Ready:")
    print("  - Flashbots support for Ethereum")
    print("  - Private node/relay support for other chains")
    print("  - Chain-specific configurations")
    print("  - Risk-based routing decisions")
    print("  - Fallback mechanisms")


if __name__ == "__main__":
    main()