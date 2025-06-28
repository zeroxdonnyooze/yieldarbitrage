#!/usr/bin/env python3
"""
Demonstration script for the Path Segment Analyzer.

This script shows how the analyzer can break down complex DeFi paths into
executable segments based on edge execution properties.
"""
import sys
import os

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.pathfinding.path_segment_analyzer import PathSegmentAnalyzer
from yield_arbitrage.graph_engine.models import (
    YieldGraphEdge, EdgeType, EdgeExecutionProperties, EdgeConstraints, EdgeState
)


def create_edge(edge_id: str, source: str, target: str, edge_type: EdgeType = EdgeType.TRADE,
                chain: str = "ethereum", **props) -> YieldGraphEdge:
    """Helper to create test edges."""
    return YieldGraphEdge(
        edge_id=edge_id,
        source_asset_id=source,
        target_asset_id=target,
        edge_type=edge_type,
        protocol_name="test_protocol",
        chain_name=chain,
        execution_properties=EdgeExecutionProperties(**props),
        constraints=EdgeConstraints(),
        state=EdgeState()
    )


def demo_simple_arbitrage():
    """Demonstrate simple arbitrage segmentation."""
    print("🔄 Simple Arbitrage Path:")
    print("   USDC → WETH → DAI → USDC")
    
    edges = [
        create_edge("swap1", "USDC", "WETH", gas_estimate=120_000),
        create_edge("swap2", "WETH", "DAI", gas_estimate=130_000),
        create_edge("swap3", "DAI", "USDC", gas_estimate=125_000)
    ]
    
    analyzer = PathSegmentAnalyzer()
    segments = analyzer.analyze_path(edges)
    
    print(f"\n📊 Analysis Result: {len(segments)} segment(s)")
    for i, segment in enumerate(segments):
        summary = analyzer.get_segment_summary(segment)
        print(f"   Segment {i}: {summary['type']} - {summary['edge_count']} edges")
        print(f"      {summary['input_asset']} → {summary['output_asset']}")
        print(f"      Gas: {summary['gas_estimate']:,}, MEV: {summary['mev_sensitivity']:.2f}")
    print()


def demo_cross_chain_path():
    """Demonstrate cross-chain path segmentation."""
    print("🌉 Cross-Chain Arbitrage Path:")
    print("   ETH_USDC → ETH_WETH → ARB_WETH → ARB_USDC")
    
    edges = [
        create_edge("eth_swap", "ETH_USDC", "ETH_WETH", chain="ethereum"),
        create_edge("bridge", "ETH_WETH", "ARB_WETH", 
                   edge_type=EdgeType.BRIDGE, chain="arbitrum", requires_bridge=True),
        create_edge("arb_swap", "ARB_WETH", "ARB_USDC", chain="arbitrum")
    ]
    
    analyzer = PathSegmentAnalyzer()
    segments = analyzer.analyze_path(edges)
    
    print(f"\n📊 Analysis Result: {len(segments)} segment(s)")
    for i, segment in enumerate(segments):
        summary = analyzer.get_segment_summary(segment)
        print(f"   Segment {i}: {summary['type']} - {summary['edge_count']} edges")
        print(f"      {summary['input_asset']} → {summary['output_asset']}")
        print(f"      Chain: {summary['source_chain']} → {summary['target_chain']}")
    print()


def demo_yield_farming_path():
    """Demonstrate yield farming path with time delays."""
    print("🌾 Yield Farming Path:")
    print("   USDC → WETH → stETH (stake) → wstETH → USDC")
    
    edges = [
        create_edge("swap1", "USDC", "WETH"),
        create_edge("swap2", "WETH", "stETH"),
        create_edge("stake", "stETH", "wstETH", 
                   edge_type=EdgeType.STAKE, requires_time_delay=3600),  # 1 hour
        create_edge("swap3", "wstETH", "USDC")
    ]
    
    analyzer = PathSegmentAnalyzer()
    segments = analyzer.analyze_path(edges)
    
    print(f"\n📊 Analysis Result: {len(segments)} segment(s)")
    for i, segment in enumerate(segments):
        summary = analyzer.get_segment_summary(segment)
        print(f"   Segment {i}: {summary['type']} - {summary['edge_count']} edges")
        print(f"      {summary['input_asset']} → {summary['output_asset']}")
        if summary['delay_seconds']:
            print(f"      Delay: {summary['delay_seconds']}s")
    print()


def demo_flash_loan_arbitrage():
    """Demonstrate flash loan arbitrage segmentation."""
    print("⚡ Flash Loan Arbitrage Path:")
    print("   Flash USDC → WETH → rare_token → USDC → Repay")
    
    edges = [
        create_edge("flash", "USDC", "USDC_LOAN", 
                   edge_type=EdgeType.FLASH_LOAN, min_liquidity_required=500_000),
        create_edge("swap1", "USDC_LOAN", "WETH", min_liquidity_required=500_000),
        create_edge("swap2", "WETH", "RARE_TOKEN", mev_sensitivity=0.8),
        create_edge("swap3", "RARE_TOKEN", "USDC", mev_sensitivity=0.9)
    ]
    
    analyzer = PathSegmentAnalyzer()
    segments = analyzer.analyze_path(edges)
    
    print(f"\n📊 Analysis Result: {len(segments)} segment(s)")
    for i, segment in enumerate(segments):
        summary = analyzer.get_segment_summary(segment)
        print(f"   Segment {i}: {summary['type']} - {summary['edge_count']} edges")
        print(f"      {summary['input_asset']} → {summary['output_asset']}")
        if summary['requires_flash_loan']:
            print(f"      Flash Loan: ${summary['flash_loan_amount']:,}")
    print()


def demo_complex_defi_strategy():
    """Demonstrate complex DeFi strategy with multiple segment types."""
    print("🏗️ Complex DeFi Strategy:")
    print("   Multi-protocol with lending, bridging, and yield farming")
    
    edges = [
        # Initial trades
        create_edge("uni_swap", "USDC", "WETH", gas_estimate=120_000),
        create_edge("curve_swap", "WETH", "stETH", gas_estimate=180_000),
        
        # Lending (requires capital holding)
        create_edge("aave_lend", "stETH", "astETH", 
                   edge_type=EdgeType.LEND, requires_capital_holding=True),
        
        # Bridge to Arbitrum
        create_edge("bridge_op", "astETH", "ARB_astETH", 
                   edge_type=EdgeType.BRIDGE, chain="arbitrum", requires_bridge=True),
        
        # High gas operations
        create_edge("arb_swap1", "ARB_astETH", "ARB_USDC", 
                   chain="arbitrum", gas_estimate=80_000),
        create_edge("arb_swap2", "ARB_USDC", "ARB_WETH", 
                   chain="arbitrum", gas_estimate=85_000),
        
        # High MEV sensitivity
        create_edge("mev_trade", "ARB_WETH", "ARB_RARE", 
                   chain="arbitrum", mev_sensitivity=0.9)
    ]
    
    analyzer = PathSegmentAnalyzer(max_segment_gas=200_000, max_mev_sensitivity=0.6)
    segments = analyzer.analyze_path(edges)
    
    print(f"\n📊 Analysis Result: {len(segments)} segment(s)")
    for i, segment in enumerate(segments):
        summary = analyzer.get_segment_summary(segment)
        print(f"   Segment {i}: {summary['type']} - {summary['edge_count']} edges")
        print(f"      {summary['input_asset']} → {summary['output_asset']}")
        print(f"      Gas: {summary['gas_estimate']:,}, MEV: {summary['mev_sensitivity']:.2f}")
        if summary['source_chain'] != summary['target_chain']:
            print(f"      Chain: {summary['source_chain']} → {summary['target_chain']}")
    
    # Show connectivity validation
    is_connected = analyzer.validate_segment_connectivity(segments)
    print(f"\n🔗 Segment Connectivity: {'✅ Valid' if is_connected else '❌ Invalid'}")
    
    # Show statistics
    stats = analyzer.get_statistics()
    print(f"\n📈 Analyzer Statistics:")
    print(f"   Paths analyzed: {stats['paths_analyzed']}")
    print(f"   Segments created: {stats['segments_created']}")
    print(f"   Atomic segments: {stats['atomic_segments']}")
    print(f"   Time-delayed segments: {stats['delayed_segments']}")
    print(f"   Bridged segments: {stats['bridged_segments']}")
    print(f"   Flash loan segments: {stats['flash_loan_segments']}")
    print()


def main():
    """Run all demonstrations."""
    print("🎯 Path Segment Analyzer Demonstrations")
    print("=" * 50)
    print()
    
    demo_simple_arbitrage()
    demo_cross_chain_path()
    demo_yield_farming_path()
    demo_flash_loan_arbitrage()
    demo_complex_defi_strategy()
    
    print("✅ All demonstrations completed!")


if __name__ == "__main__":
    main()