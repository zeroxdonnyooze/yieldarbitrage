#!/usr/bin/env python3
"""
Demonstration script for Flash Loan Graph Engine Integration.

This script shows how flash loan opportunities are discovered and integrated
into the yield arbitrage graph engine, enabling capital-efficient strategies.
"""
import sys
import os
from decimal import Decimal

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.protocols.flash_loan_provider import FlashLoanDiscovery, FlashLoanEdgeGenerator
from yield_arbitrage.graph_engine.flash_loan_integrator import FlashLoanGraphIntegrator
from yield_arbitrage.graph_engine.models import (
    YieldGraphEdge, EdgeType, EdgeExecutionProperties, EdgeConstraints, EdgeState
)


def create_sample_edge(edge_id: str, source: str, target: str, min_amount: float = 1000.0) -> YieldGraphEdge:
    """Create a sample trading edge for demonstration."""
    return YieldGraphEdge(
        edge_id=edge_id,
        source_asset_id=source,
        target_asset_id=target,
        edge_type=EdgeType.TRADE,
        protocol_name="uniswap_v3",
        chain_name="ethereum",
        execution_properties=EdgeExecutionProperties(
            gas_estimate=120_000,
            mev_sensitivity=0.3
        ),
        constraints=EdgeConstraints(
            min_input_amount=min_amount,
            max_input_amount=1_000_000.0
        ),
        state=EdgeState(
            confidence_score=0.8,
            liquidity_available=500_000.0
        )
    )


def demo_flash_loan_discovery():
    """Demonstrate flash loan opportunity discovery."""
    print("🔍 Flash Loan Opportunity Discovery:")
    print("   Scanning Aave V3, Balancer, and Uniswap V3...")
    
    discovery = FlashLoanDiscovery(chain_name="ethereum")
    
    # Discover opportunities for major assets
    opportunities = discovery.discover_flash_loan_opportunities(
        target_assets={"USDC", "WETH", "DAI"},
        min_liquidity=Decimal('1000')
    )
    
    print(f"\n📊 Found {len(opportunities)} flash loan opportunities:")
    
    for opp in opportunities:
        print(f"\n   💰 {opp.asset_id}:")
        print(f"      Total Liquidity: ${opp.total_max_liquidity:,.0f}")
        print(f"      Providers: {len(opp.available_providers)}")
        print(f"      Best Provider: {opp.best_terms.provider.value}")
        print(f"      Best Fee Rate: {opp.best_terms.fee_rate:.4f}% ({opp.best_terms.fee_rate * 100:.4f}%)")
        print(f"      Max Amount: ${opp.best_terms.max_amount:,.0f}")
        
        # Show cost analysis
        test_amount = Decimal('50000')
        cost = discovery.calculate_flash_loan_cost(opp.best_terms, test_amount)
        print(f"      Cost for $50K: ${cost:.2f}")
    
    # Test profitability analysis
    print(f"\n💡 Profitability Analysis:")
    if opportunities:
        opp = opportunities[0]
        amount = Decimal('100000')
        
        # Test different profit scenarios
        scenarios = [
            ("High Profit Scenario", Decimal('2000')),   # $2K profit
            ("Medium Profit Scenario", Decimal('500')),  # $500 profit
            ("Low Profit Scenario", Decimal('50'))       # $50 profit
        ]
        
        for scenario_name, expected_profit in scenarios:
            is_profitable = discovery.is_flash_loan_profitable(
                opp.best_terms, amount, expected_profit
            )
            cost = discovery.calculate_flash_loan_cost(opp.best_terms, amount)
            net_profit = expected_profit - cost
            
            print(f"      {scenario_name}: {'✅ Profitable' if is_profitable else '❌ Not Profitable'}")
            print(f"         Expected Profit: ${expected_profit:.2f}")
            print(f"         Flash Loan Cost: ${cost:.2f}")
            print(f"         Net Profit: ${net_profit:.2f}")
    
    print()
    return discovery


def demo_edge_generation():
    """Demonstrate flash loan edge generation."""
    print("🔗 Flash Loan Edge Generation:")
    print("   Converting opportunities to graph edges...")
    
    discovery = FlashLoanDiscovery(chain_name="ethereum")
    generator = FlashLoanEdgeGenerator(discovery)
    
    # Get opportunities
    opportunities = discovery.discover_flash_loan_opportunities(
        target_assets={"USDC", "WETH"}
    )
    
    # Generate edges
    edge_data = generator.generate_flash_loan_edges(opportunities)
    
    print(f"\n📈 Generated {len(edge_data)} edges:")
    
    for edge in edge_data:
        edge_type = edge["edge_type"]
        source = edge["source_asset_id"]
        target = edge["target_asset_id"]
        protocol = edge["protocol_name"]
        
        print(f"\n   🔗 {edge_type} Edge:")
        print(f"      {source} → {target}")
        print(f"      Protocol: {protocol}")
        print(f"      Gas Estimate: {edge['execution_properties']['gas_estimate']:,}")
        print(f"      Max Amount: ${edge['constraints']['max_input_amount']:,.0f}")
        print(f"      Fee Rate: {edge['constraints']['fee_rate']:.4f}%")
    
    print()
    return edge_data


def demo_graph_integration():
    """Demonstrate full graph engine integration."""
    print("🏗️ Graph Engine Integration:")
    print("   Integrating flash loans into yield arbitrage graph...")
    
    integrator = FlashLoanGraphIntegrator(chain_name="ethereum")
    
    # Create some existing trading edges
    existing_edges = [
        create_sample_edge("uni_usdc_weth", "USDC", "WETH", 1000.0),
        create_sample_edge("uni_weth_dai", "WETH", "DAI", 5000.0),
        create_sample_edge("high_capital_arb", "USDC", "DAI", 50000.0),  # High capital trade
    ]
    
    # Set high confidence for the high-capital trade
    existing_edges[2].state.confidence_score = 0.9
    
    print(f"\n   📊 Existing Edges: {len(existing_edges)}")
    for edge in existing_edges:
        print(f"      {edge.edge_id}: {edge.source_asset_id} → {edge.target_asset_id}")
        print(f"         Min Amount: ${edge.constraints.min_input_amount:,.0f}")
        print(f"         Confidence: {edge.state.confidence_score:.2f}")
    
    # Integrate flash loans
    new_edges = integrator.discover_and_integrate_flash_loans(
        target_assets={"USDC", "WETH", "DAI"},
        existing_edges=existing_edges
    )
    
    print(f"\n   ⚡ New Flash Loan Edges: {len(new_edges)}")
    
    # Categorize edges
    flash_edges = [e for e in new_edges if e.edge_type == EdgeType.FLASH_LOAN]
    repay_edges = [e for e in new_edges if e.edge_type == EdgeType.FLASH_REPAY]
    enhanced_edges = [e for e in new_edges if e.metadata.get("flash_loan_enhanced")]
    
    print(f"      Flash Loan Edges: {len(flash_edges)}")
    print(f"      Repayment Edges: {len(repay_edges)}")
    print(f"      Enhanced Edges: {len(enhanced_edges)}")
    
    # Show virtual assets
    print(f"\n   💫 Virtual Assets Created: {len(integrator.virtual_assets)}")
    for asset in integrator.virtual_assets:
        capacity = integrator.get_flash_loan_capacity(asset[6:])  # Remove "FLASH_" prefix
        print(f"      {asset}: ${capacity:,.0f} capacity")
    
    # Show enhanced edges (synergies)
    if enhanced_edges:
        print(f"\n   🔥 Flash Loan Synergies Found:")
        for edge in enhanced_edges:
            original_id = edge.metadata.get("original_edge_id")
            print(f"      Enhanced: {edge.source_asset_id} → {edge.target_asset_id}")
            print(f"         Original: {original_id}")
            print(f"         Max Amount: ${edge.constraints.max_input_amount:,.0f}")
    
    print()
    return integrator, new_edges


def demo_path_validation():
    """Demonstrate flash loan path validation."""
    print("✅ Flash Loan Path Validation:")
    print("   Testing various path configurations...")
    
    integrator = FlashLoanGraphIntegrator(chain_name="ethereum")
    
    # Create flash loan edges
    flash_edges = integrator.discover_and_integrate_flash_loans(
        target_assets={"USDC"}
    )
    
    flash_edge = next(e for e in flash_edges if e.edge_type == EdgeType.FLASH_LOAN)
    repay_edge = next(e for e in flash_edges if e.edge_type == EdgeType.FLASH_REPAY)
    
    # Create intermediate trading edge
    trade_edge = create_sample_edge("arb_trade", "FLASH_USDC", "WETH")
    
    # Test valid path
    valid_path = [flash_edge, trade_edge, repay_edge]
    is_valid = integrator.validate_flash_loan_path(valid_path)
    
    print(f"\n   ✅ Valid Flash Loan Path:")
    print(f"      {flash_edge.source_asset_id} → {flash_edge.target_asset_id} (Flash Loan)")
    print(f"      {trade_edge.source_asset_id} → {trade_edge.target_asset_id} (Trade)")
    print(f"      {repay_edge.source_asset_id} → {repay_edge.target_asset_id} (Repayment)")
    print(f"      Validation Result: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Test invalid path (missing repayment)
    invalid_path = [flash_edge, trade_edge]
    is_invalid = integrator.validate_flash_loan_path(invalid_path)
    
    print(f"\n   ❌ Invalid Path (No Repayment):")
    print(f"      {flash_edge.source_asset_id} → {flash_edge.target_asset_id} (Flash Loan)")
    print(f"      {trade_edge.source_asset_id} → {trade_edge.target_asset_id} (Trade)")
    print(f"      Validation Result: {'✅ Valid' if is_invalid else '❌ Invalid'}")
    
    # Test repayment calculation
    borrowed_amount = Decimal('100000')
    repayment_amount = integrator.get_required_repayment_amount("FLASH_USDC", borrowed_amount)
    fee = repayment_amount - borrowed_amount
    
    print(f"\n   💰 Repayment Calculation:")
    print(f"      Borrowed Amount: ${borrowed_amount:,.2f}")
    print(f"      Repayment Amount: ${repayment_amount:,.2f}")
    print(f"      Flash Loan Fee: ${fee:.2f}")
    print(f"      Fee Rate: {(fee / borrowed_amount) * 100:.4f}%")
    
    print()


def demo_statistics_and_monitoring():
    """Demonstrate statistics and monitoring capabilities."""
    print("📊 Statistics and Monitoring:")
    print("   Tracking flash loan integration metrics...")
    
    discovery = FlashLoanDiscovery(chain_name="ethereum")
    integrator = FlashLoanGraphIntegrator(chain_name="ethereum")
    
    # Run integration
    integrator.discover_and_integrate_flash_loans(
        target_assets={"USDC", "WETH", "DAI", "USDT"}
    )
    
    # Get discovery statistics
    discovery_stats = discovery.get_statistics()
    print(f"\n   🔍 Discovery Statistics:")
    print(f"      Opportunities Found: {discovery_stats['opportunities_discovered']}")
    print(f"      Providers Checked: {discovery_stats['providers_checked']}")
    print(f"      Total Liquidity: ${discovery_stats['total_liquidity_discovered']:,.0f}")
    print(f"      Supported Assets: {discovery_stats['supported_assets']}")
    print(f"      Cached Opportunities: {discovery_stats['cached_opportunities']}")
    
    # Get integration statistics
    integration_stats = integrator.get_statistics()
    print(f"\n   🏗️ Integration Statistics:")
    print(f"      Flash Loan Edges Created: {integration_stats['flash_loan_edges_created']}")
    print(f"      Virtual Assets Created: {integration_stats['virtual_assets_created']}")
    print(f"      Opportunities Integrated: {integration_stats['opportunities_integrated']}")
    print(f"      Total Flash Loan Capacity: ${integration_stats['total_flash_loan_capacity']:,.0f}")
    print(f"      Active Flash Loan Edges: {integration_stats['flash_loan_edges_active']}")
    
    # Show supported assets
    supported_assets = discovery.get_supported_assets()
    print(f"\n   💱 Supported Assets ({len(supported_assets)}):")
    for asset in sorted(supported_assets):
        capacity = integrator.get_flash_loan_capacity(asset)
        if capacity:
            print(f"      {asset}: ${capacity:,.0f}")
    
    print()


def demo_real_world_scenario():
    """Demonstrate a real-world arbitrage scenario with flash loans."""
    print("🌍 Real-World Arbitrage Scenario:")
    print("   USDC arbitrage between Uniswap and Curve with flash loan...")
    
    integrator = FlashLoanGraphIntegrator(chain_name="ethereum")
    
    # Create arbitrage edges
    arbitrage_edges = [
        # Flash loan USDC
        # Trade USDC → WETH on Uniswap (assuming price difference)
        create_sample_edge("uni_usdc_weth", "FLASH_USDC", "WETH", 100000.0),
        # Trade WETH → USDC on Curve (assuming better rate)
        create_sample_edge("curve_weth_usdc", "WETH", "USDC", 0.0),
        # Repay flash loan
    ]
    
    # Set realistic parameters
    arbitrage_edges[0].state.conversion_rate = 0.0003  # 3000 USDC per ETH
    arbitrage_edges[1].state.conversion_rate = 3010.0  # 3010 USDC per ETH (profit opportunity)
    arbitrage_edges[1].state.confidence_score = 0.85
    
    # Get flash loan edges
    flash_loan_edges = integrator.discover_and_integrate_flash_loans(
        target_assets={"USDC"}
    )
    
    # Find the flash loan and repayment edges
    flash_edge = next(e for e in flash_loan_edges if e.edge_type == EdgeType.FLASH_LOAN)
    repay_edge = next(e for e in flash_loan_edges if e.edge_type == EdgeType.FLASH_REPAY)
    
    # Create complete arbitrage path
    complete_path = [flash_edge] + arbitrage_edges + [repay_edge]
    
    print(f"\n   💡 Arbitrage Strategy:")
    for i, edge in enumerate(complete_path, 1):
        if edge.edge_type == EdgeType.FLASH_LOAN:
            print(f"      {i}. Flash loan ${edge.constraints.max_input_amount:,.0f} {edge.source_asset_id}")
            print(f"         Fee: {edge.metadata.get('fee_rate', 0):.4f}%")
        elif edge.edge_type == EdgeType.FLASH_REPAY:
            borrowed = Decimal('100000')
            repayment = integrator.get_required_repayment_amount("FLASH_USDC", borrowed)
            fee = repayment - borrowed
            print(f"      {i}. Repay flash loan ${repayment:,.2f}")
            print(f"         Fee paid: ${fee:.2f}")
        else:
            rate = edge.state.conversion_rate
            print(f"      {i}. Trade {edge.source_asset_id} → {edge.target_asset_id}")
            print(f"         Rate: {rate}")
    
    # Validate the path
    is_valid = integrator.validate_flash_loan_path(complete_path)
    print(f"\n   ✅ Path Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Calculate rough profit estimate
    flash_loan_amount = Decimal('100000')
    flash_loan_fee = flash_loan_amount * Decimal('0.0009')  # 0.09% Aave fee
    
    # Simulate trade: 100K USDC → ETH → USDC with 0.33% profit
    profit_rate = Decimal('0.0033')  # 0.33% profit opportunity
    gross_profit = flash_loan_amount * profit_rate
    net_profit = gross_profit - flash_loan_fee
    
    print(f"\n   💰 Profit Analysis:")
    print(f"      Flash Loan Amount: ${flash_loan_amount:,.0f}")
    print(f"      Expected Gross Profit: ${gross_profit:.2f} ({profit_rate * 100:.2f}%)")
    print(f"      Flash Loan Fee: ${flash_loan_fee:.2f}")
    print(f"      Net Profit: ${net_profit:.2f}")
    print(f"      ROI: {(net_profit / flash_loan_fee) * 100:.1f}% vs flash loan cost")
    
    if net_profit > 0:
        print(f"      Status: ✅ Profitable strategy")
    else:
        print(f"      Status: ❌ Unprofitable strategy")
    
    print()


def main():
    """Run all flash loan integration demonstrations."""
    print("🎯 Flash Loan Graph Engine Integration Demonstrations")
    print("=" * 60)
    print()
    
    try:
        demo_flash_loan_discovery()
        demo_edge_generation()
        demo_graph_integration()
        demo_path_validation()
        demo_statistics_and_monitoring()
        demo_real_world_scenario()
        
        print("✅ All flash loan integration demonstrations completed!")
        print()
        print("💡 Key Benefits:")
        print("   • Capital-efficient arbitrage strategies")
        print("   • Automatic discovery of flash loan opportunities")
        print("   • Graph-based pathfinding with flash loan integration")
        print("   • Comprehensive validation and risk management")
        print("   • Real-time liquidity monitoring")
        print("   • Multi-provider optimization")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()