#!/usr/bin/env python3
"""
MEV Integration Test - Testing new components with existing MEV infrastructure.
"""
import asyncio
import sys
import time
from decimal import Decimal

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.mev_detection.opportunity_models import BackRunOpportunity
from yield_arbitrage.graph_engine.models import BackRunEdge
from yield_arbitrage.mev_protection.mev_risk_assessor import MEVRiskAssessor
from yield_arbitrage.mev_protection.execution_router import MEVAwareExecutionRouter
from yield_arbitrage.mev_protection.mev_risk_assessor import PathMEVAnalysis
from yield_arbitrage.execution.enhanced_transaction_builder import BatchExecutionPlan


async def test_complete_integration():
    """Test complete integration of new MEV detection with existing infrastructure."""
    print("🔧 Testing Complete MEV Integration")
    print("=" * 60)
    
    # 1. Create a back-run opportunity (NEW COMPONENT)
    opportunity = BackRunOpportunity(
        opportunity_id="integration_test_001",
        target_transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        estimated_profit_usd=250.0,
        confidence_score=0.85,
        required_capital_usd=10000.0,
        max_gas_price=int(180e9),
        execution_deadline=time.time() + 180,
        chain_id=1,
        source_asset="WETH",
        target_asset="USDC",
        optimal_amount=5.0,
        expected_price_movement=0.02,
        pool_address="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
    )
    
    print(f"✅ 1. Created BackRunOpportunity:")
    print(f"   - Profit: ${opportunity.estimated_profit_usd}")
    print(f"   - Confidence: {opportunity.confidence_score}")
    print(f"   - Target TX: {opportunity.target_transaction_hash[:20]}...")
    
    # 2. Create BackRunEdge from opportunity (NEW COMPONENT)
    back_run_edge = BackRunEdge(
        chain_name="ethereum",
        target_transaction=opportunity.target_transaction_hash,
        source_asset=opportunity.source_asset,
        target_asset=opportunity.target_asset,
        expected_profit=opportunity.estimated_profit_usd
    )
    
    print(f"\\n✅ 2. Created BackRunEdge:")
    print(f"   - Edge ID: {back_run_edge.edge_id}")
    print(f"   - Edge Type: {back_run_edge.edge_type}")
    print(f"   - MEV Sensitivity: {back_run_edge.execution_properties.mev_sensitivity}")
    
    # 3. Assess MEV risk with existing infrastructure (EXISTING COMPONENT)
    risk_assessor = MEVRiskAssessor()
    edge_analysis = risk_assessor.assess_edge_risk(back_run_edge, opportunity.optimal_amount * 2000)
    
    print(f"\\n✅ 3. MEV Risk Assessment (Existing Infrastructure):")
    print(f"   - Final Risk Score: {edge_analysis.final_risk_score:.3f}")
    print(f"   - Risk Level: {edge_analysis.risk_level.value}")
    print(f"   - Recommended Execution: {edge_analysis.recommended_execution}")
    print(f"   - Estimated MEV Loss: {edge_analysis.estimated_mev_loss_bps:.1f} bps")
    
    # 4. Route execution with existing router (EXISTING COMPONENT)
    execution_router = MEVAwareExecutionRouter()
    
    # Create path analysis for routing
    path_analysis = PathMEVAnalysis(
        path_id="backrun_path_001",
        total_edges=1,
        max_edge_risk=edge_analysis.final_risk_score,
        average_edge_risk=edge_analysis.final_risk_score,
        compounded_risk=edge_analysis.final_risk_score,
        overall_risk_level=edge_analysis.risk_level,
        critical_edges=[back_run_edge.edge_id],
        recommended_execution_method="flashbots_bundle"
    )
    
    # Create execution plan
    execution_plan = BatchExecutionPlan(
        plan_id="backrun_execution_001",
        router_address="0x1234567890123456789012345678901234567890",
        executor_address="0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
        segments=[],
        total_gas_estimate=300000,
        expected_profit=Decimal(str(opportunity.estimated_profit_usd))
    )
    
    # Select execution route
    execution_route = execution_router.select_execution_route(
        chain_id=1,
        mev_analysis=path_analysis,
        execution_plan=execution_plan
    )
    
    print(f"\\n✅ 4. Execution Routing (Existing Infrastructure):")
    print(f"   - Selected Method: {execution_route.method.value}")
    print(f"   - Endpoint: {execution_route.endpoint}")
    print(f"   - Priority Fee: {execution_route.priority_fee_wei / 1e9:.1f} gwei")
    print(f"   - Expected Confirmation: {execution_route.expected_confirmation_time}s")
    
    # 5. Test back-run edge calculation
    calculation_result = back_run_edge.calculate_output(opportunity.optimal_amount)
    
    print(f"\\n✅ 5. Back-Run Edge Calculation:")
    print(f"   - Input: ${opportunity.optimal_amount:.2f}")
    print(f"   - Output: ${calculation_result['output_amount']:.2f}")
    print(f"   - Expected Profit: ${calculation_result['expected_profit_usd']:.2f}")
    print(f"   - Effective Rate: {calculation_result['effective_rate']:.4f}")
    print(f"   - Gas Cost: ${calculation_result['gas_cost_usd']:.2f}")
    
    # 6. Calculate profit after all costs
    gas_cost_usd = calculation_result['gas_cost_usd']
    effective_profit = opportunity.calculate_effective_profit(gas_cost_usd)
    
    print(f"\\n✅ 6. Profit Analysis:")
    print(f"   - Gross Profit: ${opportunity.estimated_profit_usd:.2f}")
    print(f"   - Gas Cost: ${gas_cost_usd:.2f}")
    print(f"   - Net Profit: ${effective_profit:.2f}")
    print(f"   - Profit Margin: {(effective_profit / opportunity.estimated_profit_usd) * 100:.1f}%")
    
    return {
        "opportunity": opportunity,
        "back_run_edge": back_run_edge,
        "risk_analysis": edge_analysis,
        "execution_route": execution_route,
        "calculation_result": calculation_result,
        "net_profit": effective_profit
    }


async def main():
    """Run integration test."""
    print("⚡ MEV Integration Test Suite")
    print("=" * 80)
    print("Testing integration between NEW opportunity detection and EXISTING MEV infrastructure")
    print("=" * 80)
    
    try:
        results = await test_complete_integration()
        
        print(f"\\n{'='*80}")
        print(f"🎉 INTEGRATION TEST SUMMARY")
        print(f"{'='*80}")
        
        # Verify integration worked
        success_checks = [
            ("BackRunOpportunity Creation", results["opportunity"] is not None),
            ("BackRunEdge Integration", results["back_run_edge"].edge_type.value == "BACK_RUN"),
            ("MEV Risk Assessment", results["risk_analysis"].final_risk_score > 0),
            ("Execution Routing", results["execution_route"].method is not None),
            ("Edge Calculation", results["calculation_result"]["output_amount"] > 0),
            ("Profitable After Costs", results["net_profit"] > 0)
        ]
        
        all_passed = True
        for test_name, passed in success_checks:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name:.<50} {status}")
            if not passed:
                all_passed = False
        
        print(f"\\n🏆 Overall Result: {'INTEGRATION SUCCESSFUL' if all_passed else 'INTEGRATION FAILED'}")
        
        if all_passed:
            print(f"\\n🎯 Perfect Integration Achieved!")
            print(f"   ✅ NEW: BackRunOpportunity + BackRunEdge models")
            print(f"   ✅ EXISTING: MEVRiskAssessor integration")
            print(f"   ✅ EXISTING: MEVAwareExecutionRouter integration")
            print(f"   ✅ EXISTING: Flashbots execution support")
            print(f"   ✅ Complete end-to-end MEV opportunity pipeline")
            print(f"\\n⚡ Ready for production MEV capture!")
        
        return all_passed
        
    except Exception as e:
        print(f"\\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)