#!/usr/bin/env python3
"""
Demonstration script for the Dynamic Calldata Generator.

This script shows how the generator creates calldata for executing path segments
through the YieldArbitrageRouter smart contract.
"""
import sys
import os

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.execution.calldata_generator import CalldataGenerator
from yield_arbitrage.pathfinding.path_segment_analyzer import PathSegment, SegmentType
from yield_arbitrage.graph_engine.models import (
    YieldGraphEdge, EdgeType, EdgeExecutionProperties, EdgeConstraints, EdgeState
)


def create_edge(edge_id: str, source: str, target: str, edge_type: EdgeType = EdgeType.TRADE,
                protocol: str = "uniswap_v3", **props) -> YieldGraphEdge:
    """Helper to create test edges."""
    return YieldGraphEdge(
        edge_id=edge_id,
        source_asset_id=source,
        target_asset_id=target,
        edge_type=edge_type,
        protocol_name=protocol,
        chain_name="ethereum",
        execution_properties=EdgeExecutionProperties(**props),
        constraints=EdgeConstraints(),
        state=EdgeState()
    )


def create_segment(segment_id: str, edges: list, segment_type: SegmentType = SegmentType.ATOMIC) -> PathSegment:
    """Helper to create test segments."""
    segment = PathSegment(
        segment_id=segment_id,
        segment_type=segment_type,
        edges=edges,
        start_index=0,
        end_index=len(edges) - 1
    )
    return segment


def demo_simple_trade_calldata():
    """Demonstrate calldata generation for a simple trade."""
    print("🔄 Simple Trade Calldata Generation:")
    print("   USDC → WETH via Uniswap V3")
    
    generator = CalldataGenerator(chain_id=1)
    
    edge = create_edge("uniswap_swap", "USDC", "WETH")
    segment = create_segment("trade_seg", [edge])
    
    try:
        # Note: This will fail because we don't have the actual UniswapV3Adapter
        # but it demonstrates the flow
        segment_calldata = generator.generate_segment_calldata(
            segment,
            recipient="0x742d35Cc6634C0532925a3b8D94A3C0E8D2D46D5"
        )
        
        print(f"✅ Generated calldata for segment: {segment_calldata.segment_id}")
        print(f"   Operations: {len(segment_calldata.operations)}")
        print(f"   Flash loan required: {segment_calldata.requires_flash_loan}")
        
        # Validate the calldata
        is_valid = generator.validate_segment_calldata(segment_calldata)
        print(f"   Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
    except Exception as e:
        print(f"⚠️  Demo limitation: {e}")
        print("   (This is expected - requires actual protocol adapter integration)")
    
    print()


def demo_multi_hop_arbitrage():
    """Demonstrate calldata for multi-hop arbitrage."""
    print("🔄 Multi-Hop Arbitrage Calldata:")
    print("   USDC → WETH → DAI → USDC")
    
    generator = CalldataGenerator(chain_id=1)
    
    edges = [
        create_edge("swap_1", "USDC", "WETH"),
        create_edge("swap_2", "WETH", "DAI"),
        create_edge("swap_3", "DAI", "USDC")
    ]
    
    segment = create_segment("arbitrage_seg", edges)
    
    try:
        segment_calldata = generator.generate_segment_calldata(
            segment,
            recipient="0x742d35Cc6634C0532925a3b8D94A3C0E8D2D46D5",
            deadline=1234567890
        )
        
        print(f"✅ Generated calldata for {len(edges)}-operation segment")
        print(f"   Segment ID: {segment_calldata.segment_id}")
        print(f"   Recipient: {segment_calldata.recipient}")
        print(f"   Deadline: {segment_calldata.deadline}")
        
        for i, op in enumerate(segment_calldata.operations):
            print(f"   Operation {i}: {op.input_token} → {op.output_token}")
            print(f"      Target: {op.target_contract}")
            print(f"      Calldata length: {len(op.call_data)} bytes")
        
    except Exception as e:
        print(f"⚠️  Demo limitation: {e}")
    
    print()


def demo_lending_calldata():
    """Demonstrate calldata for lending operations."""
    print("💰 Lending Operation Calldata:")
    print("   USDC → aUSDC via Aave V3")
    
    generator = CalldataGenerator(chain_id=1)
    
    edge = create_edge("aave_supply", "USDC", "aUSDC", EdgeType.LEND, protocol="aave_v3")
    segment = create_segment("lending_seg", [edge])
    
    try:
        segment_calldata = generator.generate_segment_calldata(
            segment,
            recipient="0x742d35Cc6634C0532925a3b8D94A3C0E8D2D46D5"
        )
        
        operation = segment_calldata.operations[0]
        print(f"✅ Generated lending calldata")
        print(f"   Protocol: {operation.metadata.get('protocol', 'N/A')}")
        print(f"   Target contract: {operation.target_contract}")
        print(f"   Input token: {operation.input_token}")
        print(f"   Output token: {operation.output_token}")
        print(f"   Calldata: {operation.call_data.hex()[:20]}...")
        
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    print()


def demo_flash_loan_segment():
    """Demonstrate calldata for flash loan segment."""
    print("⚡ Flash Loan Segment Calldata:")
    print("   Flash USDC → arbitrage → repay")
    
    generator = CalldataGenerator(chain_id=1)
    
    edges = [
        create_edge("flash", "USDC", "USDC_LOAN", EdgeType.FLASH_LOAN),
        create_edge("arb_1", "USDC_LOAN", "WETH"),
        create_edge("arb_2", "WETH", "USDC")
    ]
    
    segment = create_segment("flash_seg", edges, SegmentType.FLASH_LOAN_ATOMIC)
    segment.requires_flash_loan = True
    segment.flash_loan_asset = "USDC"
    segment.flash_loan_amount = 100000
    
    try:
        segment_calldata = generator.generate_segment_calldata(
            segment,
            recipient="0x742d35Cc6634C0532925a3b8D94A3C0E8D2D46D5"
        )
        
        print(f"✅ Generated flash loan segment calldata")
        print(f"   Flash loan asset: {segment_calldata.flash_loan_asset}")
        print(f"   Flash loan amount: ${segment_calldata.flash_loan_amount:,}")
        print(f"   Operations count: {len(segment_calldata.operations)}")
        print(f"   Requires flash loan: {segment_calldata.requires_flash_loan}")
        
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    print()


def demo_router_encoding():
    """Demonstrate encoding calldata for the router contract."""
    print("📦 Router Contract Encoding:")
    print("   Converting segment calldata to router format")
    
    generator = CalldataGenerator(chain_id=1)
    
    # Create a simple mock segment calldata
    from yield_arbitrage.execution.calldata_generator import SegmentCalldata, EdgeOperationCalldata
    
    segment_calldata = SegmentCalldata(
        segment_id="demo_seg",
        operations=[
            EdgeOperationCalldata(
                edge_type=EdgeType.TRADE,
                target_contract="0xE592427A0AEce92De3Edee1F18E0157C05861564",  # Uniswap V3 Router
                input_token="0xA0b86a33E6441c2b73AC95F2DB8FFf6d4daF1E65",  # USDC
                output_token="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                input_amount=0,  # Dynamic
                min_output_amount=0,
                call_data=bytes.fromhex("414bf389000000000000000000000000a0b86a33e6441c2b73ac95f2db8fff6d4daf1e65"),
                metadata={"protocol": "uniswap_v3"}
            )
        ],
        requires_flash_loan=False,
        recipient="0x742d35Cc6634C0532925a3b8D94A3C0E8D2D46D5"
    )
    
    try:
        # Validate first
        is_valid = generator.validate_segment_calldata(segment_calldata)
        print(f"   Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
        if is_valid:
            # Encode for router
            encoded = generator.encode_segment_for_router(segment_calldata)
            print(f"   Encoded length: {len(encoded)} bytes")
            print(f"   Encoded data: {encoded.hex()[:40]}...")
        
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    print()


def demo_validation_tests():
    """Demonstrate calldata validation."""
    print("✅ Calldata Validation Tests:")
    
    generator = CalldataGenerator(chain_id=1)
    
    from yield_arbitrage.execution.calldata_generator import SegmentCalldata, EdgeOperationCalldata
    
    # Test 1: Valid calldata
    valid_calldata = SegmentCalldata(
        segment_id="valid_seg",
        operations=[
            EdgeOperationCalldata(
                edge_type=EdgeType.TRADE,
                target_contract="0x1234567890123456789012345678901234567890",
                input_token="0x1234567890123456789012345678901234567890",
                output_token="0x1234567890123456789012345678901234567890",
                input_amount=0,
                min_output_amount=0,
                call_data=b"\x12\x34",
                metadata={}
            )
        ],
        requires_flash_loan=False,
        recipient="0x1234567890123456789012345678901234567890"
    )
    
    result = generator.validate_segment_calldata(valid_calldata)
    print(f"   Valid calldata test: {'✅ Passed' if result else '❌ Failed'}")
    
    # Test 2: Invalid address
    invalid_calldata = SegmentCalldata(
        segment_id="invalid_seg",
        operations=[
            EdgeOperationCalldata(
                edge_type=EdgeType.TRADE,
                target_contract="invalid_address",
                input_token="0x1234567890123456789012345678901234567890",
                output_token="0x1234567890123456789012345678901234567890",
                input_amount=0,
                min_output_amount=0,
                call_data=b"\x12\x34",
                metadata={}
            )
        ],
        requires_flash_loan=False,
        recipient="0x1234567890123456789012345678901234567890"
    )
    
    result = generator.validate_segment_calldata(invalid_calldata)
    print(f"   Invalid address test: {'✅ Passed' if not result else '❌ Failed'}")
    
    # Test 3: Missing flash loan parameters
    flash_loan_calldata = SegmentCalldata(
        segment_id="flash_seg",
        operations=[
            EdgeOperationCalldata(
                edge_type=EdgeType.TRADE,
                target_contract="0x1234567890123456789012345678901234567890",
                input_token="0x1234567890123456789012345678901234567890",
                output_token="0x1234567890123456789012345678901234567890",
                input_amount=0,
                min_output_amount=0,
                call_data=b"\x12\x34",
                metadata={}
            )
        ],
        requires_flash_loan=True,  # Missing flash loan parameters
        recipient="0x1234567890123456789012345678901234567890"
    )
    
    result = generator.validate_segment_calldata(flash_loan_calldata)
    print(f"   Missing flash loan params test: {'✅ Passed' if not result else '❌ Failed'}")
    
    print()


def demo_statistics():
    """Demonstrate generator statistics."""
    print("📊 Generator Statistics:")
    
    generator = CalldataGenerator(chain_id=1)
    stats = generator.get_statistics()
    
    print(f"   Supported protocols: {', '.join(stats['supported_protocols'])}")
    print(f"   Chain ID: {stats['chain_id']}")
    print(f"   Protocol contracts: {stats['protocol_contracts']}")
    
    # Test different chain
    polygon_generator = CalldataGenerator(chain_id=137)
    polygon_stats = polygon_generator.get_statistics()
    print(f"   Polygon contracts: {polygon_stats['protocol_contracts']}")
    
    print()


def main():
    """Run all demonstrations."""
    print("🎯 Dynamic Calldata Generator Demonstrations")
    print("=" * 50)
    print()
    
    demo_simple_trade_calldata()
    demo_multi_hop_arbitrage()
    demo_lending_calldata()
    demo_flash_loan_segment()
    demo_router_encoding()
    demo_validation_tests()
    demo_statistics()
    
    print("✅ All demonstrations completed!")
    print()
    print("Note: Some demonstrations show expected limitations due to")
    print("missing protocol adapter implementations. This is normal for")
    print("the current development stage.")


if __name__ == "__main__":
    main()