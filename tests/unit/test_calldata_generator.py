"""
Unit tests for the Dynamic Calldata Generator.

Tests the generation of calldata for various edge types and segment configurations,
ensuring proper integration with existing protocol adapters.
"""
import pytest
from unittest.mock import Mock, patch
from typing import List

from yield_arbitrage.execution.calldata_generator import (
    CalldataGenerator, EdgeOperationCalldata, SegmentCalldata
)
from yield_arbitrage.pathfinding.path_segment_analyzer import PathSegment, SegmentType
from yield_arbitrage.graph_engine.models import (
    YieldGraphEdge, EdgeType, EdgeExecutionProperties, EdgeConstraints, EdgeState
)


def create_test_edge(
    edge_id: str,
    source_asset: str,
    target_asset: str,
    edge_type: EdgeType = EdgeType.TRADE,
    protocol: str = "uniswap_v3",
    chain: str = "ethereum"
) -> YieldGraphEdge:
    """Create a test edge for calldata generation tests."""
    return YieldGraphEdge(
        edge_id=edge_id,
        source_asset_id=source_asset,
        target_asset_id=target_asset,
        edge_type=edge_type,
        protocol_name=protocol,
        chain_name=chain,
        execution_properties=EdgeExecutionProperties(),
        constraints=EdgeConstraints(),
        state=EdgeState()
    )


def create_test_segment(
    segment_id: str,
    edges: List[YieldGraphEdge],
    segment_type: SegmentType = SegmentType.ATOMIC
) -> PathSegment:
    """Create a test segment for calldata generation tests."""
    return PathSegment(
        segment_id=segment_id,
        segment_type=segment_type,
        edges=edges,
        start_index=0,
        end_index=len(edges) - 1
    )


class TestCalldataGenerator:
    """Test suite for CalldataGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create a CalldataGenerator instance."""
        return CalldataGenerator(chain_id=1)
    
    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.chain_id == 1
        assert isinstance(generator.protocol_adapters, dict)
        assert "uniswap_v3_router" in generator.protocol_contracts
    
    def test_single_trade_segment_calldata(self, generator):
        """Test calldata generation for a single trade segment."""
        edge = create_test_edge("swap1", "USDC", "WETH")
        segment = create_test_segment("seg_0", [edge])
        
        segment_calldata = generator.generate_segment_calldata(
            segment, 
            recipient="0x1234567890123456789012345678901234567890"
        )
        
        assert segment_calldata.segment_id == "seg_0"
        assert len(segment_calldata.operations) == 1
        assert not segment_calldata.requires_flash_loan
        
        operation = segment_calldata.operations[0]
        assert operation.edge_type == EdgeType.TRADE
        assert len(operation.call_data) > 0  # Should have generated calldata
        assert operation.input_amount == 0  # Dynamic
    
    def test_multi_operation_segment(self, generator):
        """Test calldata generation for segment with multiple operations."""
        edges = [
            create_test_edge("swap1", "USDC", "WETH"),
            create_test_edge("swap2", "WETH", "DAI")
        ]
        segment = create_test_segment("seg_0", edges)
        
        segment_calldata = generator.generate_segment_calldata(
            segment,
            recipient="0x1234567890123456789012345678901234567890"
        )
        
        assert len(segment_calldata.operations) == 2
        assert all(op.edge_type == EdgeType.TRADE for op in segment_calldata.operations)
    
    def test_flash_loan_segment_calldata(self, generator):
        """Test calldata generation for flash loan segment."""
        edges = [
            create_test_edge("swap1", "USDC", "WETH"),  # Skip flash loan edge for now
            create_test_edge("swap2", "WETH", "USDC")
        ]
        segment = create_test_segment("seg_0", edges, SegmentType.FLASH_LOAN_ATOMIC)
        segment.requires_flash_loan = True
        segment.flash_loan_asset = "USDC"
        segment.flash_loan_amount = 100000
        
        segment_calldata = generator.generate_segment_calldata(
            segment,
            recipient="0x1234567890123456789012345678901234567890"
        )
        
        assert segment_calldata.requires_flash_loan
        assert segment_calldata.flash_loan_asset == "USDC"
        assert segment_calldata.flash_loan_amount == 100000
        assert len(segment_calldata.operations) == 2
    
    def test_lending_operation_calldata(self, generator):
        """Test calldata generation for lending operations."""
        edge = create_test_edge("lend", "USDC", "aUSDC", EdgeType.LEND, protocol="aave_v3")
        segment = create_test_segment("seg_0", [edge])
        
        segment_calldata = generator.generate_segment_calldata(
            segment,
            recipient="0x1234567890123456789012345678901234567890"
        )
        
        operation = segment_calldata.operations[0]
        assert operation.edge_type == EdgeType.LEND
        assert operation.target_contract == generator.protocol_contracts["aave_v3_pool"]
        assert len(operation.call_data) > 0
    
    def test_borrowing_operation_calldata(self, generator):
        """Test calldata generation for borrowing operations."""
        edge = create_test_edge("borrow", "aUSDC", "USDC", EdgeType.BORROW, protocol="aave_v3")
        segment = create_test_segment("seg_0", [edge])
        
        segment_calldata = generator.generate_segment_calldata(
            segment,
            recipient="0x1234567890123456789012345678901234567890"
        )
        
        operation = segment_calldata.operations[0]
        assert operation.edge_type == EdgeType.BORROW
        assert operation.target_contract == generator.protocol_contracts["aave_v3_pool"]
        assert len(operation.call_data) > 0
    
    def test_unsupported_edge_type_error(self, generator):
        """Test error handling for unsupported edge types."""
        edge = create_test_edge("unsupported", "A", "B", EdgeType.WAIT)
        segment = create_test_segment("seg_0", [edge])
        
        with pytest.raises(ValueError, match="Unsupported edge type"):
            generator.generate_segment_calldata(
                segment,
                recipient="0x1234567890123456789012345678901234567890"
            )
    
    def test_unsupported_protocol_error(self, generator):
        """Test error handling for unsupported protocols."""
        edge = create_test_edge("trade", "A", "B", EdgeType.TRADE, protocol="unknown")
        segment = create_test_segment("seg_0", [edge])
        
        with pytest.raises(ValueError, match="Unsupported trade protocol"):
            generator.generate_segment_calldata(
                segment,
                recipient="0x1234567890123456789012345678901234567890"
            )
    
    def test_router_encoding(self, generator):
        """Test encoding segment calldata for router contract."""
        edge = create_test_edge("swap1", "USDC", "WETH")
        segment = create_test_segment("seg_0", [edge])
        
        segment_calldata = generator.generate_segment_calldata(
            segment,
            recipient="0x1234567890123456789012345678901234567890"
        )
        
        encoded = generator.encode_segment_for_router(segment_calldata)
        
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0
    
    def test_min_output_calculation(self, generator):
        """Test minimum output amount calculation with slippage."""
        edge = create_test_edge("swap1", "USDC", "WETH")
        segment = create_test_segment("seg_0", [edge])
        
        with patch.object(generator, "_get_expected_output") as mock_output:
            mock_output.return_value = 1000
            
            segment_calldata = generator.generate_segment_calldata(
                segment,
                recipient="0x1234567890123456789012345678901234567890"
            )
            
            # Calculate min outputs with 0.5% slippage
            updated_calldata = generator.calculate_min_output_amounts(
                segment_calldata, slippage_tolerance=0.005
            )
        
        operation = updated_calldata.operations[0]
        assert operation.min_output_amount == 995  # 1000 * (1 - 0.005)
    
    def test_calldata_validation_success(self, generator):
        """Test successful calldata validation."""
        segment_calldata = SegmentCalldata(
            segment_id="seg_0",
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
        
        assert generator.validate_segment_calldata(segment_calldata)
    
    def test_calldata_validation_failures(self, generator):
        """Test calldata validation failure cases."""
        # Missing segment ID
        invalid_calldata = SegmentCalldata(
            segment_id="",
            operations=[],
            requires_flash_loan=False
        )
        assert not generator.validate_segment_calldata(invalid_calldata)
        
        # No operations
        invalid_calldata = SegmentCalldata(
            segment_id="seg_0",
            operations=[],
            requires_flash_loan=False
        )
        assert not generator.validate_segment_calldata(invalid_calldata)
        
        # Invalid token address
        invalid_calldata = SegmentCalldata(
            segment_id="seg_0",
            operations=[
                EdgeOperationCalldata(
                    edge_type=EdgeType.TRADE,
                    target_contract="0x1234567890123456789012345678901234567890",
                    input_token="invalid_address",
                    output_token="0x1234567890123456789012345678901234567890",
                    input_amount=0,
                    min_output_amount=0,
                    call_data=b"\x12\x34",
                    metadata={}
                )
            ],
            requires_flash_loan=False
        )
        assert not generator.validate_segment_calldata(invalid_calldata)
    
    def test_flash_loan_validation(self, generator):
        """Test flash loan validation requirements."""
        # Flash loan required but missing parameters
        invalid_calldata = SegmentCalldata(
            segment_id="seg_0",
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
        assert not generator.validate_segment_calldata(invalid_calldata)
    
    def test_edge_type_mapping(self, generator):
        """Test EdgeType to integer mapping."""
        assert generator._edge_type_to_int(EdgeType.TRADE) == 0
        assert generator._edge_type_to_int(EdgeType.LEND) == 4
        assert generator._edge_type_to_int(EdgeType.BORROW) == 5
        assert generator._edge_type_to_int(EdgeType.FLASH_LOAN) == 9
    
    def test_token_address_resolution(self, generator):
        """Test token ID to address resolution."""
        assert generator._get_token_address("USDC") == "0xA0b86a33E6441c2b73AC95F2DB8FFf6d4daF1E65"
        assert generator._get_token_address("WETH") == "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        assert generator._get_token_address("UNKNOWN") == "UNKNOWN"  # Pass through
    
    def test_atoken_address_mapping(self, generator):
        """Test aToken address mapping for Aave."""
        usdc_address = "0xA0b86a33E6441c2b73AC95F2DB8FFf6d4daF1E65"
        ausdc_address = generator._get_atoken_address(usdc_address)
        assert ausdc_address == "0x98C23E9d8f34FEFb1B7BD6a91B7FF122F4e16F5c"
    
    def test_address_validation(self, generator):
        """Test Ethereum address validation."""
        assert generator._is_valid_address("0x1234567890123456789012345678901234567890")
        assert not generator._is_valid_address("0x12345")  # Too short
        assert not generator._is_valid_address("1234567890123456789012345678901234567890")  # No 0x
        assert not generator._is_valid_address("")  # Empty
        assert not generator._is_valid_address("0xGGGG567890123456789012345678901234567890")  # Invalid hex
    
    def test_statistics(self, generator):
        """Test generator statistics."""
        stats = generator.get_statistics()
        
        assert "supported_protocols" in stats
        assert isinstance(stats["supported_protocols"], list)
        assert stats["chain_id"] == 1
        assert stats["protocol_contracts"] > 0
    
    def test_different_chain_initialization(self):
        """Test generator initialization for different chains."""
        polygon_generator = CalldataGenerator(chain_id=137)
        assert polygon_generator.chain_id == 137
        # Different chains might have different or no protocol contracts
        assert isinstance(polygon_generator.protocol_contracts, dict)


@pytest.mark.asyncio
async def test_calldata_generator_integration():
    """Integration test with realistic segment data."""
    generator = CalldataGenerator(chain_id=1)
    
    # Create a realistic arbitrage segment
    edges = [
        create_test_edge("uni_swap_1", "USDC", "WETH", EdgeType.TRADE, "uniswap_v3"),
        create_test_edge("uni_swap_2", "WETH", "DAI", EdgeType.TRADE, "uniswap_v3"),
        create_test_edge("uni_swap_3", "DAI", "USDC", EdgeType.TRADE, "uniswap_v3")
    ]
    
    segment = create_test_segment("arbitrage_0", edges, SegmentType.ATOMIC)
    
    # Generate calldata
    segment_calldata = generator.generate_segment_calldata(
        segment,
        recipient="0x742d35Cc6634C0532925a3b8D94A3C0E8D2D46D5",
        deadline=1234567890
    )
    
    # Validate
    assert generator.validate_segment_calldata(segment_calldata)
    
    # Encode for router
    encoded = generator.encode_segment_for_router(segment_calldata)
    assert len(encoded) > 0
    
    # Test statistics
    stats = generator.get_statistics()
    assert len(stats["supported_protocols"]) >= 0