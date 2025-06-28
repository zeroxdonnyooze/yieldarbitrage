"""Tests for edge validation and call graph extraction."""
import pytest
from unittest.mock import Mock, AsyncMock
import time

from yield_arbitrage.execution.hybrid_simulator import (
    HybridPathSimulator,
    SimulationMode,
    SimulatorConfig,
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType, EdgeState


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for testing."""
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    
    mock_oracle = Mock()
    mock_oracle.get_price_usd = AsyncMock(return_value=2000.0)
    
    return mock_redis, mock_oracle


@pytest.fixture
def simulator(mock_dependencies):
    """Create simulator for testing."""
    mock_redis, mock_oracle = mock_dependencies
    
    config = SimulatorConfig(
        confidence_threshold=0.7,
        min_liquidity_threshold=10000.0
    )
    
    return HybridPathSimulator(
        redis_client=mock_redis,
        asset_oracle=mock_oracle,
        config=config
    )


@pytest.mark.asyncio
class TestEdgeValidation:
    """Test edge validation functionality."""
    
    async def test_validate_valid_path(self, simulator):
        """Test validation of a valid path."""
        # Create valid path
        path = [
            YieldGraphEdge(
                edge_id="eth_usdc",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v2",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="usdc_dai",
                source_asset_id="ETH_MAINNET_USDC",
                target_asset_id="ETH_MAINNET_DAI",
                edge_type=EdgeType.TRADE,
                protocol_name="curve",
                chain_name="ethereum"
            )
        ]
        
        # Mock good edge states
        good_state = EdgeState(
            conversion_rate=2500.0,
            liquidity_usd=100000.0,
            gas_cost_usd=5.0,
            confidence_score=0.9,
            last_updated_timestamp=time.time()
        )
        
        simulator._get_edge_state = AsyncMock(return_value=good_state)
        
        result = await simulator._validate_path_edges(path)
        
        assert result["valid"] is True
        assert len(result["issues"]) == 0
        assert len(result["edge_states"]) == 2
        assert "eth_usdc" in result["edge_states"]
        assert "usdc_dai" in result["edge_states"]
    
    async def test_validate_disconnected_path(self, simulator):
        """Test validation of disconnected path."""
        # Create disconnected path
        path = [
            YieldGraphEdge(
                edge_id="eth_usdc",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v2",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="wbtc_dai", 
                source_asset_id="ETH_MAINNET_WBTC",  # Disconnected!
                target_asset_id="ETH_MAINNET_DAI",
                edge_type=EdgeType.TRADE,
                protocol_name="curve",
                chain_name="ethereum"
            )
        ]
        
        simulator._get_edge_state = AsyncMock(return_value=None)
        
        result = await simulator._validate_path_edges(path)
        
        assert result["valid"] is False
        assert any("Path disconnect" in issue for issue in result["issues"])
    
    async def test_validate_missing_edge_state(self, simulator):
        """Test validation with missing edge state."""
        path = [
            YieldGraphEdge(
                edge_id="eth_usdc",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v2",
                chain_name="ethereum"
            )
        ]
        
        simulator._get_edge_state = AsyncMock(return_value=None)
        
        result = await simulator._validate_path_edges(path)
        
        assert result["valid"] is True  # Missing state is warning, not error
        assert len(result["missing_states"]) == 1
        assert "eth_usdc" in result["missing_states"]
        assert any("No state data available" in warning for warning in result["warnings"])
    
    async def test_validate_stale_edge_state(self, simulator):
        """Test validation with stale edge state."""
        path = [
            YieldGraphEdge(
                edge_id="eth_usdc",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v2",
                chain_name="ethereum"
            )
        ]
        
        # Create stale state (old timestamp)
        stale_state = EdgeState(
            conversion_rate=2500.0,
            liquidity_usd=100000.0,
            gas_cost_usd=5.0,
            confidence_score=0.9,
            last_updated_timestamp=time.time() - 600  # 10 minutes ago
        )
        
        simulator._get_edge_state = AsyncMock(return_value=stale_state)
        
        result = await simulator._validate_path_edges(path)
        
        assert result["valid"] is True  # Stale is warning, not error
        assert len(result["stale_states"]) == 1
        assert "eth_usdc" in result["stale_states"]
        assert any("State data is stale" in warning for warning in result["warnings"])
    
    async def test_validate_low_confidence_state(self, simulator):
        """Test validation with low confidence state."""
        path = [
            YieldGraphEdge(
                edge_id="eth_usdc",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v2",
                chain_name="ethereum"
            )
        ]
        
        # Create low confidence state
        low_conf_state = EdgeState(
            conversion_rate=2500.0,
            liquidity_usd=100000.0,
            gas_cost_usd=5.0,
            confidence_score=0.3,  # Below 0.7 threshold
            last_updated_timestamp=time.time()
        )
        
        simulator._get_edge_state = AsyncMock(return_value=low_conf_state)
        
        result = await simulator._validate_path_edges(path)
        
        assert result["valid"] is True  # Low confidence is warning
        assert len(result["low_confidence"]) == 1
        assert "eth_usdc" in result["low_confidence"]
        assert any("Low confidence" in warning for warning in result["warnings"])
    
    async def test_validate_invalid_conversion_rate(self, simulator):
        """Test validation with invalid conversion rate."""
        path = [
            YieldGraphEdge(
                edge_id="eth_usdc",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v2",
                chain_name="ethereum"
            )
        ]
        
        # Create state with invalid conversion rate
        invalid_state = EdgeState(
            conversion_rate=0.0,  # Invalid!
            liquidity_usd=100000.0,
            gas_cost_usd=5.0,
            confidence_score=0.9,
            last_updated_timestamp=time.time()
        )
        
        simulator._get_edge_state = AsyncMock(return_value=invalid_state)
        
        result = await simulator._validate_path_edges(path)
        
        assert result["valid"] is False
        assert any("Invalid conversion rate" in issue for issue in result["issues"])


@pytest.mark.asyncio
class TestProtocolValidation:
    """Test protocol-specific validation."""
    
    async def test_validate_uniswap_edge(self, simulator):
        """Test Uniswap-specific validation."""
        uniswap_edge = YieldGraphEdge(
            edge_id="uniswap_trade",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        )
        
        result = simulator._validate_protocol_specific(uniswap_edge, 0)
        
        assert len(result["issues"]) == 0  # Valid Uniswap edge
        assert len(result["warnings"]) == 0
    
    async def test_validate_invalid_uniswap_edge_type(self, simulator):
        """Test invalid edge type for Uniswap."""
        invalid_uniswap_edge = YieldGraphEdge(
            edge_id="uniswap_lend",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.LEND,  # Wrong type for Uniswap
            protocol_name="uniswap_v2",
            chain_name="ethereum"
        )
        
        result = simulator._validate_protocol_specific(invalid_uniswap_edge, 0)
        
        assert len(result["issues"]) == 1
        assert "should be TRADE type" in result["issues"][0]
    
    async def test_validate_aave_edge(self, simulator):
        """Test Aave-specific validation."""
        aave_edge = YieldGraphEdge(
            edge_id="aave_lend",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_AWETH",
            edge_type=EdgeType.LEND,
            protocol_name="aave_v2",
            chain_name="ethereum"
        )
        
        result = simulator._validate_protocol_specific(aave_edge, 0)
        
        assert len(result["issues"]) == 0  # Valid Aave edge
        assert len(result["warnings"]) == 0
    
    async def test_validate_flash_loan_warning(self, simulator):
        """Test flash loan validation warnings."""
        flash_loan_edge = YieldGraphEdge(
            edge_id="aave_flash",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",  # Different target for flash loan
            edge_type=EdgeType.FLASH_LOAN,
            protocol_name="aave_v2",
            chain_name="ethereum"
        )
        
        result = simulator._validate_protocol_specific(flash_loan_edge, 0)
        
        assert len(result["warnings"]) >= 1
        assert any("Flash loan requires careful gas management" in warning for warning in result["warnings"])
    
    async def test_validate_bridge_warning(self, simulator):
        """Test bridge validation warnings."""
        bridge_edge = YieldGraphEdge(
            edge_id="eth_polygon_bridge",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="POLYGON_WETH",
            edge_type=EdgeType.BRIDGE,
            protocol_name="polygon_bridge",
            chain_name="ethereum"
        )
        
        result = simulator._validate_protocol_specific(bridge_edge, 0)
        
        assert len(result["warnings"]) >= 1
        assert any("Bridge operations have higher risk" in warning for warning in result["warnings"])


@pytest.mark.asyncio
class TestCallGraphExtraction:
    """Test call graph extraction from Tenderly traces."""
    
    def test_extract_call_graph_empty_trace(self, simulator):
        """Test call graph extraction with empty trace."""
        call_graph = simulator._extract_call_graph_from_trace({})
        
        assert call_graph["total_calls"] == 0
        assert len(call_graph["unique_contracts"]) == 0
        assert len(call_graph["call_hierarchy"]) == 0
    
    def test_extract_call_graph_basic_trace(self, simulator):
        """Test call graph extraction with basic trace."""
        mock_trace = {
            "transaction": {
                "trace": {
                    "to": "0x1234567890123456789012345678901234567890",
                    "gasUsed": "21000",
                    "type": "CALL",
                    "input": "0xa9059cbb",
                    "calls": [
                        {
                            "to": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                            "gasUsed": "5000",
                            "type": "CALL",
                            "input": "0x70a08231"
                        }
                    ]
                },
                "logs": [
                    {
                        "address": "0x1234567890123456789012345678901234567890",
                        "topics": [
                            "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
                        ],
                        "data": "0x"
                    }
                ],
                "gas_used": 26000
            }
        }
        
        call_graph = simulator._extract_call_graph_from_trace(mock_trace)
        
        assert call_graph["total_calls"] >= 1  # At least main call
        assert len(call_graph["unique_contracts"]) == 2
        assert call_graph["total_gas_used"] == 26000
        assert len(call_graph["events_emitted"]) == 1
        assert call_graph["events_emitted"][0]["name"] == "Transfer"
    
    def test_decode_event_names(self, simulator):
        """Test event name decoding."""
        # Transfer event
        transfer_topics = ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"]
        assert simulator._decode_event_name(transfer_topics) == "Transfer"
        
        # Approval event
        approval_topics = ["0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925"]
        assert simulator._decode_event_name(approval_topics) == "Approval"
        
        # Unknown event
        unknown_topics = ["0x1234567890123456789012345678901234567890123456789012345678901234"]
        result = simulator._decode_event_name(unknown_topics)
        assert result.startswith("unknown_")
        
        # Empty topics
        assert simulator._decode_event_name([]) == "unknown"
    
    def test_detect_protocol_interactions(self, simulator):
        """Test protocol detection from contract addresses."""
        call_graph = {
            "unique_contracts": [
                "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",  # Uniswap V2
                "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9",  # Aave
                "0x1234567890123456789012345678901234567890"   # Unknown
            ],
            "protocol_interactions": {}
        }
        
        simulator._detect_protocol_interactions(call_graph)
        
        assert "uniswap_v2" in call_graph["protocol_interactions"]
        assert "aave" in call_graph["protocol_interactions"]
        assert len(call_graph["protocol_interactions"]) == 2  # Only known protocols


@pytest.mark.asyncio
class TestIntegratedValidation:
    """Test integrated validation in simulation flow."""
    
    async def test_simulation_with_validation_success(self, simulator):
        """Test simulation with successful validation."""
        path = [
            YieldGraphEdge(
                edge_id="eth_usdc",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v2",
                chain_name="ethereum"
            )
        ]
        
        # Mock good edge state
        good_state = EdgeState(
            conversion_rate=2500.0,
            liquidity_usd=100000.0,
            gas_cost_usd=5.0,
            confidence_score=0.9,
            last_updated_timestamp=time.time()
        )
        
        simulator._get_edge_state = AsyncMock(return_value=good_state)
        
        # Mock basic simulation result
        from yield_arbitrage.execution.hybrid_simulator import SimulationResult
        mock_result = SimulationResult(
            success=True,
            simulation_mode=SimulationMode.TENDERLY.value,
            profit_usd=10.0,
            output_amount=1.004
        )
        
        simulator._simulate_tenderly = AsyncMock(return_value=mock_result)
        # Add Tenderly config to make it work
        from yield_arbitrage.execution.hybrid_simulator import TenderlyConfig
        simulator.tenderly_config = TenderlyConfig(
            api_key="test_key",
            username="test_user", 
            project_slug="test_project"
        )
        
        result = await simulator.simulate_path(
            path=path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.TENDERLY
        )
        
        assert result.success is True
        assert result.profit_usd == 10.0
        # Should not have critical validation warnings since state is good
    
    async def test_simulation_with_validation_failure(self, simulator):
        """Test simulation that fails validation."""
        # Create valid edge but mock validation to fail
        path = [
            YieldGraphEdge(
                edge_id="valid_edge",
                source_asset_id="ETH_MAINNET_WETH",
                target_asset_id="ETH_MAINNET_USDC",
                edge_type=EdgeType.TRADE,
                protocol_name="uniswap_v2",
                chain_name="ethereum"
            )
        ]
        
        # Mock validation to fail
        async def mock_validation_failure(path):
            return {
                "valid": False,
                "issues": ["Test validation failure"],
                "warnings": ["Test warning"]
            }
        
        simulator._validate_path_edges = AsyncMock(side_effect=mock_validation_failure)
        
        simulator._get_edge_state = AsyncMock(return_value=None)
        
        result = await simulator.simulate_path(
            path=path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            mode=SimulationMode.TENDERLY
        )
        
        assert result.success is False
        assert "Path validation failed" in result.revert_reason
        assert "Test validation failure" in result.revert_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])