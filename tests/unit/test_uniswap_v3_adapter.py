"""Unit tests for Uniswap V3 adapter."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from yield_arbitrage.protocols.uniswap_v3_adapter import UniswapV3Adapter
from yield_arbitrage.protocols.token_filter import TokenFilter, TokenCriteria
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeState, EdgeType
from yield_arbitrage.protocols.base_adapter import ProtocolError


class TestUniswapV3Adapter:
    """Test UniswapV3Adapter class."""
    
    @pytest.fixture
    def mock_provider(self):
        """Mock blockchain provider."""
        provider = Mock()
        provider.get_web3 = AsyncMock()
        
        # Mock Web3 instance
        mock_web3 = Mock()
        mock_web3.to_checksum_address = Mock(side_effect=lambda x: x.upper())
        mock_web3.eth = Mock()
        mock_web3.eth.contract = Mock()
        
        provider.get_web3.return_value = mock_web3
        return provider
    
    @pytest.fixture
    def mock_token_filter(self):
        """Mock token filter."""
        token_filter = Mock(spec=TokenFilter)
        token_filter.initialize = AsyncMock()
        token_filter.filter_tokens = AsyncMock(return_value=[
            "0xa0b86a33e6776c92e7f8d2c0b5f8f8c5e4a7b8c9",  # Mock WETH
            "0xa0b73e1ff0b8c5d7e8b9f9f8f5e4d3c2b1a0f9e8"   # Mock USDC
        ])
        return token_filter
    
    @pytest.fixture
    def adapter(self, mock_provider, mock_token_filter):
        """Create UniswapV3Adapter for testing."""
        return UniswapV3Adapter("ethereum", mock_provider, mock_token_filter)
    
    def test_adapter_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter.chain_name == "ethereum"
        assert adapter.protocol_name == "uniswapv3"
        assert adapter.supported_edge_types == [EdgeType.TRADE]
        assert not adapter.is_initialized
        assert len(adapter.STANDARD_FEE_TIERS) == 4
        assert adapter.discovered_pools == set()
    
    @pytest.mark.asyncio
    async def test_protocol_specific_init_success(self, adapter, mock_provider):
        """Test successful protocol-specific initialization."""
        # Mock contract creation
        mock_web3 = await mock_provider.get_web3("ethereum")
        mock_factory = Mock()
        mock_quoter = Mock()
        mock_web3.eth.contract.side_effect = [mock_factory, mock_quoter]
        
        # Mock token filter initialization
        adapter.token_filter.initialize = AsyncMock()
        
        # Test initialization
        result = await adapter._protocol_specific_init()
        
        assert result is True
        assert adapter.factory_contract is mock_factory
        assert adapter.quoter_contract is mock_quoter
        adapter.token_filter.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_protocol_specific_init_no_contracts(self, mock_provider, mock_token_filter):
        """Test initialization with unsupported chain."""
        adapter = UniswapV3Adapter("unsupported_chain", mock_provider, mock_token_filter)
        
        result = await adapter._protocol_specific_init()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_protocol_specific_init_no_web3(self, mock_provider, mock_token_filter):
        """Test initialization when Web3 is unavailable."""
        mock_provider.get_web3.return_value = None
        adapter = UniswapV3Adapter("ethereum", mock_provider, mock_token_filter)
        
        result = await adapter._protocol_specific_init()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_filtered_tokens(self, adapter):
        """Test token filtering."""
        tokens = await adapter._get_filtered_tokens()
        
        assert len(tokens) == 2
        adapter.token_filter.filter_tokens.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_filtered_tokens_empty(self, adapter):
        """Test token filtering with no results."""
        adapter.token_filter.filter_tokens.return_value = []
        
        tokens = await adapter._get_filtered_tokens()
        
        assert tokens == []
    
    @pytest.mark.asyncio
    async def test_get_pool_address_success(self, adapter):
        """Test successful pool address retrieval."""
        # Setup mocks
        await adapter._protocol_specific_init()
        mock_factory = adapter.factory_contract
        mock_factory.functions.getPool.return_value.call = AsyncMock(
            return_value="0x1234567890123456789012345678901234567890"
        )
        
        pool_address = await adapter._get_pool_address(
            "0xtoken0", "0xtoken1", 3000
        )
        
        assert pool_address == "0x1234567890123456789012345678901234567890"
    
    @pytest.mark.asyncio
    async def test_get_pool_address_none(self, adapter):
        """Test pool address retrieval when pool doesn't exist."""
        await adapter._protocol_specific_init()
        mock_factory = adapter.factory_contract
        mock_factory.functions.getPool.return_value.call = AsyncMock(
            return_value="0x0000000000000000000000000000000000000000"
        )
        
        pool_address = await adapter._get_pool_address(
            "0xtoken0", "0xtoken1", 3000
        )
        
        assert pool_address == "0x0000000000000000000000000000000000000000"
    
    @pytest.mark.asyncio
    async def test_get_basic_pool_state_success(self, adapter, mock_provider):
        """Test successful pool state retrieval."""
        # Mock Web3 and contract
        mock_web3 = await mock_provider.get_web3("ethereum")
        mock_pool_contract = Mock()
        mock_pool_contract.functions.liquidity.return_value.call = AsyncMock(return_value=1000000)
        mock_pool_contract.functions.slot0.return_value.call = AsyncMock(
            return_value=[79228162514264337593543950336, 0, 1, 1, 1, 0, True]  # Mock slot0 data
        )
        mock_web3.eth.contract.return_value = mock_pool_contract
        
        pool_state = await adapter._get_basic_pool_state("0xpool")
        
        assert pool_state is not None
        assert pool_state["liquidity"] == 1000000
        assert pool_state["sqrt_price_x96"] == 79228162514264337593543950336
        assert pool_state["tick"] == 0
    
    @pytest.mark.asyncio
    async def test_get_basic_pool_state_failure(self, adapter, mock_provider):
        """Test pool state retrieval failure."""
        mock_web3 = await mock_provider.get_web3("ethereum")
        mock_pool_contract = Mock()
        mock_pool_contract.functions.liquidity.return_value.call = AsyncMock(
            side_effect=Exception("Contract call failed")
        )
        mock_web3.eth.contract.return_value = mock_pool_contract
        
        pool_state = await adapter._get_basic_pool_state("0xpool")
        
        assert pool_state is None
    
    @pytest.mark.asyncio
    async def test_get_token_decimals_success(self, adapter, mock_provider):
        """Test successful token decimals retrieval."""
        mock_web3 = await mock_provider.get_web3("ethereum")
        mock_token_contract = Mock()
        mock_token_contract.functions.decimals.return_value.call = AsyncMock(return_value=18)
        mock_web3.eth.contract.return_value = mock_token_contract
        
        decimals = await adapter._get_token_decimals("0xtoken")
        
        assert decimals == 18
        assert adapter.token_decimals_cache["0xtoken"] == 18
    
    @pytest.mark.asyncio
    async def test_get_token_decimals_cached(self, adapter):
        """Test token decimals retrieval from cache."""
        # Pre-populate cache
        adapter.token_decimals_cache["0xtoken"] = 6
        
        decimals = await adapter._get_token_decimals("0xtoken")
        
        assert decimals == 6
    
    @pytest.mark.asyncio
    async def test_get_token_decimals_failure(self, adapter, mock_provider):
        """Test token decimals retrieval failure."""
        mock_web3 = await mock_provider.get_web3("ethereum")
        mock_token_contract = Mock()
        mock_token_contract.functions.decimals.return_value.call = AsyncMock(
            side_effect=Exception("Contract call failed")
        )
        mock_web3.eth.contract.return_value = mock_token_contract
        
        decimals = await adapter._get_token_decimals("0xtoken")
        
        assert decimals == 18  # Default value
    
    @pytest.mark.asyncio
    async def test_get_conversion_rate_success(self, adapter, mock_provider):
        """Test successful conversion rate calculation."""
        # Setup adapter
        await adapter._protocol_specific_init()
        
        # Mock token decimals
        adapter._get_token_decimals = AsyncMock(side_effect=[18, 6])  # WETH: 18, USDC: 6
        
        # Mock quoter contract
        mock_quoter = adapter.quoter_contract
        mock_quoter.functions.quoteExactInputSingle.return_value.call = AsyncMock(
            return_value=1500000000  # 1500 USDC (6 decimals) for 1M units of input
        )
        
        rate = await adapter._get_conversion_rate(
            "0xweth", "0xusdc", 3000, 1000000  # 1M units
        )
        
        # Expected calculation:
        # Input: 1M units (normalized)
        # Output: 1500000000 (raw) / 10^6 (decimals) = 1500 USDC (normalized)
        # Rate: 1500 / 1M = 0.0015 USDC per unit of input
        assert rate == 0.0015
    
    @pytest.mark.asyncio
    async def test_get_conversion_rate_failure(self, adapter, mock_provider):
        """Test conversion rate calculation failure."""
        await adapter._protocol_specific_init()
        
        # Mock failure in getting decimals
        adapter._get_token_decimals = AsyncMock(return_value=None)
        
        rate = await adapter._get_conversion_rate(
            "0xweth", "0xusdc", 3000, 1000000
        )
        
        assert rate is None
    
    @pytest.mark.asyncio
    async def test_discover_pool_edges_success(self, adapter):
        """Test successful pool edge discovery."""
        # Mock dependencies
        adapter._get_pool_address = AsyncMock(return_value="0xpool123")
        adapter._get_basic_pool_state = AsyncMock(return_value={
            "liquidity": 1000000,
            "sqrt_price_x96": 79228162514264337593543950336,
            "tick": 0,
            "tvl_usd": 2000000.0
        })
        
        edges = await adapter._discover_pool_edges("0xtoken0", "0xtoken1", 3000)
        
        assert len(edges) == 2  # Bidirectional edges
        assert all(isinstance(edge, YieldGraphEdge) for edge in edges)
        assert all(edge.edge_type == EdgeType.TRADE for edge in edges)
        assert all(edge.protocol_name == "uniswapv3" for edge in edges)
        assert all(edge.chain_name == "ethereum" for edge in edges)
        
        # Check that pool is marked as discovered
        assert "0xpool123" in adapter.discovered_pools
        
        # Check edge metadata (now stored in adapter cache)
        edge = edges[0]
        metadata = adapter.pool_metadata_cache[edge.edge_id]
        assert metadata["pool_address"] == "0xpool123"
        assert metadata["fee_tier"] == 3000
        assert metadata["fee_percentage"] == 0.003  # 0.3%
    
    @pytest.mark.asyncio
    async def test_discover_pool_edges_no_pool(self, adapter):
        """Test pool edge discovery when pool doesn't exist."""
        adapter._get_pool_address = AsyncMock(return_value="0x0000000000000000000000000000000000000000")
        
        edges = await adapter._discover_pool_edges("0xtoken0", "0xtoken1", 3000)
        
        assert edges == []
    
    @pytest.mark.asyncio
    async def test_discover_pool_edges_no_liquidity(self, adapter):
        """Test pool edge discovery when pool has no liquidity."""
        adapter._get_pool_address = AsyncMock(return_value="0xpool123")
        adapter._get_basic_pool_state = AsyncMock(return_value={
            "liquidity": 0,  # No liquidity
            "sqrt_price_x96": 79228162514264337593543950336,
            "tick": 0
        })
        
        edges = await adapter._discover_pool_edges("0xtoken0", "0xtoken1", 3000)
        
        assert edges == []
    
    @pytest.mark.asyncio
    async def test_discover_pool_edges_already_discovered(self, adapter):
        """Test pool edge discovery when pool already discovered."""
        # Mark pool as already discovered
        adapter.discovered_pools.add("0xpool123")
        
        adapter._get_pool_address = AsyncMock(return_value="0xpool123")
        
        edges = await adapter._discover_pool_edges("0xtoken0", "0xtoken1", 3000)
        
        assert edges == []
    
    @pytest.mark.asyncio
    async def test_discover_edges_integration(self, adapter):
        """Test full edge discovery integration."""
        # Mock token filtering
        adapter._get_filtered_tokens = AsyncMock(return_value=["0xtoken0", "0xtoken1"])
        
        # Mock pool discovery
        adapter._discover_pool_edges = AsyncMock(return_value=[
            YieldGraphEdge(
                edge_id="test_edge",
                edge_type=EdgeType.TRADE,
                source_asset_id="ethereum_TOKEN_0xtoken0",
                target_asset_id="ethereum_TOKEN_0xtoken1",
                protocol_name="uniswapv3",
                chain_name="ethereum"
            )
        ])
        
        edges = await adapter.discover_edges()
        
        # Should discover edges for all fee tiers
        expected_calls = len(adapter.STANDARD_FEE_TIERS)  # 4 fee tiers for 1 token pair
        assert adapter._discover_pool_edges.call_count == expected_calls
        assert len(edges) == expected_calls  # One edge per call
    
    @pytest.mark.asyncio
    async def test_discover_edges_no_tokens(self, adapter):
        """Test edge discovery with no tokens."""
        adapter._get_filtered_tokens = AsyncMock(return_value=[])
        
        edges = await adapter.discover_edges()
        
        assert edges == []
    
    @pytest.mark.asyncio
    async def test_update_edge_state_success(self, adapter):
        """Test successful edge state update."""
        # Create edge
        edge = YieldGraphEdge(
            edge_id="test_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="ethereum_TOKEN_0xtoken0",
            target_asset_id="ethereum_TOKEN_0xtoken1",
            protocol_name="uniswapv3",
            chain_name="ethereum"
        )
        
        # Add metadata to adapter cache
        adapter.pool_metadata_cache["test_edge"] = {
            "pool_address": "0xpool123",
            "token0_address": "0xtoken0",
            "token1_address": "0xtoken1",
            "fee_tier": 3000
        }
        
        # Mock dependencies
        adapter._get_detailed_pool_state = AsyncMock(return_value={
            "liquidity": 1000000,
            "tvl_usd": 2000000.0
        })
        adapter._get_conversion_rate = AsyncMock(return_value=1500.0)
        
        # Update edge state
        new_state = await adapter.update_edge_state(edge)
        
        assert isinstance(new_state, EdgeState)
        assert new_state.conversion_rate == 1500.0
        assert new_state.liquidity_usd == 2000000.0
        assert new_state.confidence_score == 0.95
    
    @pytest.mark.asyncio
    async def test_update_edge_state_missing_metadata(self, adapter):
        """Test edge state update with missing metadata."""
        edge = YieldGraphEdge(
            edge_id="test_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="ethereum_TOKEN_0xtoken0",
            target_asset_id="ethereum_TOKEN_0xtoken1",
            protocol_name="uniswapv3",
            chain_name="ethereum"
        )
        
        # Don't add metadata to cache - should fail
        with pytest.raises(ProtocolError, match="Missing pool address metadata"):
            await adapter.update_edge_state(edge)
    
    @pytest.mark.asyncio
    async def test_update_edge_state_failure(self, adapter):
        """Test edge state update failure."""
        edge = YieldGraphEdge(
            edge_id="test_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="ethereum_TOKEN_0xtoken0",
            target_asset_id="ethereum_TOKEN_0xtoken1",
            protocol_name="uniswapv3",
            chain_name="ethereum",
            state=EdgeState(confidence_score=0.8)
        )
        
        # Add metadata to adapter cache
        adapter.pool_metadata_cache["test_edge"] = {
            "pool_address": "0xpool123",
            "token0_address": "0xtoken0",
            "token1_address": "0xtoken1",
            "fee_tier": 3000
        }
        
        # Mock failure
        adapter._get_detailed_pool_state = AsyncMock(return_value=None)
        
        # Should return existing state with reduced confidence
        new_state = await adapter.update_edge_state(edge)
        
        assert new_state.confidence_score <= 0.4  # Reduced from 0.8
    
    def test_estimate_gas_cost(self, adapter):
        """Test gas cost estimation."""
        cost = adapter._estimate_gas_cost()
        assert cost == 15.0  # Ethereum gas cost
        
        # Test other chains
        adapter.chain_name = "arbitrum"
        assert adapter._estimate_gas_cost() == 2.0
        
        adapter.chain_name = "unknown"
        assert adapter._estimate_gas_cost() == 5.0  # Default
    
    def test_get_supported_tokens(self, adapter):
        """Test getting supported tokens."""
        tokens = adapter.get_supported_tokens()
        assert isinstance(tokens, list)
        # Should contain well-known Ethereum tokens
        assert len(tokens) > 0
    
    def test_string_representations(self, adapter):
        """Test string representations."""
        str_repr = str(adapter)
        assert "UniswapV3Adapter" in str_repr
        assert "ethereum" in str_repr
        assert "pools=0" in str_repr
        
        repr_str = repr(adapter)
        assert "UniswapV3Adapter" in repr_str
        assert "protocol=uniswapv3" in repr_str
        assert "chain=ethereum" in repr_str
        assert "initialized=False" in repr_str