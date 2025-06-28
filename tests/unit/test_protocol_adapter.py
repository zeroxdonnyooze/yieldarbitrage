"""Unit tests for protocol adapter base class and registry."""
import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timezone

from yield_arbitrage.protocols.base_adapter import (
    ProtocolAdapter,
    ProtocolError,
    NetworkError
)
from yield_arbitrage.protocols.adapter_registry import (
    ProtocolAdapterRegistry,
    AdapterInfo,
    protocol_registry
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeState, EdgeType


class MockProtocolAdapter(ProtocolAdapter):
    """Mock protocol adapter for testing."""
    
    def __init__(self, chain_name: str, provider, should_fail_init: bool = False):
        super().__init__(chain_name, provider)
        self.protocol_name = "MockProtocol"
        self.supported_edge_types = ["TRADE"]
        self.should_fail_init = should_fail_init
        self.discover_called = False
        self.update_called = False
    
    async def discover_edges(self):
        """Mock edge discovery."""
        self.discover_called = True
        self._record_discovery_success(2)
        
        # Return mock edges
        return [
            YieldGraphEdge(
                edge_id="mock_edge_1",
                edge_type=EdgeType.TRADE,
                source_asset_id=f"{self.chain_name}_TOKEN_A",
                target_asset_id=f"{self.chain_name}_TOKEN_B",
                protocol_name=self.protocol_name,
                chain_name=self.chain_name
            ),
            YieldGraphEdge(
                edge_id="mock_edge_2",
                edge_type=EdgeType.TRADE,
                source_asset_id=f"{self.chain_name}_TOKEN_B",
                target_asset_id=f"{self.chain_name}_TOKEN_A",
                protocol_name=self.protocol_name,
                chain_name=self.chain_name
            )
        ]
    
    async def update_edge_state(self, edge: YieldGraphEdge):
        """Mock edge state update."""
        self.update_called = True
        self._record_update_success(0.5)
        
        return EdgeState(
            conversion_rate=1.05,
            liquidity_usd=100000.0,
            gas_cost_usd=5.0,
            last_updated_timestamp=datetime.now(timezone.utc).timestamp()
        )
    
    async def _protocol_specific_init(self):
        """Mock protocol-specific initialization."""
        if self.should_fail_init:
            return False
        return True


class FailingMockAdapter(ProtocolAdapter):
    """Mock adapter that fails for testing error handling."""
    
    def __init__(self, chain_name: str, provider):
        super().__init__(chain_name, provider)
        self.protocol_name = "FailingMock"
    
    async def discover_edges(self):
        """Always fails."""
        self._record_discovery_error()
        raise ProtocolError("Discovery failed", self.protocol_name, self.chain_name)
    
    async def update_edge_state(self, edge: YieldGraphEdge):
        """Always fails."""
        self._record_update_error()
        raise NetworkError("Update failed", self.chain_name, 3)


class TestProtocolAdapter:
    """Test the ProtocolAdapter base class."""
    
    @pytest.fixture
    def mock_provider(self):
        """Mock blockchain provider."""
        provider = Mock()
        provider.get_web3 = AsyncMock()
        return provider
    
    @pytest.fixture
    def adapter(self, mock_provider):
        """Create a mock adapter for testing."""
        return MockProtocolAdapter("ethereum", mock_provider)
    
    @pytest.fixture
    def failing_adapter(self, mock_provider):
        """Create a failing adapter for testing."""
        return FailingMockAdapter("ethereum", mock_provider)
    
    def test_adapter_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter.chain_name == "ethereum"
        assert adapter.protocol_name == "MockProtocol"
        assert adapter.supported_edge_types == ["TRADE"]
        assert not adapter.is_initialized
        assert adapter.provider is not None
    
    @pytest.mark.asyncio
    async def test_successful_initialization(self, adapter):
        """Test successful adapter initialization."""
        success = await adapter.initialize()
        
        assert success is True
        assert adapter.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_failed_initialization(self, mock_provider):
        """Test failed adapter initialization."""
        adapter = MockProtocolAdapter("ethereum", mock_provider, should_fail_init=True)
        
        success = await adapter.initialize()
        
        assert success is False
        assert adapter.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_discover_edges(self, adapter):
        """Test edge discovery."""
        edges = await adapter.discover_edges()
        
        assert len(edges) == 2
        assert adapter.discover_called is True
        assert all(isinstance(edge, YieldGraphEdge) for edge in edges)
        assert all(edge.protocol_name == "MockProtocol" for edge in edges)
        assert all(edge.chain_name == "ethereum" for edge in edges)
        
        # Check stats
        stats = adapter.get_discovery_stats()
        assert stats["discovery_stats"]["edges_discovered"] == 2
    
    @pytest.mark.asyncio
    async def test_update_edge_state(self, adapter):
        """Test edge state update."""
        edge = YieldGraphEdge(
            edge_id="test_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="ethereum_TOKEN_A",
            target_asset_id="ethereum_TOKEN_B",
            protocol_name="MockProtocol",
            chain_name="ethereum"
        )
        
        state = await adapter.update_edge_state(edge)
        
        assert adapter.update_called is True
        assert isinstance(state, EdgeState)
        assert state.conversion_rate == 1.05
        assert state.liquidity_usd == 100000.0
        
        # Check stats
        stats = adapter.get_discovery_stats()
        assert stats["update_stats"]["updates_performed"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_update_edges(self, adapter):
        """Test batch edge update."""
        edges = [
            YieldGraphEdge(
                edge_id="test_edge_1",
                edge_type=EdgeType.TRADE,
                source_asset_id="ethereum_TOKEN_A",
                target_asset_id="ethereum_TOKEN_B",
                protocol_name="MockProtocol",
                chain_name="ethereum"
            ),
            YieldGraphEdge(
                edge_id="test_edge_2",
                edge_type=EdgeType.TRADE,
                source_asset_id="ethereum_TOKEN_B",
                target_asset_id="ethereum_TOKEN_A",
                protocol_name="MockProtocol",
                chain_name="ethereum"
            )
        ]
        
        results = await adapter.batch_update_edges(edges)
        
        assert len(results) == 2
        assert "test_edge_1" in results
        assert "test_edge_2" in results
        assert all(isinstance(state, EdgeState) for state in results.values())
    
    @pytest.mark.asyncio
    async def test_discovery_error_handling(self, failing_adapter):
        """Test error handling in edge discovery."""
        with pytest.raises(ProtocolError):
            await failing_adapter.discover_edges()
        
        stats = failing_adapter.get_discovery_stats()
        assert stats["discovery_stats"]["discovery_errors"] == 1
    
    @pytest.mark.asyncio
    async def test_update_error_handling(self, failing_adapter):
        """Test error handling in edge updates."""
        edge = YieldGraphEdge(
            edge_id="test_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="ethereum_TOKEN_A",
            target_asset_id="ethereum_TOKEN_B",
            protocol_name="FailingMock",
            chain_name="ethereum"
        )
        
        with pytest.raises(NetworkError):
            await failing_adapter.update_edge_state(edge)
        
        stats = failing_adapter.get_discovery_stats()
        assert stats["update_stats"]["update_errors"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_update_with_errors(self, failing_adapter):
        """Test batch update with some failures."""
        edge = YieldGraphEdge(
            edge_id="test_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="ethereum_TOKEN_A",
            target_asset_id="ethereum_TOKEN_B",
            protocol_name="FailingMock",
            chain_name="ethereum"
        )
        
        results = await failing_adapter.batch_update_edges([edge])
        
        # Should return existing state when update fails
        assert len(results) == 1
        assert "test_edge" in results
        assert results["test_edge"] == edge.state
    
    def test_string_representations(self, adapter):
        """Test string representations."""
        str_repr = str(adapter)
        assert "MockProtocol" in str_repr
        assert "ethereum" in str_repr
        
        repr_str = repr(adapter)
        assert "MockProtocolAdapter" in repr_str
        assert "protocol=MockProtocol" in repr_str
        assert "chain=ethereum" in repr_str


class TestProtocolErrors:
    """Test custom exception classes."""
    
    def test_protocol_error(self):
        """Test ProtocolError."""
        error = ProtocolError("Test error", "TestProtocol", "ethereum")
        
        assert error.protocol == "TestProtocol"
        assert error.chain == "ethereum"
        assert "[TestProtocol@ethereum]" in str(error)
    
    def test_protocol_error_without_context(self):
        """Test ProtocolError without protocol/chain context."""
        error = ProtocolError("Test error")
        
        assert error.protocol is None
        assert error.chain is None
        assert str(error) == "Test error"
    
    def test_network_error(self):
        """Test NetworkError."""
        error = NetworkError("Connection failed", "ethereum", 3)
        
        assert error.chain == "ethereum"
        assert error.retry_count == 3
        assert "[ethereum]" in str(error)
        assert "retries: 3" in str(error)


class TestProtocolAdapterRegistry:
    """Test the ProtocolAdapterRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry for testing."""
        return ProtocolAdapterRegistry()
    
    @pytest.fixture
    def mock_provider(self):
        """Mock blockchain provider."""
        provider = Mock()
        provider.get_web3 = AsyncMock()
        return provider
    
    def test_register_adapter(self, registry):
        """Test adapter registration."""
        success = registry.register_adapter(
            MockProtocolAdapter,
            "MockProtocol",
            ["ethereum", "arbitrum"],
            "Mock protocol for testing"
        )
        
        assert success is True
        assert "MockProtocol" in registry.list_protocols()
        assert registry.list_chains_for_protocol("MockProtocol") == ["ethereum", "arbitrum"]
    
    def test_register_invalid_adapter(self, registry):
        """Test registering invalid adapter class."""
        success = registry.register_adapter(
            str,  # Invalid adapter class
            "InvalidProtocol",
            ["ethereum"],
            "Invalid adapter"
        )
        
        assert success is False
        assert "InvalidProtocol" not in registry.list_protocols()
    
    @pytest.mark.asyncio
    async def test_initialize_adapter(self, registry, mock_provider):
        """Test adapter initialization."""
        # Register adapter first
        registry.register_adapter(
            MockProtocolAdapter,
            "MockProtocol",
            ["ethereum", "arbitrum"],
            "Mock protocol for testing"
        )
        
        # Initialize adapter
        adapter = await registry.initialize_adapter("MockProtocol", "ethereum", mock_provider)
        
        assert adapter is not None
        assert adapter.is_initialized is True
        assert adapter.protocol_name == "MockProtocol"
        assert adapter.chain_name == "ethereum"
        
        # Check that adapter is stored
        stored_adapter = registry.get_adapter("MockProtocol", "ethereum")
        assert stored_adapter is adapter
    
    @pytest.mark.asyncio
    async def test_initialize_unregistered_adapter(self, registry, mock_provider):
        """Test initializing unregistered adapter."""
        adapter = await registry.initialize_adapter("UnregisteredProtocol", "ethereum", mock_provider)
        
        assert adapter is None
    
    @pytest.mark.asyncio
    async def test_initialize_unsupported_chain(self, registry, mock_provider):
        """Test initializing adapter for unsupported chain."""
        registry.register_adapter(
            MockProtocolAdapter,
            "MockProtocol",
            ["ethereum"],  # Only ethereum supported
            "Mock protocol for testing"
        )
        
        adapter = await registry.initialize_adapter("MockProtocol", "polygon", mock_provider)
        
        assert adapter is None
    
    @pytest.mark.asyncio
    async def test_initialize_disabled_protocol(self, registry, mock_provider):
        """Test initializing disabled protocol."""
        registry.register_adapter(
            MockProtocolAdapter,
            "MockProtocol",
            ["ethereum"],
            "Mock protocol for testing",
            enabled=False
        )
        
        adapter = await registry.initialize_adapter("MockProtocol", "ethereum", mock_provider)
        
        assert adapter is None
    
    @pytest.mark.asyncio
    async def test_initialize_existing_adapter(self, registry, mock_provider):
        """Test initializing adapter that already exists."""
        registry.register_adapter(
            MockProtocolAdapter,
            "MockProtocol",
            ["ethereum"],
            "Mock protocol for testing"
        )
        
        # Initialize once
        adapter1 = await registry.initialize_adapter("MockProtocol", "ethereum", mock_provider)
        
        # Initialize again - should return existing adapter
        adapter2 = await registry.initialize_adapter("MockProtocol", "ethereum", mock_provider)
        
        assert adapter1 is adapter2
    
    def test_list_operations(self, registry):
        """Test list operations."""
        # Initially empty
        assert registry.list_protocols() == []
        assert registry.list_initialized_adapters() == []
        
        # Register some adapters
        registry.register_adapter(
            MockProtocolAdapter,
            "Protocol1",
            ["ethereum", "arbitrum"],
            "First protocol"
        )
        registry.register_adapter(
            MockProtocolAdapter,
            "Protocol2",
            ["base"],
            "Second protocol"
        )
        
        protocols = registry.list_protocols()
        assert "Protocol1" in protocols
        assert "Protocol2" in protocols
        
        chains1 = registry.list_chains_for_protocol("Protocol1")
        assert chains1 == ["ethereum", "arbitrum"]
        
        chains2 = registry.list_chains_for_protocol("Protocol2")
        assert chains2 == ["base"]
    
    def test_enable_disable_protocol(self, registry):
        """Test enabling and disabling protocols."""
        registry.register_adapter(
            MockProtocolAdapter,
            "MockProtocol",
            ["ethereum"],
            "Mock protocol for testing"
        )
        
        # Protocol should be enabled by default
        stats = registry.get_adapter_stats()
        assert stats["protocols"]["MockProtocol"]["is_enabled"] is True
        
        # Disable protocol
        success = registry.disable_protocol("MockProtocol")
        assert success is True
        
        stats = registry.get_adapter_stats()
        assert stats["protocols"]["MockProtocol"]["is_enabled"] is False
        
        # Re-enable protocol
        success = registry.enable_protocol("MockProtocol")
        assert success is True
        
        stats = registry.get_adapter_stats()
        assert stats["protocols"]["MockProtocol"]["is_enabled"] is True
    
    def test_enable_disable_nonexistent_protocol(self, registry):
        """Test enabling/disabling nonexistent protocol."""
        assert registry.enable_protocol("NonexistentProtocol") is False
        assert registry.disable_protocol("NonexistentProtocol") is False
    
    def test_adapter_stats(self, registry):
        """Test adapter statistics."""
        # Register adapters
        registry.register_adapter(
            MockProtocolAdapter,
            "Protocol1",
            ["ethereum"],
            "First protocol"
        )
        registry.register_adapter(
            MockProtocolAdapter,
            "Protocol2",
            ["arbitrum"],
            "Second protocol",
            enabled=False
        )
        
        stats = registry.get_adapter_stats()
        
        assert stats["registered_protocols"] == 2
        assert stats["enabled_protocols"] == 1
        assert stats["initialized_adapters"] == 0
        
        assert "Protocol1" in stats["protocols"]
        assert "Protocol2" in stats["protocols"]
        assert stats["protocols"]["Protocol1"]["is_enabled"] is True
        assert stats["protocols"]["Protocol2"]["is_enabled"] is False
    
    @pytest.mark.asyncio
    async def test_shutdown_all(self, registry, mock_provider):
        """Test shutting down all adapters."""
        registry.register_adapter(
            MockProtocolAdapter,
            "MockProtocol",
            ["ethereum"],
            "Mock protocol for testing"
        )
        
        # Initialize adapter
        adapter = await registry.initialize_adapter("MockProtocol", "ethereum", mock_provider)
        assert adapter is not None
        
        # Shutdown all
        await registry.shutdown_all()
        
        # Adapter should no longer be accessible
        stored_adapter = registry.get_adapter("MockProtocol", "ethereum")
        assert stored_adapter is None