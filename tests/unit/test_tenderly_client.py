"""Unit tests for Tenderly client integration."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json

from yield_arbitrage.execution.tenderly_client import (
    TenderlyClient,
    TenderlyTransaction,
    TenderlySimulationResult,
    TenderlyFork,
    TenderlyNetworkId,
    TenderlyAPIError,
    TenderlyAuthError,
    TenderlyRateLimitError,
    TenderlyNetworkError
)


class TestTenderlyTransaction:
    """Test TenderlyTransaction dataclass."""
    
    def test_transaction_creation(self):
        """Test basic transaction creation."""
        tx = TenderlyTransaction(
            from_address="0x1234567890123456789012345678901234567890",
            to_address="0x0987654321098765432109876543210987654321",
            value="1000000000000000000",  # 1 ETH in wei
            data="0x12345678"
        )
        
        assert tx.from_address == "0x1234567890123456789012345678901234567890"
        assert tx.to_address == "0x0987654321098765432109876543210987654321"
        assert tx.value == "1000000000000000000"
        assert tx.data == "0x12345678"
    
    def test_transaction_to_dict(self):
        """Test transaction serialization."""
        tx = TenderlyTransaction(
            from_address="0x1234567890123456789012345678901234567890",
            to_address="0x0987654321098765432109876543210987654321",
            value="1000000000000000000",
            gas=21000,
            gas_price="20000000000",
            data="0x12345678"
        )
        
        tx_dict = tx.to_dict()
        
        assert tx_dict["from"] == "0x1234567890123456789012345678901234567890"
        assert tx_dict["to"] == "0x0987654321098765432109876543210987654321"
        assert tx_dict["value"] == "1000000000000000000"
        assert tx_dict["gas"] == hex(21000)
        assert tx_dict["gas_price"] == "20000000000"
        assert tx_dict["input"] == "0x12345678"


class TestTenderlyClient:
    """Test TenderlyClient functionality."""
    
    @pytest.fixture
    def tenderly_config(self):
        """Create Tenderly configuration."""
        return {
            "api_key": "test_api_key",
            "username": "test_user",
            "project_slug": "test_project"
        }
    
    @pytest.fixture
    def tenderly_client(self, tenderly_config):
        """Create TenderlyClient instance."""
        return TenderlyClient(**tenderly_config)
    
    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session."""
        session = Mock()
        session.close = AsyncMock()
        return session
    
    @pytest.fixture
    def sample_transaction(self):
        """Create a sample transaction."""
        return TenderlyTransaction(
            from_address="0x1234567890123456789012345678901234567890",
            to_address="0xA0b86a33E6417c5c6eF57e11d8CAf7d8C0f7C8F8",  # USDC
            data="0xa9059cbb000000000000000000000000987654321098765432109876543210987654321000000000000000000000000000000000000000000000000000000000000186a0"
        )
    
    def test_client_initialization(self, tenderly_client):
        """Test client initialization."""
        assert tenderly_client.api_key == "test_api_key"
        assert tenderly_client.username == "test_user"
        assert tenderly_client.project_slug == "test_project"
        assert tenderly_client.session is None
        assert len(tenderly_client._active_forks) == 0
    
    @pytest.mark.asyncio
    async def test_client_initialization_and_cleanup(self, tenderly_client):
        """Test async initialization and cleanup."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session.close = AsyncMock()
            
            # Create a proper async context manager mock
            mock_response = Mock()
            mock_response.status = 200
            
            async_context_manager = AsyncMock()
            async_context_manager.__aenter__.return_value = mock_response
            async_context_manager.__aexit__.return_value = None
            
            mock_session.get.return_value = async_context_manager
            mock_session_class.return_value = mock_session
            
            # Initialize
            await tenderly_client.initialize()
            assert tenderly_client.session is not None
            
            # Cleanup
            await tenderly_client.close()
            mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_validation_auth_error(self, tenderly_client):
        """Test API validation with authentication error."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status = 401
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            tenderly_client.session = mock_session
            
            with pytest.raises(TenderlyAuthError):
                await tenderly_client._validate_api_access()
    
    @pytest.mark.asyncio
    async def test_simulation_success(self, tenderly_client, sample_transaction):
        """Test successful transaction simulation."""
        # Mock successful simulation response
        simulation_response = {
            "transaction": {
                "status": True,
                "gas_used": "0x5208",  # 21000 in hex
                "hash": "0xabcdef",
                "block_number": 18000000,
                "logs": [],
                "call_trace": {"calls": []}
            },
            "state_changes": {}
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=simulation_response)
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            tenderly_client.session = mock_session
            
            result = await tenderly_client.simulate_transaction(sample_transaction)
            
            assert result.success is True
            assert result.gas_used == 21000
            assert result.transaction_hash == "0xabcdef"
            assert result.block_number == 18000000
    
    @pytest.mark.asyncio
    async def test_simulation_failure(self, tenderly_client, sample_transaction):
        """Test failed transaction simulation."""
        simulation_response = {
            "transaction": {
                "status": False,
                "gas_used": "0x0",
                "error_message": "Transaction reverted",
                "call_trace": {
                    "output": "0x08c379a00000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000001a496e73756666696369656e7420616c6c6f77616e63650000000000000000000000"
                }
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=simulation_response)
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            tenderly_client.session = mock_session
            
            result = await tenderly_client.simulate_transaction(sample_transaction)
            
            assert result.success is False
            assert result.gas_used == 0
            assert result.error_message == "Transaction reverted"
            assert result.revert_reason is not None
    
    @pytest.mark.asyncio
    async def test_rate_limit_error(self, tenderly_client, sample_transaction):
        """Test rate limit handling."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status = 429
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            tenderly_client.session = mock_session
            
            with pytest.raises(TenderlyRateLimitError):
                await tenderly_client.simulate_transaction(sample_transaction)
    
    @pytest.mark.asyncio
    async def test_create_fork_success(self, tenderly_client):
        """Test successful fork creation."""
        fork_response = {
            "root_transaction": {
                "fork_id": "test_fork_123",
                "block_number": 18000000
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=fork_response)
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            tenderly_client.session = mock_session
            
            fork = await tenderly_client.create_fork(
                TenderlyNetworkId.ETHEREUM,
                block_number=18000000,
                alias="test_fork"
            )
            
            assert fork.fork_id == "test_fork_123"
            assert fork.network_id == "1"
            assert fork.block_number == 18000000
            assert fork.alias == "test_fork"
            assert fork.fork_id in tenderly_client._active_forks
    
    @pytest.mark.asyncio
    async def test_delete_fork_success(self, tenderly_client):
        """Test successful fork deletion."""
        # Add a fork to active forks
        fork = TenderlyFork(
            fork_id="test_fork_123",
            network_id="1",
            block_number=18000000,
            created_at=datetime.utcnow()
        )
        tenderly_client._active_forks["test_fork_123"] = fork
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status = 204
            mock_session.delete.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            tenderly_client.session = mock_session
            
            success = await tenderly_client.delete_fork("test_fork_123")
            
            assert success is True
            assert "test_fork_123" not in tenderly_client._active_forks
    
    @pytest.mark.asyncio
    async def test_transaction_bundle_simulation(self, tenderly_client):
        """Test simulation of transaction bundle."""
        transactions = [
            TenderlyTransaction(
                from_address="0x1234567890123456789012345678901234567890",
                to_address="0xA0b86a33E6417c5c6eF57e11d8CAf7d8C0f7C8F8",
                data="0x095ea7b3"  # approve
            ),
            TenderlyTransaction(
                from_address="0x1234567890123456789012345678901234567890",
                to_address="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                data="0x38ed1739"  # swapExactTokensForTokens
            )
        ]
        
        # Mock fork creation and simulation responses
        fork_response = {
            "root_transaction": {
                "fork_id": "bundle_fork_123",
                "block_number": 18000000
            }
        }
        
        sim_response = {
            "transaction": {
                "status": True,
                "gas_used": "0x186a0",  # 100000 in hex
                "hash": "0xabcdef"
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            
            # Mock responses
            mock_post_response = Mock()
            mock_post_response.status = 200
            mock_post_response.json = AsyncMock()
            
            mock_delete_response = Mock()
            mock_delete_response.status = 204
            
            # Setup response sequence
            responses = [fork_response] + [sim_response] * len(transactions)
            mock_post_response.json.side_effect = responses
            
            mock_session.post.return_value.__aenter__.return_value = mock_post_response
            mock_session.delete.return_value.__aenter__.return_value = mock_delete_response
            mock_session_class.return_value = mock_session
            
            tenderly_client.session = mock_session
            
            results = await tenderly_client.simulate_transaction_bundle(transactions)
            
            assert len(results) == 2
            assert all(result.success for result in results)
            assert all(result.gas_used == 100000 for result in results)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, tenderly_client):
        """Test rate limiting functionality."""
        # Fill up the rate limit tracker
        now = datetime.utcnow()
        for i in range(50):
            tenderly_client._last_requests.append(now - timedelta(seconds=i))
        
        # This should trigger rate limiting
        with patch('asyncio.sleep') as mock_sleep:
            await tenderly_client._ensure_rate_limit()
            mock_sleep.assert_called_once()
    
    def test_revert_reason_decoding(self, tenderly_client):
        """Test revert reason decoding."""
        # Standard Error(string) encoding for "Insufficient allowance"
        encoded_output = "0x08c379a00000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000001a496e73756666696369656e7420616c6c6f77616e63650000000000000000000000"
        
        reason = tenderly_client._decode_revert_reason(encoded_output)
        assert reason == "Insufficient allowance"
        
        # Test with invalid output
        invalid_reason = tenderly_client._decode_revert_reason("0x1234")
        assert invalid_reason.startswith("Raw output:")
        
        # Test with empty output
        empty_reason = tenderly_client._decode_revert_reason("0x")
        assert empty_reason is None
    
    def test_get_stats(self, tenderly_client):
        """Test statistics retrieval."""
        stats = tenderly_client.get_stats()
        
        assert "simulations_run" in stats
        assert "forks_created" in stats
        assert "forks_deleted" in stats
        assert "api_errors" in stats
        assert "active_forks" in stats
        assert "session_active" in stats
        
        # Initially all should be zero/False
        assert stats["simulations_run"] == 0
        assert stats["forks_created"] == 0
        assert stats["active_forks"] == 0
        assert stats["session_active"] is False


class TestTenderlySimulationResult:
    """Test TenderlySimulationResult dataclass."""
    
    def test_successful_result_creation(self):
        """Test creation of successful simulation result."""
        result = TenderlySimulationResult(
            success=True,
            gas_used=21000,
            gas_cost_usd=10.5,
            transaction_hash="0xabcdef",
            simulation_time_ms=250.0
        )
        
        assert result.success is True
        assert result.gas_used == 21000
        assert result.gas_cost_usd == 10.5
        assert result.transaction_hash == "0xabcdef"
        assert result.simulation_time_ms == 250.0
    
    def test_failed_result_creation(self):
        """Test creation of failed simulation result."""
        result = TenderlySimulationResult(
            success=False,
            gas_used=0,
            error_message="Transaction reverted",
            revert_reason="Insufficient balance"
        )
        
        assert result.success is False
        assert result.gas_used == 0
        assert result.error_message == "Transaction reverted"
        assert result.revert_reason == "Insufficient balance"


class TestTenderlyFork:
    """Test TenderlyFork dataclass."""
    
    def test_fork_creation(self):
        """Test fork creation."""
        created_at = datetime.utcnow()
        fork = TenderlyFork(
            fork_id="test_fork_123",
            network_id="1",
            block_number=18000000,
            created_at=created_at,
            alias="test_fork"
        )
        
        assert fork.fork_id == "test_fork_123"
        assert fork.network_id == "1"
        assert fork.block_number == 18000000
        assert fork.created_at == created_at
        assert fork.alias == "test_fork"
        assert fork.is_active is True
        assert fork.transactions_count == 0