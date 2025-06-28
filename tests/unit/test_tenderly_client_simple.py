"""Simplified unit tests for Tenderly client integration."""
import pytest
from unittest.mock import Mock, AsyncMock, patch

from yield_arbitrage.execution.tenderly_client import (
    TenderlyClient,
    TenderlyTransaction,
    TenderlySimulationResult,
    TenderlyNetworkId,
)


class TestTenderlyClientBasic:
    """Basic tests for TenderlyClient."""
    
    @pytest.fixture
    def tenderly_client(self):
        """Create TenderlyClient instance."""
        return TenderlyClient(
            api_key="test_api_key",
            username="test_user",
            project_slug="test_project"
        )
    
    def test_client_initialization(self, tenderly_client):
        """Test client initialization."""
        assert tenderly_client.api_key == "test_api_key"
        assert tenderly_client.username == "test_user"
        assert tenderly_client.project_slug == "test_project"
        assert tenderly_client.session is None
    
    def test_get_stats(self, tenderly_client):
        """Test statistics retrieval."""
        stats = tenderly_client.get_stats()
        
        assert "simulations_run" in stats
        assert "forks_created" in stats
        assert "session_active" in stats
        assert stats["simulations_run"] == 0
        assert stats["session_active"] is False
    
    def test_revert_reason_decoding(self, tenderly_client):
        """Test revert reason decoding."""
        # Standard Error(string) encoding for "Insufficient allowance"
        encoded_output = "0x08c379a00000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000001a496e73756666696369656e7420616c6c6f77616e63650000000000000000000000"
        
        reason = tenderly_client._decode_revert_reason(encoded_output)
        assert reason == "Insufficient allowance"
        
        # Test with empty output
        empty_reason = tenderly_client._decode_revert_reason("0x")
        assert empty_reason is None


class TestTenderlyTransaction:
    """Test TenderlyTransaction dataclass."""
    
    def test_transaction_creation(self):
        """Test basic transaction creation."""
        tx = TenderlyTransaction(
            from_address="0x1234567890123456789012345678901234567890",
            to_address="0x0987654321098765432109876543210987654321",
            value="1000000000000000000",
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


class TestTenderlySimulationResult:
    """Test TenderlySimulationResult."""
    
    def test_successful_result_creation(self):
        """Test creation of successful simulation result."""
        result = TenderlySimulationResult(
            success=True,
            gas_used=21000,
            gas_cost_usd=10.5,
            transaction_hash="0xabcdef"
        )
        
        assert result.success is True
        assert result.gas_used == 21000
        assert result.gas_cost_usd == 10.5
        assert result.transaction_hash == "0xabcdef"
    
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