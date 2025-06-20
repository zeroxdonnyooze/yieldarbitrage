"""Unit tests for blockchain provider."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

# Import after adding src to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from yield_arbitrage.blockchain_connector import BlockchainProvider
from yield_arbitrage.blockchain_connector.provider import ChainConfig


class TestChainConfig:
    """Test ChainConfig class."""
    
    def test_chain_config_creation(self):
        """Test creating a chain configuration."""
        config = ChainConfig(
            name="Ethereum",
            chain_id=1,
            rpc_url="https://ethereum.example.com",
            multicall_address="0x123...",
            block_explorer_url="https://etherscan.io",
            native_currency="ETH"
        )
        
        assert config.name == "Ethereum"
        assert config.chain_id == 1
        assert config.rpc_url == "https://ethereum.example.com"
        assert config.multicall_address == "0x123..."
        assert config.block_explorer_url == "https://etherscan.io"
        assert config.native_currency == "ETH"
    
    def test_chain_config_defaults(self):
        """Test chain configuration with default values."""
        config = ChainConfig(
            name="TestChain",
            chain_id=999,
            rpc_url="https://test.com"
        )
        
        assert config.name == "TestChain"
        assert config.chain_id == 999
        assert config.rpc_url == "https://test.com"
        assert config.multicall_address is None
        assert config.block_explorer_url is None
        assert config.native_currency == "ETH"


class TestBlockchainProvider:
    """Test BlockchainProvider class."""
    
    def setup_method(self):
        """Reset provider before each test."""
        self.provider = BlockchainProvider()
    
    def test_provider_initialization(self):
        """Test provider creates with correct initial state."""
        assert isinstance(self.provider.web3_instances, dict)
        assert isinstance(self.provider.chain_configs, dict)
        assert self.provider._initialized is False
        assert len(self.provider.web3_instances) == 0
        assert len(self.provider.chain_configs) == 0
    
    @pytest.mark.asyncio
    async def test_get_supported_chains_empty(self):
        """Test getting supported chains when none configured."""
        with patch.object(self.provider, 'initialize') as mock_init:
            mock_init.return_value = None
            chains = await self.provider.get_supported_chains()
            assert chains == []
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_setup_chain_configs(self):
        """Test setting up chain configurations."""
        with patch('yield_arbitrage.blockchain_connector.provider.settings') as mock_settings:
            mock_settings.ethereum_rpc_url = "https://eth.example.com"
            mock_settings.arbitrum_rpc_url = "https://arb.example.com"
            mock_settings.base_rpc_url = None  # Not configured
            mock_settings.sonic_rpc_url = None
            mock_settings.berachain_rpc_url = None
            
            await self.provider._setup_chain_configs()
            
            assert "ethereum" in self.provider.chain_configs
            assert "arbitrum" in self.provider.chain_configs
            assert "base" not in self.provider.chain_configs
            
            eth_config = self.provider.chain_configs["ethereum"]
            assert eth_config.name == "Ethereum"
            assert eth_config.chain_id == 1
            assert eth_config.rpc_url == "https://eth.example.com"
    
    @pytest.mark.asyncio
    async def test_get_web3_not_initialized(self):
        """Test getting Web3 instance triggers initialization."""
        with patch.object(self.provider, 'initialize') as mock_init:
            mock_init.return_value = None
            
            result = await self.provider.get_web3("ethereum")
            assert result is None
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_chain_config_not_initialized(self):
        """Test getting chain config triggers initialization."""
        with patch.object(self.provider, 'initialize') as mock_init:
            mock_init.return_value = None
            
            result = await self.provider.get_chain_config("ethereum")
            assert result is None
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_is_connected_no_web3(self):
        """Test is_connected returns False when no Web3 instance."""
        with patch.object(self.provider, 'get_web3') as mock_get_web3:
            mock_get_web3.return_value = None
            
            result = await self.provider.is_connected("ethereum")
            assert result is False
    
    @pytest.mark.asyncio
    async def test_is_connected_success(self):
        """Test is_connected returns True when Web3 is connected."""
        mock_w3 = AsyncMock()
        mock_w3.is_connected.return_value = True
        
        with patch.object(self.provider, 'get_web3') as mock_get_web3:
            mock_get_web3.return_value = mock_w3
            
            result = await self.provider.is_connected("ethereum")
            assert result is True
            mock_w3.is_connected.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_is_connected_exception(self):
        """Test is_connected returns False on exception."""
        mock_w3 = AsyncMock()
        mock_w3.is_connected.side_effect = Exception("Connection error")
        
        with patch.object(self.provider, 'get_web3') as mock_get_web3:
            mock_get_web3.return_value = mock_w3
            
            result = await self.provider.is_connected("ethereum")
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_block_number_success(self):
        """Test getting block number successfully."""
        mock_w3 = AsyncMock()
        
        # Set up the mock to return 12345 when awaited
        async def mock_block_number():
            return 12345
        mock_w3.eth.block_number = mock_block_number()
        
        with patch.object(self.provider, 'get_web3') as mock_get_web3:
            mock_get_web3.return_value = mock_w3
            
            result = await self.provider.get_block_number("ethereum")
            assert result == 12345
    
    @pytest.mark.asyncio
    async def test_get_block_number_no_web3(self):
        """Test getting block number when no Web3 instance."""
        with patch.object(self.provider, 'get_web3') as mock_get_web3:
            mock_get_web3.return_value = None
            
            result = await self.provider.get_block_number("ethereum")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_get_gas_price_success(self):
        """Test getting gas price successfully."""
        mock_w3 = AsyncMock()
        
        # Set up the mock to return 20 Gwei when awaited
        async def mock_gas_price():
            return 20000000000
        mock_w3.eth.gas_price = mock_gas_price()
        
        with patch.object(self.provider, 'get_web3') as mock_get_web3:
            mock_get_web3.return_value = mock_w3
            
            result = await self.provider.get_gas_price("ethereum")
            assert result == 20000000000
    
    @pytest.mark.asyncio
    async def test_get_balance_success(self):
        """Test getting balance successfully."""
        mock_w3 = AsyncMock()
        mock_w3.eth.get_balance.return_value = 1000000000000000000  # 1 ETH
        
        with patch.object(self.provider, 'get_web3') as mock_get_web3:
            mock_get_web3.return_value = mock_w3
            
            result = await self.provider.get_balance("ethereum", "0x123...")
            assert result == 1000000000000000000
            mock_w3.eth.get_balance.assert_called_once_with("0x123...")
    
    @pytest.mark.asyncio
    async def test_get_chain_health_not_configured(self):
        """Test getting health for non-configured chain."""
        with patch.object(self.provider, 'get_web3') as mock_get_web3, \
             patch.object(self.provider, 'get_chain_config') as mock_get_config:
            mock_get_web3.return_value = None
            mock_get_config.return_value = None
            
            result = await self.provider.get_chain_health("unknown")
            
            assert result["chain"] == "unknown"
            assert result["status"] == "not_configured"
            assert result["connected"] is False
    
    @pytest.mark.asyncio
    async def test_get_chain_health_connected(self):
        """Test getting health for connected chain."""
        mock_w3 = AsyncMock()
        mock_w3.is_connected.return_value = True
        
        mock_config = ChainConfig("Ethereum", 1, "https://test.com")
        
        with patch.object(self.provider, 'get_web3') as mock_get_web3, \
             patch.object(self.provider, 'get_chain_config') as mock_get_config, \
             patch.object(self.provider, 'get_block_number') as mock_block, \
             patch.object(self.provider, 'get_gas_price') as mock_gas:
            
            mock_get_web3.return_value = mock_w3
            mock_get_config.return_value = mock_config
            mock_block.return_value = 12345
            mock_gas.return_value = 20000000000
            
            result = await self.provider.get_chain_health("ethereum")
            
            assert result["chain"] == "ethereum"
            assert result["name"] == "Ethereum"
            assert result["chain_id"] == 1
            assert result["status"] == "healthy"
            assert result["connected"] is True
            assert result["block_number"] == 12345
            assert result["gas_price"] == 20000000000
            assert result["native_currency"] == "ETH"
    
    @pytest.mark.asyncio
    async def test_close_provider(self):
        """Test closing provider cleans up properly."""
        # Set up some mock instances
        mock_w3_1 = AsyncMock()
        mock_w3_2 = AsyncMock()
        mock_provider_1 = AsyncMock()
        mock_provider_2 = AsyncMock()
        
        mock_w3_1.provider = mock_provider_1
        mock_w3_2.provider = mock_provider_2
        
        self.provider.web3_instances = {
            "ethereum": mock_w3_1,
            "arbitrum": mock_w3_2
        }
        self.provider._initialized = True
        
        await self.provider.close()
        
        assert len(self.provider.web3_instances) == 0
        assert self.provider._initialized is False


def test_blockchain_provider_imports():
    """Test that blockchain provider can be imported correctly."""
    from yield_arbitrage.blockchain_connector import BlockchainProvider
    
    provider = BlockchainProvider()
    assert provider is not None
    assert hasattr(provider, 'initialize')
    assert hasattr(provider, 'get_web3')
    assert hasattr(provider, 'get_supported_chains')


if __name__ == "__main__":
    pytest.main([__file__])