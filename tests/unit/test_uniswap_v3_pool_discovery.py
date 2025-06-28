"""Unit tests for Uniswap V3 pool discovery system."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from yield_arbitrage.protocols.uniswap_v3_pool_discovery import (
    UniswapV3PoolDiscovery,
    PoolDiscoveryConfig,
    PoolInfo
)


class TestPoolDiscoveryConfig:
    """Test PoolDiscoveryConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PoolDiscoveryConfig()
        
        assert config.max_pools_per_batch == 50
        assert config.discovery_timeout_seconds == 300
        assert config.min_liquidity_threshold == 10000.0
        assert config.max_gas_price_gwei == 50
        assert config.enable_event_scanning is True
        assert config.event_scan_blocks == 10000
        assert config.retry_failed_pools is True
        assert config.max_retries == 3
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PoolDiscoveryConfig(
            max_pools_per_batch=25,
            min_liquidity_threshold=50000.0,
            enable_event_scanning=False
        )
        
        assert config.max_pools_per_batch == 25
        assert config.min_liquidity_threshold == 50000.0
        assert config.enable_event_scanning is False


class TestPoolInfo:
    """Test PoolInfo dataclass."""
    
    def test_pool_info_creation(self):
        """Test creating PoolInfo instance."""
        created_time = datetime.now(timezone.utc)
        
        pool = PoolInfo(
            pool_address="0xpool123",
            token0_address="0xtoken0",
            token1_address="0xtoken1",
            token0_symbol="WETH",
            token1_symbol="USDC",
            token0_decimals=18,
            token1_decimals=6,
            fee_tier=3000,
            liquidity=1000000,
            sqrt_price_x96=79228162514264337593543950336,
            tick=0,
            tick_spacing=60,
            protocol_fee=0,
            tvl_usd=2000000.0,
            volume_24h_usd=500000.0,
            created_block=18000000,
            created_timestamp=created_time,
            is_active=True
        )
        
        assert pool.pool_address == "0xpool123"
        assert pool.token0_symbol == "WETH"
        assert pool.token1_symbol == "USDC"
        assert pool.fee_tier == 3000
        assert pool.tvl_usd == 2000000.0
        assert pool.is_active is True
        assert pool.created_timestamp == created_time


class TestUniswapV3PoolDiscovery:
    """Test UniswapV3PoolDiscovery class."""
    
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
        mock_web3.eth.get_block_number = AsyncMock(return_value=18000000)
        # Mock gas_price as an awaitable
        async def mock_gas_price():
            return 20000000000  # 20 gwei
        mock_web3.eth.gas_price = mock_gas_price()
        
        provider.get_web3.return_value = mock_web3
        return provider
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return PoolDiscoveryConfig(
            max_pools_per_batch=10,
            min_liquidity_threshold=5000.0,
            max_gas_price_gwei=30
        )
    
    @pytest.fixture
    def discovery(self, mock_provider, config):
        """Create pool discovery instance."""
        return UniswapV3PoolDiscovery(mock_provider, "ethereum", config)
    
    def test_initialization(self, discovery):
        """Test discovery system initialization."""
        assert discovery.chain_name == "ethereum"
        assert discovery.config.max_pools_per_batch == 10
        assert len(discovery.discovered_pools) == 0
        assert len(discovery.failed_pools) == 0
        assert discovery.discovery_stats["pools_discovered"] == 0
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, discovery, mock_provider):
        """Test successful initialization."""
        mock_web3 = await mock_provider.get_web3("ethereum")
        mock_factory = Mock()
        mock_web3.eth.contract.return_value = mock_factory
        
        result = await discovery.initialize()
        
        assert result is True
        assert discovery.factory_contract is mock_factory
        assert discovery.web3 is mock_web3
    
    @pytest.mark.asyncio
    async def test_initialize_no_contracts(self, mock_provider, config):
        """Test initialization with unsupported chain."""
        discovery = UniswapV3PoolDiscovery(mock_provider, "unsupported_chain", config)
        
        result = await discovery.initialize()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_initialize_no_web3(self, mock_provider, config):
        """Test initialization when Web3 is unavailable."""
        mock_provider.get_web3.return_value = None
        discovery = UniswapV3PoolDiscovery(mock_provider, "ethereum", config)
        
        result = await discovery.initialize()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_gas_conditions_success(self, discovery, mock_provider):
        """Test gas condition check when conditions are favorable."""
        await discovery.initialize()
        
        result = await discovery._check_gas_conditions()
        
        assert result is True  # 20 gwei < 30 gwei limit
    
    @pytest.mark.asyncio
    async def test_check_gas_conditions_high_gas(self, discovery, mock_provider):
        """Test gas condition check when gas is too high."""
        await discovery.initialize()
        
        # Set high gas price (50 gwei > 30 gwei limit)
        # Create a proper async property mock
        async def mock_gas_price():
            return 50000000000
        
        discovery.web3.eth.gas_price = mock_gas_price()
        
        result = await discovery._check_gas_conditions()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_token_info_success(self, discovery, mock_provider):
        """Test successful token info retrieval."""
        await discovery.initialize()
        
        # Mock token contract
        mock_token = Mock()
        mock_token.functions.symbol.return_value.call = AsyncMock(return_value="WETH")
        mock_token.functions.decimals.return_value.call = AsyncMock(return_value=18)
        discovery.web3.eth.contract.return_value = mock_token
        
        token_info = await discovery._get_token_info("0xtoken")
        
        assert token_info["symbol"] == "WETH"
        assert token_info["decimals"] == 18
    
    @pytest.mark.asyncio
    async def test_get_token_info_failure(self, discovery, mock_provider):
        """Test token info retrieval failure."""
        await discovery.initialize()
        
        # Mock token contract that fails
        mock_token = Mock()
        mock_token.functions.symbol.return_value.call = AsyncMock(
            side_effect=Exception("Contract call failed")
        )
        discovery.web3.eth.contract.return_value = mock_token
        
        token_info = await discovery._get_token_info("0xtoken")
        
        assert token_info["symbol"] == "UNKNOWN"
        assert token_info["decimals"] == 18
    
    @pytest.mark.asyncio
    async def test_estimate_pool_tvl(self, discovery, mock_provider):
        """Test pool TVL estimation."""
        await discovery.initialize()
        
        # Mock token contracts for balance calls
        mock_token0 = Mock()
        mock_token0.functions.balanceOf.return_value.call = AsyncMock(return_value=1000000000000000000)  # 1 token with 18 decimals
        
        mock_token1 = Mock()
        mock_token1.functions.balanceOf.return_value.call = AsyncMock(return_value=1500000000)  # 1500 tokens with 6 decimals
        
        discovery.web3.eth.contract.side_effect = [mock_token0, mock_token1]
        
        tvl = await discovery._estimate_pool_tvl(
            "0xpool", "0xtoken0", "0xtoken1", 18, 6, 1000000, 79228162514264337593543950336
        )
        
        assert tvl is not None
        assert tvl > 0
    
    @pytest.mark.asyncio
    async def test_discover_single_pool_success(self, discovery, mock_provider):
        """Test successful single pool discovery."""
        await discovery.initialize()
        
        # Mock factory contract
        discovery.factory_contract.functions.getPool.return_value.call = AsyncMock(
            return_value="0xpool123"
        )
        
        # Mock _get_detailed_pool_info
        mock_pool_info = PoolInfo(
            pool_address="0xpool123",
            token0_address="0xtoken0",
            token1_address="0xtoken1",
            token0_symbol="WETH",
            token1_symbol="USDC",
            token0_decimals=18,
            token1_decimals=6,
            fee_tier=3000,
            liquidity=1000000,
            sqrt_price_x96=79228162514264337593543950336,
            tick=0,
            tick_spacing=60,
            protocol_fee=0,
            tvl_usd=50000.0,
            is_active=True
        )
        
        discovery._get_detailed_pool_info = AsyncMock(return_value=mock_pool_info)
        
        result = await discovery._discover_single_pool("0xtoken0", "0xtoken1", 3000)
        
        assert result is not None
        assert result.pool_address == "0xpool123"
        assert result.fee_tier == 3000
        # Note: _discover_single_pool doesn't add to discovered_pools - that's done by higher-level methods
    
    @pytest.mark.asyncio
    async def test_discover_single_pool_no_pool(self, discovery, mock_provider):
        """Test single pool discovery when pool doesn't exist."""
        await discovery.initialize()
        
        # Mock factory returning zero address
        discovery.factory_contract.functions.getPool.return_value.call = AsyncMock(
            return_value="0x0000000000000000000000000000000000000000"
        )
        
        result = await discovery._discover_single_pool("0xtoken0", "0xtoken1", 3000)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_discover_pools_by_tokens(self, discovery, mock_provider):
        """Test discovering pools for token list."""
        await discovery.initialize()
        
        # Mock successful discovery for one pool
        mock_pool_info = PoolInfo(
            pool_address="0xpool123",
            token0_address="0xtoken0",
            token1_address="0xtoken1",
            token0_symbol="WETH",
            token1_symbol="USDC",
            token0_decimals=18,
            token1_decimals=6,
            fee_tier=3000,
            liquidity=1000000,
            sqrt_price_x96=79228162514264337593543950336,
            tick=0,
            tick_spacing=60,
            protocol_fee=0,
            tvl_usd=50000.0,
            is_active=True
        )
        
        discovery._discover_single_pool = AsyncMock(return_value=mock_pool_info)
        
        # Test with two tokens
        token_addresses = ["0xtoken0", "0xtoken1"]
        pools = await discovery.discover_pools_by_tokens(token_addresses)
        
        # Should discover 4 pools (1 pair × 4 fee tiers)
        assert len(pools) == 4
        assert all(pool.pool_address == "0xpool123" for pool in pools)
        assert discovery.discovery_stats["pools_discovered"] == 4
    
    @pytest.mark.asyncio
    async def test_discover_pools_high_gas(self, discovery, mock_provider):
        """Test pool discovery with high gas prices."""
        await discovery.initialize()
        
        # Set high gas price
        discovery.web3.eth.gas_price = 60000000000  # 60 gwei > 30 gwei limit
        
        pools = await discovery.discover_pools_by_tokens(["0xtoken0", "0xtoken1"])
        
        assert len(pools) == 0  # Should skip discovery due to high gas
    
    @pytest.mark.asyncio
    async def test_discover_pools_by_events(self, discovery, mock_provider):
        """Test event-based pool discovery."""
        await discovery.initialize()
        
        # Mock PoolCreated events
        mock_event = Mock()
        mock_event.args = Mock()
        mock_event.args.pool = "0xpool123"
        mock_event.args.token0 = "0xtoken0"
        mock_event.args.token1 = "0xtoken1"
        mock_event.args.fee = 3000
        mock_event.blockNumber = 18000000
        
        mock_filter = Mock()
        mock_filter.get_all_entries = AsyncMock(return_value=[mock_event])
        
        discovery.factory_contract.events.PoolCreated.create_filter.return_value = mock_filter
        
        # Mock block data
        mock_block = Mock()
        mock_block.timestamp = 1700000000
        discovery.web3.eth.get_block = AsyncMock(return_value=mock_block)
        
        # Mock pool info processing
        mock_pool_info = PoolInfo(
            pool_address="0xpool123",
            token0_address="0xtoken0",
            token1_address="0xtoken1",
            token0_symbol="WETH",
            token1_symbol="USDC",
            token0_decimals=18,
            token1_decimals=6,
            fee_tier=3000,
            liquidity=1000000,
            sqrt_price_x96=79228162514264337593543950336,
            tick=0,
            tick_spacing=60,
            protocol_fee=0,
            tvl_usd=50000.0,
            is_active=True,
            created_block=18000000,
            created_timestamp=datetime.fromtimestamp(1700000000, tz=timezone.utc)
        )
        
        discovery._get_detailed_pool_info = AsyncMock(return_value=mock_pool_info)
        
        pools = await discovery.discover_pools_by_events(from_block=17990000)
        
        assert len(pools) == 1
        assert pools[0].pool_address == "0xpool123"
        assert pools[0].created_block == 18000000
    
    @pytest.mark.asyncio
    async def test_refresh_pool_data(self, discovery, mock_provider):
        """Test refreshing existing pool data."""
        await discovery.initialize()
        
        # Add a pool to discovered pools
        original_pool = PoolInfo(
            pool_address="0xpool123",
            token0_address="0xtoken0",
            token1_address="0xtoken1",
            token0_symbol="WETH",
            token1_symbol="USDC",
            token0_decimals=18,
            token1_decimals=6,
            fee_tier=3000,
            liquidity=1000000,
            sqrt_price_x96=79228162514264337593543950336,
            tick=0,
            tick_spacing=60,
            protocol_fee=0,
            tvl_usd=50000.0,
            is_active=True,
            created_block=18000000
        )
        
        discovery.discovered_pools["0xpool123"] = original_pool
        
        # Mock refreshed pool data
        refreshed_pool = PoolInfo(
            pool_address="0xpool123",
            token0_address="0xtoken0",
            token1_address="0xtoken1",
            token0_symbol="WETH",
            token1_symbol="USDC",
            token0_decimals=18,
            token1_decimals=6,
            fee_tier=3000,
            liquidity=2000000,  # Updated liquidity
            sqrt_price_x96=79228162514264337593543950336,
            tick=0,
            tick_spacing=60,
            protocol_fee=0,
            tvl_usd=100000.0,  # Updated TVL
            is_active=True,
            created_block=18000000  # Preserved from original
        )
        
        discovery._refresh_single_pool = AsyncMock(return_value=refreshed_pool)
        
        refreshed_pools = await discovery.refresh_pool_data(["0xpool123"])
        
        assert len(refreshed_pools) == 1
        assert refreshed_pools["0xpool123"].liquidity == 2000000
        assert refreshed_pools["0xpool123"].tvl_usd == 100000.0
        assert refreshed_pools["0xpool123"].created_block == 18000000  # Preserved
    
    def test_get_discovery_stats(self, discovery):
        """Test getting discovery statistics."""
        # Set some stats
        discovery.discovery_stats.update({
            "pools_discovered": 10,
            "pools_failed": 2,
            "blocks_scanned": 5000,
            "discovery_time": 45.5
        })
        
        discovery.discovered_pools["0xpool1"] = Mock()
        discovery.discovered_pools["0xpool2"] = Mock()
        discovery.failed_pools.add("0xfailed1")
        
        stats = discovery.get_discovery_stats()
        
        assert stats["pools_discovered"] == 10
        assert stats["pools_failed"] == 2
        assert stats["total_pools_cached"] == 2
        assert stats["failed_pools"] == 1
        assert stats["config"]["max_pools_per_batch"] == 10
    
    def test_get_pools_by_token(self, discovery):
        """Test getting pools by token address."""
        # Add some pools
        pool1 = PoolInfo(
            pool_address="0xpool1",
            token0_address="0xweth",
            token1_address="0xusdc",
            token0_symbol="WETH",
            token1_symbol="USDC",
            token0_decimals=18,
            token1_decimals=6,
            fee_tier=3000,
            liquidity=1000000,
            sqrt_price_x96=79228162514264337593543950336,
            tick=0,
            tick_spacing=60,
            protocol_fee=0
        )
        
        pool2 = PoolInfo(
            pool_address="0xpool2",
            token0_address="0xusdc",
            token1_address="0xusdt",
            token0_symbol="USDC",
            token1_symbol="USDT",
            token0_decimals=6,
            token1_decimals=6,
            fee_tier=500,
            liquidity=2000000,
            sqrt_price_x96=79228162514264337593543950336,
            tick=0,
            tick_spacing=10,
            protocol_fee=0
        )
        
        discovery.discovered_pools["0xpool1"] = pool1
        discovery.discovered_pools["0xpool2"] = pool2
        
        # Get pools containing USDC
        usdc_pools = discovery.get_pools_by_token("0xusdc")
        
        assert len(usdc_pools) == 2
        assert any(pool.pool_address == "0xpool1" for pool in usdc_pools)
        assert any(pool.pool_address == "0xpool2" for pool in usdc_pools)
        
        # Get pools containing WETH
        weth_pools = discovery.get_pools_by_token("0xweth")
        
        assert len(weth_pools) == 1
        assert weth_pools[0].pool_address == "0xpool1"
    
    def test_get_pools_by_fee_tier(self, discovery):
        """Test getting pools by fee tier."""
        # Use same pools as above
        pool1 = PoolInfo(
            pool_address="0xpool1", token0_address="0xweth", token1_address="0xusdc",
            token0_symbol="WETH", token1_symbol="USDC", token0_decimals=18, token1_decimals=6,
            fee_tier=3000, liquidity=1000000, sqrt_price_x96=79228162514264337593543950336,
            tick=0, tick_spacing=60, protocol_fee=0
        )
        
        pool2 = PoolInfo(
            pool_address="0xpool2", token0_address="0xusdc", token1_address="0xusdt",
            token0_symbol="USDC", token1_symbol="USDT", token0_decimals=6, token1_decimals=6,
            fee_tier=500, liquidity=2000000, sqrt_price_x96=79228162514264337593543950336,
            tick=0, tick_spacing=10, protocol_fee=0
        )
        
        discovery.discovered_pools["0xpool1"] = pool1
        discovery.discovered_pools["0xpool2"] = pool2
        
        # Get 0.3% fee tier pools
        fee_3000_pools = discovery.get_pools_by_fee_tier(3000)
        assert len(fee_3000_pools) == 1
        assert fee_3000_pools[0].pool_address == "0xpool1"
        
        # Get 0.05% fee tier pools
        fee_500_pools = discovery.get_pools_by_fee_tier(500)
        assert len(fee_500_pools) == 1
        assert fee_500_pools[0].pool_address == "0xpool2"
    
    def test_clear_failed_pools(self, discovery):
        """Test clearing failed pools cache."""
        discovery.failed_pools.add("0xfailed1")
        discovery.failed_pools.add("0xfailed2")
        
        assert len(discovery.failed_pools) == 2
        
        discovery.clear_failed_pools()
        
        assert len(discovery.failed_pools) == 0