"""Unit tests for Uniswap V3 state updater system."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
from decimal import Decimal

from yield_arbitrage.protocols.uniswap_v3_state_updater import (
    UniswapV3StateUpdater,
    StateUpdateConfig,
    PoolStateSnapshot
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeState, EdgeType


class TestStateUpdateConfig:
    """Test StateUpdateConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StateUpdateConfig()
        
        assert config.max_concurrent_updates == 20
        assert config.update_timeout_seconds == 30
        assert config.price_staleness_threshold_seconds == 300
        assert config.enable_price_impact_calculation is True
        assert config.enable_volume_tracking is True
        assert config.cache_pool_states is True
        assert config.cache_ttl_seconds == 60
        assert config.retry_failed_updates is True
        assert config.max_retries == 2
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = StateUpdateConfig(
            max_concurrent_updates=10,
            update_timeout_seconds=60,
            enable_price_impact_calculation=False,
            cache_ttl_seconds=120
        )
        
        assert config.max_concurrent_updates == 10
        assert config.update_timeout_seconds == 60
        assert config.enable_price_impact_calculation is False
        assert config.cache_ttl_seconds == 120


class TestPoolStateSnapshot:
    """Test PoolStateSnapshot dataclass."""
    
    def test_pool_state_creation(self):
        """Test creating PoolStateSnapshot instance."""
        timestamp = datetime.now(timezone.utc)
        
        snapshot = PoolStateSnapshot(
            pool_address="0xpool123",
            block_number=18500000,
            timestamp=timestamp,
            liquidity=5000000000,
            sqrt_price_x96=79228162514264337593543950336,
            tick=0,
            fee_growth_global_0_x128=1000000000000000000,
            fee_growth_global_1_x128=2000000000000000000,
            protocol_fees_token0=500000,
            protocol_fees_token1=750000,
            token0_balance=1000000000000000000,  # 1 token with 18 decimals
            token1_balance=2000000000,  # 2000 tokens with 6 decimals
            price_token0_per_token1=0.0005,  # 1 token0 = 0.0005 token1
            price_token1_per_token0=2000.0,   # 1 token1 = 2000 token0
            tvl_usd=5000000.0,
            volume_24h_usd=1000000.0,
            price_impact_1_percent=0.001,
            price_impact_5_percent=0.005,
            effective_liquidity=4500000.0
        )
        
        assert snapshot.pool_address == "0xpool123"
        assert snapshot.block_number == 18500000
        assert snapshot.timestamp == timestamp
        assert snapshot.liquidity == 5000000000
        assert snapshot.tvl_usd == 5000000.0
        assert snapshot.price_impact_1_percent == 0.001


class TestUniswapV3StateUpdater:
    """Test UniswapV3StateUpdater class."""
    
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
        mock_web3.eth.get_block_number = AsyncMock(return_value=18500000)
        
        provider.get_web3.return_value = mock_web3
        return provider
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return StateUpdateConfig(
            max_concurrent_updates=5,
            update_timeout_seconds=15,
            enable_price_impact_calculation=True,
            cache_ttl_seconds=30
        )
    
    @pytest.fixture
    def state_updater(self, mock_provider, config):
        """Create state updater instance."""
        return UniswapV3StateUpdater(mock_provider, "ethereum", config)
    
    def test_initialization(self, state_updater):
        """Test state updater initialization."""
        assert state_updater.chain_name == "ethereum"
        assert state_updater.config.max_concurrent_updates == 5
        assert len(state_updater.pool_state_cache) == 0
        assert len(state_updater.price_cache) == 0
        assert state_updater.update_stats["updates_performed"] == 0
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, state_updater, mock_provider):
        """Test successful initialization."""
        mock_web3 = await mock_provider.get_web3("ethereum")
        mock_quoter = Mock()
        mock_web3.eth.contract.return_value = mock_quoter
        
        result = await state_updater.initialize("0xquoter123")
        
        assert result is True
        assert state_updater.quoter_contract is mock_quoter
        assert state_updater.web3 is mock_web3
    
    @pytest.mark.asyncio
    async def test_initialize_no_web3(self, mock_provider, config):
        """Test initialization when Web3 is unavailable."""
        mock_provider.get_web3.return_value = None
        state_updater = UniswapV3StateUpdater(mock_provider, "ethereum", config)
        
        result = await state_updater.initialize("0xquoter123")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_update_edge_state_success(self, state_updater, mock_provider):
        """Test successful edge state update."""
        await state_updater.initialize("0xquoter123")
        
        # Create mock edge and metadata
        edge = YieldGraphEdge(
            edge_id="test_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="asset_a",
            target_asset_id="asset_b",
            protocol_name="uniswapv3",
            chain_name="ethereum",
            state=EdgeState(
                conversion_rate=1.0,
                liquidity_usd=100000.0,
                gas_cost_usd=15.0,
                last_updated_timestamp=datetime.now(timezone.utc).timestamp(),
                confidence_score=0.9
            )
        )
        
        metadata = {
            "pool_address": "0xpool123",
            "token0_address": "0xtoken0",
            "token1_address": "0xtoken1",
            "fee_tier": 3000,
            "token0_decimals": 18,
            "token1_decimals": 6,
            "token0_symbol": "WETH",
            "token1_symbol": "USDC"
        }
        
        # Mock pool state snapshot
        timestamp = datetime.now(timezone.utc)
        mock_snapshot = PoolStateSnapshot(
            pool_address="0xpool123",
            block_number=18500000,
            timestamp=timestamp,
            liquidity=5000000000,
            sqrt_price_x96=79228162514264337593543950336,
            tick=0,
            fee_growth_global_0_x128=1000000000000000000,
            fee_growth_global_1_x128=2000000000000000000,
            protocol_fees_token0=500000,
            protocol_fees_token1=750000,
            token0_balance=1000000000000000000,
            token1_balance=2000000000,
            price_token0_per_token1=0.0005,
            price_token1_per_token0=2000.0,
            tvl_usd=5000000.0
        )
        
        state_updater._get_pool_state_snapshot = AsyncMock(return_value=mock_snapshot)
        state_updater._calculate_conversion_rates = AsyncMock(return_value={"base_rate": 0.0005})
        state_updater._calculate_price_impacts = AsyncMock(return_value={"impact_1pct": 0.001})
        state_updater._get_volume_data = AsyncMock(return_value={"volume_24h_usd": 1000000})
        state_updater._calculate_delta_exposure = AsyncMock(return_value={"WETH": 0.6, "USDC": 0.4})
        
        # Update edge state
        updated_state = await state_updater.update_edge_state(edge, metadata)
        
        assert isinstance(updated_state, EdgeState)
        assert updated_state.conversion_rate == 0.0005
        assert updated_state.liquidity_usd == 5000000.0
        assert updated_state.confidence_score > 0.8
        assert state_updater.update_stats["updates_performed"] == 1
    
    @pytest.mark.asyncio
    async def test_update_edge_state_failure(self, state_updater, mock_provider):
        """Test edge state update failure handling."""
        await state_updater.initialize("0xquoter123")
        
        edge = YieldGraphEdge(
            edge_id="test_edge",
            edge_type=EdgeType.TRADE,
            source_asset_id="asset_a",
            target_asset_id="asset_b",
            protocol_name="uniswapv3",
            chain_name="ethereum",
            state=EdgeState(
                conversion_rate=1.0,
                liquidity_usd=100000.0,
                gas_cost_usd=15.0,
                last_updated_timestamp=datetime.now(timezone.utc).timestamp(),
                confidence_score=0.9
            )
        )
        
        metadata = {
            "pool_address": "0xpool123",
            "token0_address": "0xtoken0",
            "token1_address": "0xtoken1",
            "fee_tier": 3000
        }
        
        # Mock failure in pool state snapshot
        state_updater._get_pool_state_snapshot = AsyncMock(return_value=None)
        
        # Update edge state (should return degraded state)
        updated_state = await state_updater.update_edge_state(edge, metadata)
        
        assert isinstance(updated_state, EdgeState)
        # Check that it's a degraded state (confidence_score should be reduced)
        assert updated_state.confidence_score <= 0.5  # Should create degraded state
        assert state_updater.update_stats["updates_failed"] == 1
    
    @pytest.mark.asyncio
    async def test_calculate_conversion_rates(self, state_updater, mock_provider):
        """Test conversion rate calculation."""
        await state_updater.initialize("0xquoter123")
        
        # Mock quoter contract responses
        mock_quoter = state_updater.quoter_contract
        mock_quoter.functions.quoteExactInputSingle.return_value.call = AsyncMock(
            side_effect=[
                1500000,      # 1 unit -> 1.5 USDC
                150000000,    # 100 units -> 150 USDC  
                1500000000,   # 1000 units -> 1500 USDC
                15000000000,  # 10K units -> 15K USDC
                150000000000  # 100K units -> 150K USDC
            ]
        )
        
        metadata = {
            "token0_decimals": 18,
            "token1_decimals": 6
        }
        
        rates = await state_updater._calculate_conversion_rates(
            "0xtoken0", "0xtoken1", 3000, metadata
        )
        
        assert "base_rate" in rates
        assert "rate_1" in rates
        assert "rate_100" in rates
        assert rates["base_rate"] == 1.5  # 1.5 USDC per token (normalized)
    
    @pytest.mark.asyncio
    async def test_calculate_price_impacts(self, state_updater, mock_provider):
        """Test price impact calculation."""
        await state_updater.initialize("0xquoter123")
        
        # Mock pool contract for slot0
        mock_pool = Mock()
        mock_pool.functions.slot0.return_value.call = AsyncMock(
            return_value=[79228162514264337593543950336, 0, 0, 0, 0, 0, True]  # slot0 data
        )
        state_updater.web3.eth.contract.return_value = mock_pool
        
        # Mock quoter responses for different trade sizes
        mock_quoter = state_updater.quoter_contract
        mock_quoter.functions.quoteExactInputSingle.return_value.call = AsyncMock(
            side_effect=[
                10000000,    # 1% trade
                50000000,    # 5% trade
                100000000    # 10% trade
            ]
        )
        
        metadata = {
            "token0_decimals": 18,
            "token1_decimals": 6
        }
        
        impacts = await state_updater._calculate_price_impacts(
            "0xpool123", "0xtoken0", "0xtoken1", 3000, metadata
        )
        
        assert "impact_1pct" in impacts
        assert "impact_5pct" in impacts
        assert "impact_10pct" in impacts
    
    @pytest.mark.asyncio
    async def test_batch_update_edges(self, state_updater, mock_provider):
        """Test batch edge state updates."""
        await state_updater.initialize("0xquoter123")
        
        # Create multiple edges
        edges_with_metadata = []
        for i in range(3):
            edge = YieldGraphEdge(
                edge_id=f"test_edge_{i}",
                edge_type=EdgeType.TRADE,
                source_asset_id=f"asset_a_{i}",
                target_asset_id=f"asset_b_{i}",
                protocol_name="uniswapv3",
                chain_name="ethereum",
                state=EdgeState(
                    conversion_rate=1.0,
                    liquidity_usd=100000.0,
                    gas_cost_usd=15.0,
                    last_updated_timestamp=datetime.now(timezone.utc).timestamp(),
                    confidence_score=0.9
                )
            )
            
            metadata = {
                "pool_address": f"0xpool{i}",
                "token0_address": f"0xtoken0_{i}",
                "token1_address": f"0xtoken1_{i}",
                "fee_tier": 3000
            }
            
            edges_with_metadata.append((edge, metadata))
        
        # Mock successful update for all edges
        state_updater.update_edge_state = AsyncMock(
            return_value=EdgeState(
                conversion_rate=0.0005,
                liquidity_usd=5000000.0,
                gas_cost_usd=15.0,
                last_updated_timestamp=datetime.now(timezone.utc).timestamp(),
                confidence_score=0.95
            )
        )
        
        # Batch update
        results = await state_updater.batch_update_edges(edges_with_metadata)
        
        assert len(results) == 3
        assert all(isinstance(state, EdgeState) for state in results.values())
    
    def test_get_update_stats(self, state_updater):
        """Test getting update statistics."""
        # Set some stats
        state_updater.update_stats.update({
            "updates_performed": 15,
            "updates_failed": 2,
            "cache_hits": 8,
            "avg_update_time_ms": 250.5
        })
        
        # Add some cache entries
        state_updater.pool_state_cache["0xpool1"] = Mock()
        state_updater.pool_state_cache["0xpool2"] = Mock()
        
        stats = state_updater.get_update_stats()
        
        assert stats["updates_performed"] == 15
        assert stats["updates_failed"] == 2
        assert stats["cache_size"] == 2
        assert stats["cache_hit_rate"] == (8 / 15) * 100
        assert stats["success_rate"] == (15 / 17) * 100
        assert "config" in stats
    
    def test_cache_cleanup(self, state_updater):
        """Test cache cleanup functionality."""
        from datetime import timedelta
        
        # Add expired cache entry (way in the past)
        old_timestamp = datetime.now(timezone.utc) - timedelta(seconds=state_updater.config.cache_ttl_seconds * 10)
        expired_snapshot = PoolStateSnapshot(
            pool_address="0xexpired",
            block_number=18000000,
            timestamp=old_timestamp,
            liquidity=1000000,
            sqrt_price_x96=79228162514264337593543950336,
            tick=0,
            fee_growth_global_0_x128=0,
            fee_growth_global_1_x128=0,
            protocol_fees_token0=0,
            protocol_fees_token1=0,
            token0_balance=1000000,
            token1_balance=2000000,
            price_token0_per_token1=0.5,
            price_token1_per_token0=2.0
        )
        
        # Add fresh cache entry
        fresh_timestamp = datetime.now(timezone.utc)
        fresh_snapshot = PoolStateSnapshot(
            pool_address="0xfresh",
            block_number=18500000,
            timestamp=fresh_timestamp,
            liquidity=5000000,
            sqrt_price_x96=79228162514264337593543950336,
            tick=0,
            fee_growth_global_0_x128=0,
            fee_growth_global_1_x128=0,
            protocol_fees_token0=0,
            protocol_fees_token1=0,
            token0_balance=5000000,
            token1_balance=10000000,
            price_token0_per_token1=0.5,
            price_token1_per_token0=2.0
        )
        
        state_updater.pool_state_cache["0xexpired"] = expired_snapshot
        state_updater.pool_state_cache["0xfresh"] = fresh_snapshot
        
        assert len(state_updater.pool_state_cache) == 2
        
        # Force cache cleanup by setting last cleanup time to way in the past
        state_updater.last_cache_cleanup = old_timestamp
        
        # Cleanup cache
        state_updater.cleanup_cache()
        
        # Only fresh entry should remain
        assert len(state_updater.pool_state_cache) == 1
        assert "0xfresh" in state_updater.pool_state_cache
        assert "0xexpired" not in state_updater.pool_state_cache
    
    def test_confidence_score_calculation(self, state_updater):
        """Test confidence score calculation."""
        # Fresh, high-liquidity pool
        fresh_snapshot = PoolStateSnapshot(
            pool_address="0xpool",
            block_number=18500000,
            timestamp=datetime.now(timezone.utc),
            liquidity=5000000,  # High liquidity
            sqrt_price_x96=79228162514264337593543950336,
            tick=0,
            fee_growth_global_0_x128=0,
            fee_growth_global_1_x128=0,
            protocol_fees_token0=0,
            protocol_fees_token1=0,
            token0_balance=5000000,
            token1_balance=10000000,
            price_token0_per_token1=0.5,
            price_token1_per_token0=2.0
        )
        
        rates = {"base_rate": 0.0005}
        impacts = {"impact_1pct": 0.001}
        
        score = state_updater._calculate_confidence_score(fresh_snapshot, rates, impacts)
        
        # Should be high confidence for fresh, high-liquidity pool with valid rate
        assert score > 0.9
        
        # Test low liquidity
        fresh_snapshot.liquidity = 500  # Low liquidity
        score_low_liq = state_updater._calculate_confidence_score(fresh_snapshot, rates, impacts)
        assert score_low_liq < score
        
        # Test missing rate
        rates_no_base = {"rate_1": 0.0005}
        score_no_rate = state_updater._calculate_confidence_score(fresh_snapshot, rates_no_base, impacts)
        assert score_no_rate < 0.4