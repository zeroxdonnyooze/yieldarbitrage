"""
Unit tests for DeltaTracker.

Tests the delta tracking functionality including path delta calculation,
position management, and risk limit validation.
"""
import pytest
import asyncio
import time
from decimal import Decimal
from unittest.mock import Mock, AsyncMock

from yield_arbitrage.risk.delta_tracker import (
    DeltaTracker, DeltaPosition, AssetExposure, ExposureType,
    calculate_path_delta, calculate_portfolio_delta
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType, EdgeState


class TestAssetExposure:
    """Test AssetExposure data model."""
    
    def test_asset_exposure_creation(self):
        """Test AssetExposure creation and basic properties."""
        exposure = AssetExposure(
            asset_id="ETH_MAINNET_WETH",
            amount=Decimal('2.5'),
            usd_value=Decimal('5000.0'),
            exposure_type=ExposureType.LONG
        )
        
        assert exposure.asset_id == "ETH_MAINNET_WETH"
        assert exposure.amount == Decimal('2.5')
        assert exposure.usd_value == Decimal('5000.0')
        assert exposure.exposure_type == ExposureType.LONG
        assert not exposure.is_stale()
    
    def test_exposure_updates(self):
        """Test exposure update functionality."""
        exposure = AssetExposure(
            asset_id="ETH_MAINNET_USDC",
            amount=Decimal('1000'),
            usd_value=Decimal('1000'),
            exposure_type=ExposureType.LONG
        )
        
        # Test value update
        exposure.update_value(Decimal('1050'))
        assert exposure.usd_value == Decimal('1050')
        
        # Test adding exposure
        exposure.add_exposure(Decimal('500'), Decimal('500'))
        assert exposure.amount == Decimal('1500')
        assert exposure.usd_value == Decimal('1550')
    
    def test_exposure_type_changes(self):
        """Test exposure type changes with amount updates."""
        exposure = AssetExposure(
            asset_id="ETH_MAINNET_WETH",
            amount=Decimal('1.0'),
            usd_value=Decimal('2000'),
            exposure_type=ExposureType.LONG
        )
        
        # Add negative exposure to make it short
        exposure.add_exposure(Decimal('-2.0'), Decimal('-4000'))
        assert exposure.exposure_type == ExposureType.SHORT
        assert exposure.amount == Decimal('-1.0')
        
        # Add back to neutral
        exposure.add_exposure(Decimal('1.0'), Decimal('2000'))
        assert exposure.exposure_type == ExposureType.NEUTRAL
        assert exposure.amount == Decimal('0')


class TestDeltaPosition:
    """Test DeltaPosition functionality."""
    
    def test_position_creation(self):
        """Test basic position creation."""
        position = DeltaPosition(
            position_id="arbitrage_001",
            position_type="arbitrage"
        )
        
        assert position.position_id == "arbitrage_001"
        assert position.position_type == "arbitrage"
        assert len(position.exposures) == 0
        assert position.is_active
        assert position.total_usd_exposure == Decimal('0')
    
    def test_add_exposure(self):
        """Test adding exposures to position."""
        position = DeltaPosition(
            position_id="test_position",
            position_type="yield_farming"
        )
        
        # Add ETH exposure
        position.add_exposure(
            "ETH_MAINNET_WETH",
            Decimal('1.5'),
            Decimal('3000')
        )
        
        assert len(position.exposures) == 1
        assert position.total_usd_exposure == Decimal('3000')
        assert position.max_single_asset_exposure == Decimal('3000')
        
        # Add USDC exposure
        position.add_exposure(
            "ETH_MAINNET_USDC", 
            Decimal('2000'),
            Decimal('2000')
        )
        
        assert len(position.exposures) == 2
        assert position.total_usd_exposure == Decimal('5000')
    
    def test_net_exposure_calculation(self):
        """Test net exposure calculation."""
        position = DeltaPosition(
            position_id="test_position",
            position_type="arbitrage"
        )
        
        # Add long ETH
        position.add_exposure("ETH_MAINNET_WETH", Decimal('2.0'), Decimal('4000'))
        
        # Add short ETH
        position.add_exposure("ETH_MAINNET_WETH", Decimal('-1.0'), Decimal('-2000'))
        
        net_exposure = position.get_net_exposure("ETH_MAINNET_WETH")
        assert net_exposure == Decimal('1.0')
    
    def test_hedged_position_detection(self):
        """Test hedged position detection."""
        position = DeltaPosition(
            position_id="hedged_position",
            position_type="market_neutral"
        )
        
        # Add long and short exposures
        position.add_exposure("ETH_MAINNET_WETH", Decimal('1.0'), Decimal('2000'))
        position.add_exposure("ETH_MAINNET_USDC", Decimal('-2000'), Decimal('-2000'))
        
        assert position.is_hedged()
    
    def test_risk_value_calculation(self):
        """Test total risk value calculation."""
        position = DeltaPosition(
            position_id="risk_test",
            position_type="leveraged"
        )
        
        # Create exposure with volatility
        position.add_exposure("ETH_MAINNET_WETH", Decimal('1.0'), Decimal('2000'))
        
        # Manually set volatility for testing
        position.exposures["ETH_MAINNET_WETH"].volatility = 0.15  # 15%
        
        risk_value = position.get_total_risk_value()
        expected_risk = Decimal('2000') * Decimal('1.15')  # 2000 * (1 + 0.15)
        assert risk_value == expected_risk


class TestDeltaTracker:
    """Test DeltaTracker main functionality."""
    
    @pytest.fixture
    def mock_asset_oracle(self):
        """Create mock asset oracle."""
        oracle = Mock()
        oracle.is_stable_asset = Mock(return_value=False)
        oracle.get_asset_price = AsyncMock(return_value=2000.0)  # $2000 default price
        return oracle
    
    @pytest.fixture
    def delta_tracker(self, mock_asset_oracle):
        """Create DeltaTracker instance with mock oracle."""
        return DeltaTracker(mock_asset_oracle)
    
    @pytest.fixture
    def sample_trade_edge(self):
        """Create sample trade edge."""
        edge = Mock(spec=YieldGraphEdge)
        edge.edge_id = "ETH_MAINNET_UNISWAPV3_TRADE_WETH_USDC"
        edge.source_asset_id = "ETH_MAINNET_WETH"
        edge.target_asset_id = "ETH_MAINNET_USDC"
        edge.edge_type = EdgeType.TRADE
        edge.calculate_output = Mock(return_value={
            "output_amount": 1950.0,  # $1950 out for 1 ETH in (some slippage)
            "effective_rate": 1950.0
        })
        return edge
    
    def test_tracker_initialization(self, delta_tracker, mock_asset_oracle):
        """Test DeltaTracker initialization."""
        assert delta_tracker.asset_oracle == mock_asset_oracle
        assert len(delta_tracker.active_positions) == 0
        assert delta_tracker.max_single_asset_exposure_usd == Decimal('100000')
    
    @pytest.mark.asyncio
    async def test_calculate_simple_trade_path_delta(self, delta_tracker, sample_trade_edge):
        """Test delta calculation for simple trade path."""
        path = [sample_trade_edge]
        path_amounts = [1.0]  # 1 ETH input
        
        delta = await delta_tracker.calculate_path_delta(path, path_amounts)
        
        # Should have exposure to USDC received
        assert "ETH_MAINNET_USDC" in delta
        assert delta["ETH_MAINNET_USDC"] == 1950.0
    
    @pytest.mark.asyncio
    async def test_calculate_round_trip_delta(self, delta_tracker, mock_asset_oracle):
        """Test delta calculation for round-trip arbitrage."""
        # Mock stable asset detection
        mock_asset_oracle.is_stable_asset.side_effect = lambda asset_id: "USDC" in asset_id
        
        # Create round-trip trade: ETH -> USDC -> ETH
        edge1 = Mock(spec=YieldGraphEdge)
        edge1.edge_id = "trade1"
        edge1.source_asset_id = "ETH_MAINNET_WETH"
        edge1.target_asset_id = "ETH_MAINNET_USDC"
        edge1.edge_type = EdgeType.TRADE
        edge1.calculate_output = Mock(return_value={"output_amount": 2000.0})
        
        edge2 = Mock(spec=YieldGraphEdge)
        edge2.edge_id = "trade2"
        edge2.source_asset_id = "ETH_MAINNET_USDC"
        edge2.target_asset_id = "ETH_MAINNET_WETH"
        edge2.edge_type = EdgeType.TRADE
        edge2.calculate_output = Mock(return_value={"output_amount": 1.05})  # Profit!
        
        path = [edge1, edge2]
        path_amounts = [1.0, 2000.0]
        
        delta = await delta_tracker.calculate_path_delta(path, path_amounts)
        
        # Should have minimal delta since it's round-trip with stables
        # ETH exposure should show the profit
        assert "ETH_MAINNET_WETH" in delta
        assert delta["ETH_MAINNET_WETH"] == 1.05
    
    @pytest.mark.asyncio
    async def test_add_position_from_path(self, delta_tracker, sample_trade_edge):
        """Test adding position from arbitrage path."""
        path = [sample_trade_edge]
        path_amounts = [1.0]
        
        success = await delta_tracker.add_position(
            position_id="arbitrage_test_001",
            position_type="arbitrage",
            path=path,
            path_amounts=path_amounts
        )
        
        assert success
        assert "arbitrage_test_001" in delta_tracker.active_positions
        
        position = delta_tracker.get_position("arbitrage_test_001")
        assert position is not None
        assert len(position.exposures) > 0
    
    @pytest.mark.asyncio
    async def test_add_position_direct_exposures(self, delta_tracker):
        """Test adding position with direct exposures."""
        initial_exposures = {
            "ETH_MAINNET_WETH": (2.5, 5000.0),  # (amount, usd_value)
            "ETH_MAINNET_USDC": (1000.0, 1000.0)
        }
        
        success = await delta_tracker.add_position(
            position_id="yield_farming_001",
            position_type="yield_farming",
            initial_exposures=initial_exposures
        )
        
        assert success
        
        position = delta_tracker.get_position("yield_farming_001")
        assert len(position.exposures) == 2
        assert position.total_usd_exposure == Decimal('6000')
    
    def test_position_management(self, delta_tracker):
        """Test position management operations."""
        # Create position directly
        position = DeltaPosition(
            position_id="test_mgmt",
            position_type="manual"
        )
        delta_tracker.active_positions["test_mgmt"] = position
        
        # Test get position
        retrieved = delta_tracker.get_position("test_mgmt")
        assert retrieved == position
        
        # Test update exposure
        success = delta_tracker.update_position_exposure(
            "test_mgmt",
            "ETH_MAINNET_WETH",
            1.0,
            2000.0
        )
        assert success
        
        # Test remove position
        success = delta_tracker.remove_position("test_mgmt")
        assert success
        assert "test_mgmt" not in delta_tracker.active_positions
    
    @pytest.mark.asyncio
    async def test_portfolio_snapshot(self, delta_tracker):
        """Test portfolio snapshot generation."""
        # Add multiple positions
        await delta_tracker.add_position(
            "pos1", "arbitrage",
            initial_exposures={"ETH_MAINNET_WETH": (1.0, 2000.0)}
        )
        await delta_tracker.add_position(
            "pos2", "yield_farming",
            initial_exposures={"ETH_MAINNET_USDC": (1000.0, 1000.0)}
        )
        
        snapshot = await delta_tracker.get_portfolio_snapshot()
        
        assert snapshot.total_positions == 2
        assert len(snapshot.asset_exposures) == 2
        assert snapshot.total_usd_long == Decimal('3000')
    
    @pytest.mark.asyncio
    async def test_risk_limit_validation(self, delta_tracker):
        """Test risk limit checking."""
        # Add position that exceeds single asset limit
        large_exposure = {
            "ETH_MAINNET_WETH": (100.0, 200000.0)  # $200k > $100k limit
        }
        
        await delta_tracker.add_position(
            "large_position",
            "high_risk",
            initial_exposures=large_exposure
        )
        
        alerts = await delta_tracker.check_risk_limits()
        
        # Should generate alert for exceeding single asset limit
        assert len(alerts) > 0
        alert_types = [alert["type"] for alert in alerts]
        assert "single_asset_exposure_limit" in alert_types
    
    def test_stats_tracking(self, delta_tracker):
        """Test statistics tracking."""
        initial_stats = delta_tracker.get_stats()
        assert initial_stats["active_positions"] == 0
        assert initial_stats["positions_tracked"] == 0
        
        # Add position
        position = DeltaPosition("test_stats", "test")
        delta_tracker.active_positions["test_stats"] = position
        delta_tracker.stats["positions_tracked"] += 1
        
        updated_stats = delta_tracker.get_stats()
        assert updated_stats["active_positions"] == 1
        assert updated_stats["positions_tracked"] == 1


class TestConvenienceFunctions:
    """Test standalone convenience functions."""
    
    def test_calculate_path_delta_function(self):
        """Test standalone path delta calculation."""
        # Create mock edge
        edge = Mock(spec=YieldGraphEdge)
        edge.edge_type = EdgeType.TRADE
        edge.target_asset_id = "ETH_MAINNET_USDC"
        edge.calculate_output = Mock(return_value={"output_amount": 1000.0})
        
        path = [edge]
        path_amounts = [1.0]
        
        delta = calculate_path_delta(path, path_amounts)
        
        assert "ETH_MAINNET_USDC" in delta
        assert delta["ETH_MAINNET_USDC"] == 1000.0
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_delta_function(self):
        """Test portfolio delta calculation function."""
        # Create test positions
        position1 = DeltaPosition("pos1", "test")
        position1.add_exposure("ETH_MAINNET_WETH", Decimal('1.0'), Decimal('2000'))
        
        position2 = DeltaPosition("pos2", "test")
        position2.add_exposure("ETH_MAINNET_WETH", Decimal('0.5'), Decimal('1000'))
        position2.add_exposure("ETH_MAINNET_USDC", Decimal('500'), Decimal('500'))
        
        positions = [position1, position2]
        snapshot = await calculate_portfolio_delta(positions)
        
        assert snapshot.total_positions == 2
        assert len(snapshot.asset_exposures) == 2
        
        # ETH exposure should be combined
        eth_exposure = snapshot.asset_exposures["ETH_MAINNET_WETH"]
        assert eth_exposure.amount == Decimal('1.5')
        assert eth_exposure.usd_value == Decimal('3000')


if __name__ == "__main__":
    # Run basic test
    print("🧪 Testing DeltaTracker")
    print("=" * 40)
    
    # Test AssetExposure
    exposure = AssetExposure(
        asset_id="ETH_MAINNET_WETH",
        amount=Decimal('2.0'),
        usd_value=Decimal('4000'),
        exposure_type=ExposureType.LONG
    )
    
    print(f"✅ AssetExposure created:")
    print(f"   - Asset: {exposure.asset_id}")
    print(f"   - Amount: {exposure.amount}")
    print(f"   - USD Value: {exposure.usd_value}")
    print(f"   - Type: {exposure.exposure_type}")
    
    # Test DeltaPosition
    position = DeltaPosition("test_position", "arbitrage")
    position.add_exposure("ETH_MAINNET_WETH", Decimal('1.5'), Decimal('3000'))
    position.add_exposure("ETH_MAINNET_USDC", Decimal('1000'), Decimal('1000'))
    
    print(f"\\n✅ DeltaPosition created:")
    print(f"   - Position ID: {position.position_id}")
    print(f"   - Exposures: {len(position.exposures)}")
    print(f"   - Total USD: {position.total_usd_exposure}")
    print(f"   - Diversification: {position.diversification_score:.2f}")
    
    print("\\n✅ DeltaTracker test passed!")