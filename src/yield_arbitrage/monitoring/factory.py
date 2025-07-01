"""Factory for creating production monitoring components."""
import logging
import os
from typing import Optional, Any

from ..cache import get_redis
from ..blockchain_connector.provider import blockchain_provider
from ..execution.asset_oracle import ProductionOracleManager
from ..risk.delta_tracker import DeltaTracker
from ..database.execution_logger import get_execution_logger
from .position_monitor import PositionMonitor

logger = logging.getLogger(__name__)


class MonitoringComponentFactory:
    """Factory for creating production monitoring components with proper dependencies."""
    
    def __init__(self):
        self._asset_oracle: Optional[ProductionOracleManager] = None
        self._delta_tracker: Optional[DeltaTracker] = None
        self._execution_logger = None
        self._position_monitor: Optional[PositionMonitor] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all components in the correct dependency order."""
        if self._initialized:
            return
        
        logger.info("🏭 Initializing monitoring component factory...")
        
        try:
            # 1. Initialize Asset Oracle (no dependencies)
            await self._initialize_asset_oracle()
            
            # 2. Initialize Delta Tracker (depends on Asset Oracle)
            await self._initialize_delta_tracker()
            
            # 3. Initialize Execution Logger (no dependencies)
            await self._initialize_execution_logger()
            
            # 4. Initialize Position Monitor (depends on Delta Tracker and Asset Oracle)
            await self._initialize_position_monitor()
            
            self._initialized = True
            logger.info("✅ Monitoring component factory initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize monitoring component factory: {e}")
            raise
    
    async def _initialize_asset_oracle(self) -> None:
        """Initialize the production asset oracle."""
        logger.info("📊 Initializing Asset Oracle...")
        
        try:
            # Get required dependencies
            redis_client = await get_redis()
            
            # Initialize blockchain provider if not already done
            if not blockchain_provider._initialized:
                await blockchain_provider.initialize()
            
            # Get CoinGecko API key if available
            coingecko_api_key = os.getenv('COINGECKO_API_KEY')
            if coingecko_api_key:
                logger.info("Using CoinGecko API key for enhanced rate limits")
            else:
                logger.info("No CoinGecko API key found, using free tier")
            
            # Create production oracle manager
            self._asset_oracle = ProductionOracleManager(
                redis_client=redis_client,
                blockchain_provider=blockchain_provider,
                coingecko_api_key=coingecko_api_key,
                defillama_enabled=True,
                on_chain_enabled=True
            )
            
            await self._asset_oracle.initialize()
            logger.info("✅ Asset Oracle initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Asset Oracle: {e}")
            # Create a fallback mock oracle
            logger.warning("Creating fallback mock oracle")
            self._asset_oracle = MockAssetOracle()
    
    async def _initialize_delta_tracker(self) -> None:
        """Initialize the delta tracker."""
        logger.info("📈 Initializing Delta Tracker...")
        
        try:
            if not self._asset_oracle:
                raise ValueError("Asset Oracle must be initialized before Delta Tracker")
            
            self._delta_tracker = DeltaTracker(asset_oracle=self._asset_oracle)
            logger.info("✅ Delta Tracker initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Delta Tracker: {e}")
            raise
    
    async def _initialize_execution_logger(self) -> None:
        """Initialize the execution logger."""
        logger.info("📝 Initializing Execution Logger...")
        
        try:
            self._execution_logger = get_execution_logger()
            logger.info("✅ Execution Logger initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Execution Logger: {e}")
            raise
    
    async def _initialize_position_monitor(self) -> None:
        """Initialize the position monitor."""
        logger.info("🔍 Initializing Position Monitor...")
        
        try:
            if not self._delta_tracker:
                raise ValueError("Delta Tracker must be initialized before Position Monitor")
            if not self._asset_oracle:
                raise ValueError("Asset Oracle must be initialized before Position Monitor")
            
            self._position_monitor = PositionMonitor(
                delta_tracker=self._delta_tracker,
                asset_oracle=self._asset_oracle
            )
            logger.info("✅ Position Monitor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Position Monitor: {e}")
            raise
    
    def get_asset_oracle(self) -> Optional[ProductionOracleManager]:
        """Get the initialized asset oracle."""
        return self._asset_oracle
    
    def get_delta_tracker(self) -> Optional[DeltaTracker]:
        """Get the initialized delta tracker."""
        return self._delta_tracker
    
    def get_execution_logger(self):
        """Get the initialized execution logger."""
        return self._execution_logger
    
    def get_position_monitor(self) -> Optional[PositionMonitor]:
        """Get the initialized position monitor."""
        return self._position_monitor
    
    async def close(self) -> None:
        """Close all components."""
        logger.info("🛑 Closing monitoring components...")
        
        if self._asset_oracle and hasattr(self._asset_oracle, 'close'):
            await self._asset_oracle.close()
        
        self._initialized = False
        logger.info("✅ Monitoring components closed")


# Mock oracle for fallback
class MockAssetOracle:
    """Mock asset oracle that returns fixed prices for testing."""
    
    async def get_price_usd(self, asset_id: str) -> Optional[float]:
        """Return mock prices for common assets."""
        mock_prices = {
            "ETH_MAINNET_WETH": 3000.0,
            "ETH_MAINNET_USDC": 1.0,
            "ETH_MAINNET_USDT": 1.0,
            "ETH_MAINNET_DAI": 1.0,
            "ETH_MAINNET_WBTC": 65000.0,
        }
        return mock_prices.get(asset_id, 100.0)  # Default to $100
    
    async def get_price_details(self, asset_id: str):
        """Return mock price details."""
        from ..execution.asset_oracle import AssetPrice
        from datetime import datetime
        
        price = await self.get_price_usd(asset_id)
        if price is None:
            return None
        
        return AssetPrice(
            asset_id=asset_id,
            symbol=asset_id.split('_')[-1] if '_' in asset_id else asset_id,
            price_usd=price,
            timestamp=datetime.utcnow(),
            source="mock",
            confidence=0.5
        )
    
    async def get_prices_batch(self, asset_ids: list) -> dict:
        """Return mock batch prices."""
        return {asset_id: await self.get_price_usd(asset_id) for asset_id in asset_ids}


# Global factory instance
monitoring_factory = MonitoringComponentFactory()


async def get_monitoring_components():
    """Get all monitoring components, initializing if needed."""
    if not monitoring_factory._initialized:
        await monitoring_factory.initialize()
    
    return {
        'asset_oracle': monitoring_factory.get_asset_oracle(),
        'delta_tracker': monitoring_factory.get_delta_tracker(),
        'execution_logger': monitoring_factory.get_execution_logger(),
        'position_monitor': monitoring_factory.get_position_monitor()
    }