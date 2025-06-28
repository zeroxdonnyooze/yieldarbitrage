"""
Delta Tracker for Market Exposure Management.

This module provides comprehensive market exposure tracking and delta management
for arbitrage paths and active positions. Integrates with existing asset oracle
and monitoring infrastructure.
"""
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from decimal import Decimal
from enum import Enum

from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType
from yield_arbitrage.execution.asset_oracle import AssetOracleBase

logger = logging.getLogger(__name__)


class ExposureType(str, Enum):
    """Types of market exposure."""
    LONG = "long"           # Long exposure to asset
    SHORT = "short"         # Short exposure to asset
    NEUTRAL = "neutral"     # No net exposure


@dataclass
class AssetExposure:
    """Represents exposure to a single asset."""
    asset_id: str
    amount: Decimal  # Net amount held (positive = long, negative = short)
    usd_value: Decimal  # Current USD value of exposure
    exposure_type: ExposureType
    confidence: float = 1.0  # Confidence in exposure calculation (0-1)
    
    # Risk metrics
    volatility: Optional[float] = None
    liquidity_score: Optional[float] = None
    correlation_score: Optional[float] = None
    
    # Timestamps
    first_exposure_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def update_value(self, new_usd_value: Decimal):
        """Update USD value and timestamp."""
        self.usd_value = new_usd_value
        self.last_updated = time.time()
    
    def add_exposure(self, amount: Decimal, usd_value: Decimal):
        """Add to existing exposure."""
        self.amount += amount
        self.usd_value += usd_value
        self.last_updated = time.time()
        
        # Update exposure type
        if self.amount > 0:
            self.exposure_type = ExposureType.LONG
        elif self.amount < 0:
            self.exposure_type = ExposureType.SHORT
        else:
            self.exposure_type = ExposureType.NEUTRAL
    
    def is_stale(self, max_age_seconds: int = 300) -> bool:
        """Check if exposure data is stale."""
        return time.time() - self.last_updated > max_age_seconds


@dataclass
class DeltaPosition:
    """Represents a complete position with multiple asset exposures."""
    position_id: str
    position_type: str  # e.g., 'arbitrage', 'yield_farming', 'lending'
    exposures: Dict[str, AssetExposure] = field(default_factory=dict)
    
    # Position metadata
    entry_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    is_active: bool = True
    
    # Risk metrics
    total_usd_exposure: Decimal = Decimal('0')
    max_single_asset_exposure: Decimal = Decimal('0')
    diversification_score: float = 0.0
    
    def add_exposure(self, asset_id: str, amount: Decimal, usd_value: Decimal, 
                    asset_oracle: Optional[AssetOracleBase] = None):
        """Add or update asset exposure in this position."""
        if asset_id in self.exposures:
            self.exposures[asset_id].add_exposure(amount, usd_value)
        else:
            exposure_type = ExposureType.LONG if amount > 0 else (
                ExposureType.SHORT if amount < 0 else ExposureType.NEUTRAL
            )
            
            self.exposures[asset_id] = AssetExposure(
                asset_id=asset_id,
                amount=amount,
                usd_value=usd_value,
                exposure_type=exposure_type
            )
            
            # Enhance with oracle data if available
            if asset_oracle:
                self._enhance_exposure_with_oracle(asset_id, asset_oracle)
        
        self._update_position_metrics()
    
    def _enhance_exposure_with_oracle(self, asset_id: str, asset_oracle: AssetOracleBase):
        """Enhance exposure with data from asset oracle."""
        try:
            # Check if asset is stable based on naming convention
            is_stable = self._is_stable_asset_by_name(asset_id)
            
            if asset_id in self.exposures:
                exposure = self.exposures[asset_id]
                
                # Set volatility based on asset type
                if is_stable:
                    exposure.volatility = 0.01  # 1% assumed volatility for stables
                    exposure.liquidity_score = 0.9  # High liquidity for stables
                else:
                    exposure.volatility = 0.15  # 15% assumed volatility for non-stables
                    exposure.liquidity_score = 0.7  # Medium liquidity assumption
                
        except Exception as e:
            logger.warning(f"Failed to enhance exposure with oracle data: {e}")
    
    def _is_stable_asset_by_name(self, asset_id: str) -> bool:
        """Check if asset is stable based on naming convention."""
        stable_indicators = ['USDC', 'USDT', 'DAI', 'FRAX', 'LUSD', 'BUSD']
        return any(indicator in asset_id.upper() for indicator in stable_indicators)
    
    def _update_position_metrics(self):
        """Update position-level risk metrics."""
        self.last_updated = time.time()
        
        # Calculate total USD exposure
        self.total_usd_exposure = sum(
            abs(exposure.usd_value) for exposure in self.exposures.values()
        )
        
        # Find maximum single asset exposure
        if self.exposures:
            self.max_single_asset_exposure = max(
                abs(exposure.usd_value) for exposure in self.exposures.values()
            )
        
        # Calculate simple diversification score
        if len(self.exposures) > 1 and self.total_usd_exposure > 0:
            max_weight = self.max_single_asset_exposure / self.total_usd_exposure
            self.diversification_score = 1.0 - float(max_weight)
        else:
            self.diversification_score = 0.0
    
    def get_net_exposure(self, asset_id: str) -> Decimal:
        """Get net exposure for specific asset."""
        if asset_id in self.exposures:
            return self.exposures[asset_id].amount
        return Decimal('0')
    
    def get_total_risk_value(self) -> Decimal:
        """Calculate total risk-adjusted value."""
        total_risk = Decimal('0')
        
        for exposure in self.exposures.values():
            # Apply volatility adjustment if available
            if exposure.volatility:
                risk_multiplier = Decimal(str(1 + exposure.volatility))
                total_risk += abs(exposure.usd_value) * risk_multiplier
            else:
                total_risk += abs(exposure.usd_value)
        
        return total_risk
    
    def is_hedged(self, correlation_threshold: float = 0.8) -> bool:
        """Check if position appears to be hedged."""
        # Simple heuristic: if we have both long and short exposures
        has_long = any(exp.exposure_type == ExposureType.LONG for exp in self.exposures.values())
        has_short = any(exp.exposure_type == ExposureType.SHORT for exp in self.exposures.values())
        
        return has_long and has_short


@dataclass
class DeltaSnapshot:
    """Snapshot of market exposure at a point in time."""
    timestamp: float = field(default_factory=time.time)
    total_positions: int = 0
    
    # Aggregate exposure by asset
    asset_exposures: Dict[str, AssetExposure] = field(default_factory=dict)
    
    # Portfolio metrics
    total_usd_long: Decimal = Decimal('0')
    total_usd_short: Decimal = Decimal('0')
    net_usd_exposure: Decimal = Decimal('0')
    
    # Risk metrics
    largest_single_exposure: Decimal = Decimal('0')
    num_assets_exposed: int = 0
    portfolio_correlation: float = 0.0
    
    def calculate_metrics(self):
        """Calculate portfolio-level metrics."""
        self.total_usd_long = sum(
            exp.usd_value for exp in self.asset_exposures.values()
            if exp.exposure_type == ExposureType.LONG
        )
        
        self.total_usd_short = sum(
            abs(exp.usd_value) for exp in self.asset_exposures.values()
            if exp.exposure_type == ExposureType.SHORT
        )
        
        self.net_usd_exposure = self.total_usd_long - self.total_usd_short
        
        if self.asset_exposures:
            self.largest_single_exposure = max(
                abs(exp.usd_value) for exp in self.asset_exposures.values()
            )
        
        self.num_assets_exposed = len([
            exp for exp in self.asset_exposures.values()
            if exp.exposure_type != ExposureType.NEUTRAL
        ])


class DeltaTracker:
    """
    Comprehensive market exposure tracker for arbitrage and yield strategies.
    
    Integrates with existing asset oracle infrastructure to track real-time
    market exposure across all active positions and paths.
    """
    
    def __init__(self, asset_oracle: AssetOracleBase):
        """
        Initialize delta tracker with asset oracle.
        
        Args:
            asset_oracle: Asset oracle for price data and asset properties
        """
        self.asset_oracle = asset_oracle
        
        # Position tracking
        self.active_positions: Dict[str, DeltaPosition] = {}
        
        # Configuration
        self.max_single_asset_exposure_usd = Decimal('100000')  # $100k limit
        self.max_total_exposure_usd = Decimal('1000000')  # $1M limit
        self.volatility_threshold = 0.20  # 20% volatility threshold
        
        # Statistics
        self.stats = {
            "positions_tracked": 0,
            "exposures_calculated": 0,
            "risk_alerts_generated": 0,
            "last_snapshot_time": 0.0
        }
    
    async def calculate_path_delta(
        self, 
        path: List[YieldGraphEdge], 
        path_amounts: List[float]
    ) -> Dict[str, float]:
        """
        Calculate net market exposure (delta) for a complete arbitrage path.
        
        Args:
            path: List of edges representing the arbitrage path
            path_amounts: Amount at each step of the path
            
        Returns:
            Dictionary mapping asset_id to net exposure amount
        """
        logger.debug(f"Calculating delta for path with {len(path)} edges")
        
        delta_exposure = defaultdict(float)
        current_holdings = defaultdict(float)
        
        try:
            # Process each edge in the path
            for i, edge in enumerate(path):
                input_amount = path_amounts[i] if i < len(path_amounts) else 0
                
                # Handle different edge types
                if edge.edge_type == EdgeType.TRADE:
                    # Trading: gain target asset, lose source asset
                    output_result = edge.calculate_output(input_amount)
                    if "error" not in output_result:
                        output_amount = output_result.get("output_amount", 0)
                        
                        # Update holdings
                        current_holdings[edge.source_asset_id] -= input_amount
                        current_holdings[edge.target_asset_id] += output_amount
                        
                        # Track exposure during holding period
                        if not await self._is_stable_asset(edge.target_asset_id):
                            delta_exposure[edge.target_asset_id] += output_amount
                
                elif edge.edge_type == EdgeType.LEND:
                    # Lending: deposit asset, receive yield-bearing token
                    current_holdings[edge.source_asset_id] -= input_amount
                    current_holdings[edge.target_asset_id] += input_amount  # 1:1 for lending
                    
                    # No delta for stable assets or if immediately redeemed
                    if not await self._is_stable_asset(edge.source_asset_id):
                        delta_exposure[edge.source_asset_id] += input_amount
                
                elif edge.edge_type == EdgeType.BORROW:
                    # Borrowing: receive asset, create debt obligation
                    borrowed_amount = input_amount
                    current_holdings[edge.target_asset_id] += borrowed_amount
                    
                    # Short exposure to borrowed asset
                    if not await self._is_stable_asset(edge.target_asset_id):
                        delta_exposure[edge.target_asset_id] -= borrowed_amount
                
                elif edge.edge_type == EdgeType.FLASH_LOAN:
                    # Flash loan: temporary liquidity, must be repaid
                    # No net delta if properly repaid in same transaction
                    pass
                
                elif edge.edge_type == EdgeType.WAIT:
                    # Waiting: maintain current exposure
                    # Delta persists during wait period
                    pass
            
            # Clean up near-zero exposures
            cleaned_delta = {
                asset_id: amount for asset_id, amount in delta_exposure.items()
                if abs(amount) > 1e-6  # Remove dust amounts
            }
            
            self.stats["exposures_calculated"] += 1
            logger.debug(f"Calculated delta exposure for {len(cleaned_delta)} assets")
            
            return cleaned_delta
            
        except Exception as e:
            logger.error(f"Error calculating path delta: {e}")
            return {}
    
    async def add_position(
        self, 
        position_id: str, 
        position_type: str, 
        path: Optional[List[YieldGraphEdge]] = None,
        path_amounts: Optional[List[float]] = None,
        initial_exposures: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> bool:
        """
        Add a new position for tracking.
        
        Args:
            position_id: Unique identifier for the position
            position_type: Type of position (e.g., 'arbitrage', 'yield_farming')
            path: Optional path that created this position
            path_amounts: Optional amounts for path-based positions
            initial_exposures: Optional direct exposure specification (asset_id -> (amount, usd_value))
            
        Returns:
            True if position was added successfully
        """
        try:
            if position_id in self.active_positions:
                logger.warning(f"Position {position_id} already exists")
                return False
            
            # Create new position
            position = DeltaPosition(
                position_id=position_id,
                position_type=position_type
            )
            
            # Add exposures from path calculation
            if path and path_amounts:
                path_delta = await self.calculate_path_delta(path, path_amounts)
                
                for asset_id, amount in path_delta.items():
                    # Get current price for USD value calculation
                    usd_value = await self._calculate_usd_value(asset_id, amount)
                    position.add_exposure(
                        asset_id, 
                        Decimal(str(amount)), 
                        usd_value, 
                        self.asset_oracle
                    )
            
            # Add direct exposures if provided
            if initial_exposures:
                for asset_id, (amount, usd_val) in initial_exposures.items():
                    position.add_exposure(
                        asset_id,
                        Decimal(str(amount)),
                        Decimal(str(usd_val)),
                        self.asset_oracle
                    )
            
            # Validate position against risk limits
            if not self._validate_position_risk(position):
                logger.warning(f"Position {position_id} exceeds risk limits")
                return False
            
            # Add to tracking
            self.active_positions[position_id] = position
            self.stats["positions_tracked"] += 1
            
            logger.info(f"Added position {position_id} with {len(position.exposures)} exposures")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position {position_id}: {e}")
            return False
    
    def update_position_exposure(
        self, 
        position_id: str, 
        asset_id: str, 
        amount_delta: float,
        usd_value_delta: float
    ) -> bool:
        """
        Update exposure for an existing position.
        
        Args:
            position_id: Position to update
            asset_id: Asset being updated
            amount_delta: Change in amount (can be negative)
            usd_value_delta: Change in USD value
            
        Returns:
            True if update was successful
        """
        try:
            if position_id not in self.active_positions:
                logger.error(f"Position {position_id} not found")
                return False
            
            position = self.active_positions[position_id]
            position.add_exposure(
                asset_id,
                Decimal(str(amount_delta)),
                Decimal(str(usd_value_delta)),
                self.asset_oracle
            )
            
            # Re-validate risk limits
            if not self._validate_position_risk(position):
                logger.warning(f"Position {position_id} exceeds risk limits after update")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating position exposure: {e}")
            return False
    
    def remove_position(self, position_id: str) -> bool:
        """Remove position from tracking."""
        if position_id in self.active_positions:
            del self.active_positions[position_id]
            logger.info(f"Removed position {position_id}")
            return True
        return False
    
    def get_position(self, position_id: str) -> Optional[DeltaPosition]:
        """Get position by ID."""
        return self.active_positions.get(position_id)
    
    def get_all_positions(self) -> Dict[str, DeltaPosition]:
        """Get all active positions."""
        return self.active_positions.copy()
    
    async def get_portfolio_snapshot(self) -> DeltaSnapshot:
        """
        Get current portfolio exposure snapshot.
        
        Returns:
            DeltaSnapshot with aggregated portfolio data
        """
        snapshot = DeltaSnapshot(total_positions=len(self.active_positions))
        
        # Aggregate exposures across all positions
        for position in self.active_positions.values():
            for asset_id, exposure in position.exposures.items():
                if asset_id in snapshot.asset_exposures:
                    # Combine exposures
                    existing = snapshot.asset_exposures[asset_id]
                    existing.add_exposure(exposure.amount, exposure.usd_value)
                else:
                    # Create new aggregate exposure
                    snapshot.asset_exposures[asset_id] = AssetExposure(
                        asset_id=asset_id,
                        amount=exposure.amount,
                        usd_value=exposure.usd_value,
                        exposure_type=exposure.exposure_type,
                        confidence=exposure.confidence
                    )
        
        # Calculate portfolio metrics
        snapshot.calculate_metrics()
        
        self.stats["last_snapshot_time"] = snapshot.timestamp
        return snapshot
    
    async def check_risk_limits(self) -> List[Dict[str, Any]]:
        """
        Check all positions against risk limits.
        
        Returns:
            List of risk alerts
        """
        alerts = []
        
        try:
            # Get portfolio snapshot
            snapshot = await self.get_portfolio_snapshot()
            
            # Check total exposure limit
            total_exposure = snapshot.total_usd_long + snapshot.total_usd_short
            if total_exposure > self.max_total_exposure_usd:
                alerts.append({
                    "type": "total_exposure_limit",
                    "severity": "high",
                    "message": f"Total exposure ${total_exposure:,.2f} exceeds limit ${self.max_total_exposure_usd:,.2f}",
                    "value": float(total_exposure),
                    "limit": float(self.max_total_exposure_usd)
                })
            
            # Check single asset exposure limits
            for asset_id, exposure in snapshot.asset_exposures.items():
                if abs(exposure.usd_value) > self.max_single_asset_exposure_usd:
                    alerts.append({
                        "type": "single_asset_exposure_limit",
                        "severity": "medium",
                        "message": f"Exposure to {asset_id} ${exposure.usd_value:,.2f} exceeds limit",
                        "asset_id": asset_id,
                        "value": float(abs(exposure.usd_value)),
                        "limit": float(self.max_single_asset_exposure_usd)
                    })
            
            # Check volatility exposure
            for asset_id, exposure in snapshot.asset_exposures.items():
                if exposure.volatility and exposure.volatility > self.volatility_threshold:
                    risk_value = abs(exposure.usd_value) * Decimal(str(exposure.volatility))
                    if risk_value > self.max_single_asset_exposure_usd * Decimal('0.1'):
                        alerts.append({
                            "type": "volatility_risk",
                            "severity": "medium",
                            "message": f"High volatility exposure to {asset_id}",
                            "asset_id": asset_id,
                            "volatility": exposure.volatility,
                            "risk_value": float(risk_value)
                        })
            
            if alerts:
                self.stats["risk_alerts_generated"] += len(alerts)
                logger.warning(f"Generated {len(alerts)} risk alerts")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return []
    
    def _validate_position_risk(self, position: DeltaPosition) -> bool:
        """Validate position against risk limits."""
        # Check total position size
        if position.total_usd_exposure > self.max_single_asset_exposure_usd * 5:
            return False
        
        # Check individual asset exposure
        if position.max_single_asset_exposure > self.max_single_asset_exposure_usd:
            return False
        
        return True
    
    async def _is_stable_asset(self, asset_id: str) -> bool:
        """Check if asset is considered stable."""
        # Use naming convention for stability check
        return self._is_stable_asset_by_name(asset_id)
    
    async def _calculate_usd_value(self, asset_id: str, amount: float) -> Decimal:
        """Calculate USD value of asset amount."""
        try:
            price = await self.asset_oracle.get_asset_price(asset_id)
            return Decimal(str(amount)) * Decimal(str(price))
        except Exception as e:
            logger.warning(f"Failed to get price for {asset_id}: {e}")
            return Decimal('0')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            **self.stats,
            "active_positions": len(self.active_positions),
            "total_exposures": sum(len(pos.exposures) for pos in self.active_positions.values())
        }


# Convenience functions

def calculate_path_delta(path: List[YieldGraphEdge], path_amounts: List[float]) -> Dict[str, float]:
    """
    Convenience function to calculate path delta without full tracker.
    
    Args:
        path: List of yield graph edges
        path_amounts: Amounts at each step
        
    Returns:
        Dictionary of asset exposures
    """
    delta_exposure = defaultdict(float)
    
    for i, edge in enumerate(path):
        input_amount = path_amounts[i] if i < len(path_amounts) else 0
        
        # Simple delta calculation for basic edge types
        if edge.edge_type == EdgeType.TRADE:
            output_result = edge.calculate_output(input_amount)
            if "error" not in output_result:
                output_amount = output_result.get("output_amount", 0)
                delta_exposure[edge.target_asset_id] += output_amount
        
        elif edge.edge_type in [EdgeType.LEND, EdgeType.BORROW]:
            delta_exposure[edge.target_asset_id] += input_amount
    
    return dict(delta_exposure)


async def calculate_portfolio_delta(positions: List[DeltaPosition]) -> DeltaSnapshot:
    """
    Calculate aggregate portfolio delta from multiple positions.
    
    Args:
        positions: List of delta positions
        
    Returns:
        Portfolio delta snapshot
    """
    snapshot = DeltaSnapshot(total_positions=len(positions))
    
    for position in positions:
        for asset_id, exposure in position.exposures.items():
            if asset_id in snapshot.asset_exposures:
                existing = snapshot.asset_exposures[asset_id]
                existing.add_exposure(exposure.amount, exposure.usd_value)
            else:
                snapshot.asset_exposures[asset_id] = AssetExposure(
                    asset_id=asset_id,
                    amount=exposure.amount,
                    usd_value=exposure.usd_value,
                    exposure_type=exposure.exposure_type
                )
    
    snapshot.calculate_metrics()
    return snapshot