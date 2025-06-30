"""
Database and Service Adapters for Telegram Bot.

This module provides adapters that connect the Telegram bot to the actual
database and service components when the full system components are not available.
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import text

from yield_arbitrage.database.connection import get_session

logger = logging.getLogger(__name__)


class DatabaseGraphAdapter:
    """Adapter to get graph data from database when GraphEngine is not available."""
    
    def __init__(self):
        self.last_update = datetime.now(timezone.utc)
        self._edges_cache = []
        self._nodes_cache = []
    
    @property
    def edges(self):
        """Get edges from database or cache."""
        return self._edges_cache
    
    @property 
    def nodes(self):
        """Get nodes from database or cache."""
        return self._nodes_cache
    
    async def refresh_data(self):
        """Refresh graph data from database."""
        try:
            async with get_session() as session:
                # Query edges from database
                edge_result = await session.execute(
                    text("SELECT id, protocol_name, token_in, token_out FROM edges LIMIT 1000")
                )
                self._edges_cache = [f"edge_{row.id}" for row in edge_result.fetchall()]
                
                # Query nodes (tokens) from database  
                node_result = await session.execute(
                    text("SELECT DISTINCT token_address FROM edges LIMIT 200")
                )
                self._nodes_cache = [f"node_{row.token_address[:8]}" for row in node_result.fetchall()]
                
                self.last_update = datetime.now(timezone.utc)
                logger.info(f"Graph data refreshed: {len(self._edges_cache)} edges, {len(self._nodes_cache)} nodes")
                
        except Exception as e:
            logger.error(f"Failed to refresh graph data: {e}")
            # Use mock data as fallback
            self._edges_cache = [f"edge_{i}" for i in range(100)]
            self._nodes_cache = [f"node_{i}" for i in range(20)]


class DatabasePositionMonitor:
    """Adapter to monitor positions from database."""
    
    def __init__(self):
        self.is_monitoring = True
        self.active_positions = []
        self.alert_history = []
    
    async def get_active_positions(self) -> List[str]:
        """Get active positions from database."""
        try:
            async with get_session() as session:
                result = await session.execute(
                    text("SELECT position_id FROM positions WHERE status = 'active' LIMIT 50")
                )
                self.active_positions = [row.position_id for row in result.fetchall()]
                return self.active_positions
        except Exception as e:
            logger.error(f"Failed to get active positions: {e}")
            return []
    
    async def get_recent_alerts(self, hours: int = 24) -> List[Any]:
        """Get recent alerts from database."""
        try:
            async with get_session() as session:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
                result = await session.execute(
                    text("""
                        SELECT position_id, alert_type, severity, message, created_at, recommended_action
                        FROM position_alerts 
                        WHERE created_at > :cutoff 
                        ORDER BY created_at DESC 
                        LIMIT 20
                    """),
                    {"cutoff": cutoff}
                )
                
                alerts = []
                for row in result.fetchall():
                    alert = type('Alert', (), {})()
                    alert.position_id = row.position_id
                    alert.alert_type = row.alert_type
                    alert.severity = type('Severity', (), {'value': row.severity})()
                    alert.message = row.message
                    alert.timestamp = row.created_at
                    alert.recommended_action = row.recommended_action
                    alerts.append(alert)
                
                self.alert_history = alerts
                return alerts
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []


class DatabaseDeltaTracker:
    """Adapter to track position deltas from database."""
    
    async def get_all_positions(self) -> Dict[str, Any]:
        """Get all positions with their current state."""
        try:
            async with get_session() as session:
                result = await session.execute(
                    text("""
                        SELECT position_id, position_type, current_value_usd, 
                               initial_value_usd, health_status, created_at
                        FROM positions 
                        WHERE status = 'active'
                        ORDER BY created_at DESC
                        LIMIT 20
                    """)
                )
                
                positions = {}
                for row in result.fetchall():
                    position = type('Position', (), {})()
                    position.position_id = row.position_id
                    position.position_type = row.position_type
                    position.current_value_usd = float(row.current_value_usd or 0)
                    position.initial_value_usd = float(row.initial_value_usd or 0)
                    position.health_status = row.health_status
                    position.created_at = row.created_at
                    position.status = 'active'
                    positions[row.position_id] = position
                
                return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}
    
    async def calculate_portfolio_health(self) -> Dict[str, Any]:
        """Calculate overall portfolio health from database."""
        try:
            async with get_session() as session:
                result = await session.execute(
                    text("""
                        SELECT 
                            COUNT(*) as position_count,
                            SUM(ABS(current_value_usd)) as total_value_usd,
                            SUM(current_value_usd - initial_value_usd) as unrealized_pnl_usd,
                            AVG(CASE WHEN health_status = 'error' THEN 1.0 
                                     WHEN health_status = 'warning' THEN 0.5 
                                     ELSE 0.0 END) as risk_score
                        FROM positions 
                        WHERE status = 'active'
                    """)
                )
                
                row = result.fetchone()
                if row:
                    return {
                        'total_value_usd': float(row.total_value_usd or 0),
                        'unrealized_pnl_usd': float(row.unrealized_pnl_usd or 0),
                        'liquidation_risk_score': float(row.risk_score or 0),
                        'position_count': int(row.position_count or 0)
                    }
                else:
                    return {
                        'total_value_usd': 0.0,
                        'unrealized_pnl_usd': 0.0,
                        'liquidation_risk_score': 0.0,
                        'position_count': 0
                    }
        except Exception as e:
            logger.error(f"Failed to calculate portfolio health: {e}")
            return {
                'total_value_usd': 0.0,
                'unrealized_pnl_usd': 0.0,
                'liquidation_risk_score': 0.0,
                'position_count': 0
            }


class DatabaseExecutionLogger:
    """Adapter to get execution data from database."""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'records_created': 0,
            'records_updated': 0,
            'write_errors': 0,
            'last_write_time': datetime.now(timezone.utc).timestamp()
        }
    
    async def get_execution_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get execution analytics from database."""
        try:
            async with get_session() as session:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
                result = await session.execute(
                    text("""
                        SELECT 
                            COUNT(*) as total_executions,
                            SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_executions,
                            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_executions,
                            AVG(predicted_profit_usd) as avg_predicted_profit_usd,
                            AVG(simulation_time_ms) as avg_simulation_time_ms
                        FROM executions 
                        WHERE created_at > :cutoff
                    """),
                    {"cutoff": cutoff}
                )
                
                row = result.fetchone()
                if row and row.total_executions:
                    total = int(row.total_executions)
                    successful = int(row.successful_executions or 0)
                    return {
                        'total_executions': total,
                        'successful_executions': successful,
                        'failed_executions': int(row.failed_executions or 0),
                        'success_rate': successful / total if total > 0 else 0,
                        'avg_predicted_profit_usd': float(row.avg_predicted_profit_usd or 0),
                        'avg_simulation_time_ms': float(row.avg_simulation_time_ms or 0),
                        'total_predicted_profit_usd': successful * float(row.avg_predicted_profit_usd or 0),
                        'most_common_protocols': {'database': total}  # Simplified
                    }
                else:
                    return {
                        'total_executions': 0,
                        'successful_executions': 0,
                        'failed_executions': 0,
                        'success_rate': 0.0,
                        'avg_predicted_profit_usd': 0.0,
                        'avg_simulation_time_ms': 0.0,
                        'total_predicted_profit_usd': 0.0,
                        'most_common_protocols': {}
                    }
        except Exception as e:
            logger.error(f"Failed to get execution analytics: {e}")
            return {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'success_rate': 0.0,
                'avg_predicted_profit_usd': 0.0,
                'avg_simulation_time_ms': 0.0,
                'total_predicted_profit_usd': 0.0,
                'most_common_protocols': {}
            }


# Mock components for when database is not available
class MockDataCollector:
    def __init__(self):
        self.is_running = False
        self.last_collection_time = datetime.now(timezone.utc)
        self.collections_today = 0


class MockPathfinder:
    async def search(self, start_asset: str, amount: float, beam_width: int = 10):
        """Mock pathfinder search."""
        await asyncio.sleep(0.1)  # Simulate work
        return [type('Path', (), {'edges': [f'edge_{i}']})() for i in range(min(beam_width, 3))]


class MockSimulator:
    async def simulate_path(self, path, amount: float, start_asset: str):
        """Mock path simulation."""
        await asyncio.sleep(0.1)  # Simulate work
        return {
            'is_profitable': False,
            'profit_usd': 0.0,
            'gas_cost_usd': 15.0,
            'profit_percentage': 0.0,
            'estimated_apr': 0.0,
            'risk_score': 0.5
        }