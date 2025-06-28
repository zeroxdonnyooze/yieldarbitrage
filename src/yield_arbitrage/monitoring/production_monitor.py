"""Production monitoring and validation system for DeFi arbitrage operations."""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum
import statistics
from decimal import Decimal

from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeState
from yield_arbitrage.execution.onchain_price_oracle import OnChainPriceOracle
from yield_arbitrage.protocols.production_registry import production_registry

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringCategory(str, Enum):
    """Categories of monitoring checks."""
    DATA_QUALITY = "data_quality"
    PRICE_DEVIATION = "price_deviation"
    LIQUIDITY_MONITORING = "liquidity_monitoring"
    EDGE_STATE_HEALTH = "edge_state_health"
    SYSTEM_PERFORMANCE = "system_performance"
    BLOCKCHAIN_CONNECTIVITY = "blockchain_connectivity"


@dataclass
class Alert:
    """System alert with metadata."""
    alert_id: str
    category: MonitoringCategory
    severity: AlertSeverity
    title: str
    description: str
    timestamp: float = field(default_factory=time.time)
    source_component: str = ""
    affected_assets: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None


@dataclass
class MonitoringMetric:
    """Individual monitoring metric."""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    check_name: str
    status: str  # "healthy", "degraded", "unhealthy"
    metrics: List[MonitoringMetric]
    alerts: List[Alert]
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


class ProductionMonitor:
    """
    Production monitoring and validation system for DeFi arbitrage operations.
    
    This system:
    - Monitors data quality and system health
    - Detects price deviations and anomalies
    - Tracks liquidity thresholds and edge state health
    - Generates alerts and metrics for production visibility
    - Validates real-time data against expected ranges
    """
    
    def __init__(
        self,
        blockchain_provider,
        oracle: OnChainPriceOracle,
        redis_client,
        alert_webhook_url: Optional[str] = None
    ):
        """
        Initialize the production monitoring system.
        
        Args:
            blockchain_provider: BlockchainProvider instance
            oracle: OnChainPriceOracle for price validation
            redis_client: Redis client for metrics storage
            alert_webhook_url: Optional webhook URL for alert notifications
        """
        self.blockchain_provider = blockchain_provider
        self.oracle = oracle
        self.redis_client = redis_client
        self.alert_webhook_url = alert_webhook_url
        
        # Monitoring state
        self.active_alerts: Dict[str, Alert] = {}
        self.metrics_history: Dict[str, List[MonitoringMetric]] = {}
        self.health_checks: Dict[str, Callable] = {}
        
        # Configuration
        self.monitoring_config = {
            "price_deviation_threshold": 0.05,  # 5% deviation alert
            "price_deviation_critical": 0.15,   # 15% critical deviation
            "min_liquidity_usd": 10000,          # $10k minimum liquidity
            "critical_liquidity_usd": 1000,     # $1k critical liquidity
            "edge_staleness_warning": 300,      # 5 minutes warning
            "edge_staleness_critical": 900,     # 15 minutes critical
            "gas_price_warning": 100,           # 100 gwei warning
            "gas_price_critical": 200,          # 200 gwei critical
            "check_interval_seconds": 30,       # Run checks every 30 seconds
            "alert_cooldown_seconds": 300       # 5 minute alert cooldown
        }
        
        # Statistics tracking
        self.stats = {
            "total_checks": 0,
            "healthy_checks": 0,
            "degraded_checks": 0,
            "unhealthy_checks": 0,
            "alerts_generated": 0,
            "alerts_resolved": 0,
            "average_check_time_ms": 0.0
        }
        
        # Register default health checks
        self._register_default_health_checks()
        
        # Monitoring control
        self.monitoring_active = False
        self.monitoring_task = None
    
    def _register_default_health_checks(self) -> None:
        """Register default health checks for production monitoring."""
        self.health_checks = {
            "blockchain_connectivity": self._check_blockchain_connectivity,
            "price_oracle_health": self._check_price_oracle_health,
            "data_quality_validation": self._check_data_quality,
            "liquidity_thresholds": self._check_liquidity_thresholds,
            "edge_state_freshness": self._check_edge_state_freshness,
            "gas_price_monitoring": self._check_gas_prices,
            "protocol_registry_health": self._check_protocol_registry,
            "system_performance": self._check_system_performance
        }
    
    async def start_monitoring(self) -> None:
        """Start the production monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        logger.info("Starting production monitoring system...")
        self.monitoring_active = True
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Production monitoring system started")
    
    async def stop_monitoring(self) -> None:
        """Stop the production monitoring system."""
        logger.info("Stopping production monitoring system...")
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Production monitoring system stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs health checks periodically."""
        while self.monitoring_active:
            try:
                # Run all health checks
                await self._run_all_health_checks()
                
                # Process alerts
                await self._process_alerts()
                
                # Update statistics
                self._update_monitoring_stats()
                
                # Sleep until next check
                await asyncio.sleep(self.monitoring_config["check_interval_seconds"])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Longer sleep on error
    
    async def _run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for check_name, check_func in self.health_checks.items():
            start_time = time.time()
            
            try:
                result = await check_func()
                result.duration_ms = (time.time() - start_time) * 1000
                results[check_name] = result
                
                # Store metrics
                for metric in result.metrics:
                    await self._store_metric(metric)
                
                # Handle alerts
                for alert in result.alerts:
                    await self._handle_alert(alert)
                
            except Exception as e:
                # Create error result
                error_alert = Alert(
                    alert_id=f"health_check_error_{check_name}_{int(time.time())}",
                    category=MonitoringCategory.SYSTEM_PERFORMANCE,
                    severity=AlertSeverity.ERROR,
                    title=f"Health check failed: {check_name}",
                    description=f"Health check {check_name} failed with error: {e}",
                    source_component="monitoring_system"
                )
                
                results[check_name] = HealthCheckResult(
                    check_name=check_name,
                    status="unhealthy",
                    metrics=[],
                    alerts=[error_alert],
                    duration_ms=(time.time() - start_time) * 1000
                )
                
                await self._handle_alert(error_alert)
        
        return results
    
    async def _check_blockchain_connectivity(self) -> HealthCheckResult:
        """Check blockchain connectivity across all configured chains."""
        metrics = []
        alerts = []
        status = "healthy"
        details = {}
        
        chains = ["ethereum", "arbitrum", "base", "sonic", "berachain"]
        connected_chains = 0
        
        for chain in chains:
            try:
                web3 = await self.blockchain_provider.get_web3(chain)
                if web3:
                    block_number = await web3.eth.block_number
                    gas_price = await web3.eth.gas_price
                    
                    # Add connectivity metric
                    metrics.append(MonitoringMetric(
                        name=f"blockchain_block_number",
                        value=float(block_number),
                        unit="blocks",
                        tags={"chain": chain}
                    ))
                    
                    # Add gas price metric
                    gas_price_gwei = gas_price / 1e9
                    metrics.append(MonitoringMetric(
                        name=f"gas_price",
                        value=gas_price_gwei,
                        unit="gwei",
                        tags={"chain": chain},
                        threshold_warning=self.monitoring_config["gas_price_warning"],
                        threshold_critical=self.monitoring_config["gas_price_critical"]
                    ))
                    
                    details[chain] = {
                        "connected": True,
                        "block_number": block_number,
                        "gas_price_gwei": gas_price_gwei
                    }
                    connected_chains += 1
                    
                    # Check gas price alerts
                    if gas_price_gwei > self.monitoring_config["gas_price_critical"]:
                        alerts.append(Alert(
                            alert_id=f"gas_price_critical_{chain}_{int(time.time())}",
                            category=MonitoringCategory.BLOCKCHAIN_CONNECTIVITY,
                            severity=AlertSeverity.CRITICAL,
                            title=f"Critical gas prices on {chain}",
                            description=f"Gas price on {chain} is {gas_price_gwei:.1f} gwei (critical threshold: {self.monitoring_config['gas_price_critical']} gwei)",
                            source_component="blockchain_connectivity",
                            affected_assets=[chain],
                            metrics={"gas_price_gwei": gas_price_gwei}
                        ))
                    elif gas_price_gwei > self.monitoring_config["gas_price_warning"]:
                        alerts.append(Alert(
                            alert_id=f"gas_price_warning_{chain}_{int(time.time())}",
                            category=MonitoringCategory.BLOCKCHAIN_CONNECTIVITY,
                            severity=AlertSeverity.WARNING,
                            title=f"High gas prices on {chain}",
                            description=f"Gas price on {chain} is {gas_price_gwei:.1f} gwei (warning threshold: {self.monitoring_config['gas_price_warning']} gwei)",
                            source_component="blockchain_connectivity",
                            affected_assets=[chain],
                            metrics={"gas_price_gwei": gas_price_gwei}
                        ))
                        
                else:
                    details[chain] = {"connected": False}
                    status = "degraded"
                    
            except Exception as e:
                details[chain] = {"connected": False, "error": str(e)}
                status = "degraded"
        
        # Overall connectivity metric
        connectivity_ratio = connected_chains / len(chains)
        metrics.append(MonitoringMetric(
            name="blockchain_connectivity_ratio",
            value=connectivity_ratio,
            unit="ratio",
            threshold_warning=0.8,
            threshold_critical=0.6
        ))
        
        # Generate connectivity alerts
        if connectivity_ratio < 0.6:
            status = "unhealthy"
            alerts.append(Alert(
                alert_id=f"blockchain_connectivity_critical_{int(time.time())}",
                category=MonitoringCategory.BLOCKCHAIN_CONNECTIVITY,
                severity=AlertSeverity.CRITICAL,
                title="Critical blockchain connectivity issues",
                description=f"Only {connected_chains}/{len(chains)} blockchain connections active",
                source_component="blockchain_connectivity",
                metrics={"connected_chains": connected_chains, "total_chains": len(chains)}
            ))
        elif connectivity_ratio < 0.8:
            status = "degraded"
            alerts.append(Alert(
                alert_id=f"blockchain_connectivity_warning_{int(time.time())}",
                category=MonitoringCategory.BLOCKCHAIN_CONNECTIVITY,
                severity=AlertSeverity.WARNING,
                title="Blockchain connectivity degraded",
                description=f"Only {connected_chains}/{len(chains)} blockchain connections active",
                source_component="blockchain_connectivity",
                metrics={"connected_chains": connected_chains, "total_chains": len(chains)}
            ))
        
        return HealthCheckResult(
            check_name="blockchain_connectivity",
            status=status,
            metrics=metrics,
            alerts=alerts,
            details=details
        )
    
    async def _check_price_oracle_health(self) -> HealthCheckResult:
        """Check price oracle health and price deviation detection."""
        metrics = []
        alerts = []
        status = "healthy"
        details = {}
        
        # Test assets for price validation
        test_assets = [
            "ETH_MAINNET_WETH",
            "ETH_MAINNET_USDC", 
            "ETH_MAINNET_USDT",
            "ETH_MAINNET_DAI"
        ]
        
        successful_prices = 0
        price_responses = {}
        
        for asset in test_assets:
            try:
                price = await self.oracle.get_price_usd(asset)
                if price is not None:
                    successful_prices += 1
                    price_responses[asset] = price
                    
                    # Add price metric
                    metrics.append(MonitoringMetric(
                        name="asset_price_usd",
                        value=price,
                        unit="usd",
                        tags={"asset": asset}
                    ))
                    
                    # Basic price sanity checks
                    if asset == "ETH_MAINNET_WETH":
                        if price < 1000 or price > 10000:  # ETH should be between $1k-$10k
                            alerts.append(Alert(
                                alert_id=f"price_anomaly_{asset}_{int(time.time())}",
                                category=MonitoringCategory.PRICE_DEVIATION,
                                severity=AlertSeverity.WARNING,
                                title=f"Unusual {asset} price",
                                description=f"{asset} price ${price:.2f} is outside expected range ($1000-$10000)",
                                source_component="price_oracle",
                                affected_assets=[asset],
                                metrics={"price_usd": price}
                            ))
                    elif asset in ["ETH_MAINNET_USDC", "ETH_MAINNET_USDT", "ETH_MAINNET_DAI"]:
                        if abs(price - 1.0) > 0.05:  # Stablecoins should be ~$1
                            severity = AlertSeverity.CRITICAL if abs(price - 1.0) > 0.15 else AlertSeverity.WARNING
                            alerts.append(Alert(
                                alert_id=f"stablecoin_depeg_{asset}_{int(time.time())}",
                                category=MonitoringCategory.PRICE_DEVIATION,
                                severity=severity,
                                title=f"Stablecoin depeg detected: {asset}",
                                description=f"{asset} price ${price:.4f} deviates from $1.00 peg by {abs(price - 1.0)*100:.2f}%",
                                source_component="price_oracle",
                                affected_assets=[asset],
                                metrics={"price_usd": price, "deviation_percent": abs(price - 1.0)*100}
                            ))
                    
                else:
                    price_responses[asset] = None
                    
            except Exception as e:
                price_responses[asset] = f"Error: {e}"
        
        # Overall oracle health metric
        oracle_success_rate = successful_prices / len(test_assets)
        metrics.append(MonitoringMetric(
            name="price_oracle_success_rate",
            value=oracle_success_rate,
            unit="ratio",
            threshold_warning=0.8,
            threshold_critical=0.5
        ))
        
        details = {
            "successful_prices": successful_prices,
            "total_assets_tested": len(test_assets),
            "price_responses": price_responses
        }
        
        # Generate oracle health alerts
        if oracle_success_rate < 0.5:
            status = "unhealthy"
            alerts.append(Alert(
                alert_id=f"price_oracle_critical_{int(time.time())}",
                category=MonitoringCategory.PRICE_DEVIATION,
                severity=AlertSeverity.CRITICAL,
                title="Price oracle critical failure",
                description=f"Price oracle success rate {oracle_success_rate*100:.1f}% is below critical threshold",
                source_component="price_oracle",
                metrics={"success_rate": oracle_success_rate}
            ))
        elif oracle_success_rate < 0.8:
            status = "degraded"
            alerts.append(Alert(
                alert_id=f"price_oracle_warning_{int(time.time())}",
                category=MonitoringCategory.PRICE_DEVIATION,
                severity=AlertSeverity.WARNING,
                title="Price oracle degraded performance",
                description=f"Price oracle success rate {oracle_success_rate*100:.1f}% is below warning threshold",
                source_component="price_oracle",
                metrics={"success_rate": oracle_success_rate}
            ))
        
        return HealthCheckResult(
            check_name="price_oracle_health",
            status=status,
            metrics=metrics,
            alerts=alerts,
            details=details
        )
    
    async def _check_data_quality(self) -> HealthCheckResult:
        """Check overall data quality across the system."""
        metrics = []
        alerts = []
        status = "healthy"
        details = {}
        
        # This would integrate with edge pipeline and other data sources
        # For now, provide basic data quality validation
        
        try:
            # Check Redis connectivity for data storage
            await self.redis_client.ping()
            redis_healthy = True
        except Exception as e:
            redis_healthy = False
            details["redis_error"] = str(e)
        
        metrics.append(MonitoringMetric(
            name="redis_connectivity",
            value=1.0 if redis_healthy else 0.0,
            unit="boolean"
        ))
        
        if not redis_healthy:
            status = "unhealthy"
            alerts.append(Alert(
                alert_id=f"redis_connectivity_error_{int(time.time())}",
                category=MonitoringCategory.DATA_QUALITY,
                severity=AlertSeverity.CRITICAL,
                title="Redis connectivity failure",
                description="Unable to connect to Redis for data storage",
                source_component="data_quality",
                metrics={"redis_healthy": redis_healthy}
            ))
        
        # Additional data quality checks would go here
        # - Edge state validation
        # - Data freshness checks
        # - Consistency validation
        
        details["redis_healthy"] = redis_healthy
        
        return HealthCheckResult(
            check_name="data_quality_validation",
            status=status,
            metrics=metrics,
            alerts=alerts,
            details=details
        )
    
    async def _check_liquidity_thresholds(self) -> HealthCheckResult:
        """Check liquidity thresholds across protocols."""
        metrics = []
        alerts = []
        status = "healthy"
        details = {}
        
        # This would check actual liquidity from protocol adapters
        # For now, provide framework for liquidity monitoring
        
        # Placeholder liquidity checks
        details["liquidity_monitoring"] = "Framework ready for protocol integration"
        
        return HealthCheckResult(
            check_name="liquidity_thresholds",
            status=status,
            metrics=metrics,
            alerts=alerts,
            details=details
        )
    
    async def _check_edge_state_freshness(self) -> HealthCheckResult:
        """Check edge state freshness and staleness."""
        metrics = []
        alerts = []
        status = "healthy"
        details = {}
        
        # This would check edge state timestamps from the pipeline
        # For now, provide framework for edge state monitoring
        
        details["edge_state_monitoring"] = "Framework ready for edge pipeline integration"
        
        return HealthCheckResult(
            check_name="edge_state_freshness",
            status=status,
            metrics=metrics,
            alerts=alerts,
            details=details
        )
    
    async def _check_gas_prices(self) -> HealthCheckResult:
        """Check gas prices across chains."""
        # This is handled in blockchain connectivity check
        return HealthCheckResult(
            check_name="gas_price_monitoring",
            status="healthy",
            metrics=[],
            alerts=[],
            details={"note": "Gas price monitoring integrated into blockchain connectivity check"}
        )
    
    async def _check_protocol_registry(self) -> HealthCheckResult:
        """Check protocol registry health and configuration."""
        metrics = []
        alerts = []
        status = "healthy"
        details = {}
        
        try:
            # Test protocol registry functionality
            registry_stats = production_registry.get_registry_stats()
            
            metrics.append(MonitoringMetric(
                name="protocol_registry_total_protocols",
                value=float(registry_stats["total_protocols"]),
                unit="count"
            ))
            
            metrics.append(MonitoringMetric(
                name="protocol_registry_enabled_protocols",
                value=float(registry_stats["enabled_protocols"]),
                unit="count"
            ))
            
            # Check critical protocols
            critical_protocols = ["uniswap_v3", "aave_v3"]
            missing_protocols = []
            
            for protocol_id in critical_protocols:
                protocol = production_registry.get_protocol(protocol_id)
                if not protocol or not protocol.is_enabled:
                    missing_protocols.append(protocol_id)
            
            if missing_protocols:
                status = "degraded"
                alerts.append(Alert(
                    alert_id=f"critical_protocols_missing_{int(time.time())}",
                    category=MonitoringCategory.SYSTEM_PERFORMANCE,
                    severity=AlertSeverity.WARNING,
                    title="Critical protocols missing or disabled",
                    description=f"Critical protocols not available: {', '.join(missing_protocols)}",
                    source_component="protocol_registry",
                    affected_assets=missing_protocols,
                    metrics={"missing_protocols": len(missing_protocols)}
                ))
            
            details = {
                "registry_stats": registry_stats,
                "critical_protocols_status": {
                    protocol_id: production_registry.get_protocol(protocol_id) is not None
                    for protocol_id in critical_protocols
                }
            }
            
        except Exception as e:
            status = "unhealthy"
            alerts.append(Alert(
                alert_id=f"protocol_registry_error_{int(time.time())}",
                category=MonitoringCategory.SYSTEM_PERFORMANCE,
                severity=AlertSeverity.ERROR,
                title="Protocol registry error",
                description=f"Error accessing protocol registry: {e}",
                source_component="protocol_registry"
            ))
            details["error"] = str(e)
        
        return HealthCheckResult(
            check_name="protocol_registry_health",
            status=status,
            metrics=metrics,
            alerts=alerts,
            details=details
        )
    
    async def _check_system_performance(self) -> HealthCheckResult:
        """Check overall system performance metrics."""
        metrics = []
        alerts = []
        status = "healthy"
        details = {}
        
        # Add system performance metrics
        metrics.append(MonitoringMetric(
            name="monitoring_uptime_seconds",
            value=time.time() - (self.stats.get("start_time", time.time())),
            unit="seconds"
        ))
        
        metrics.append(MonitoringMetric(
            name="total_health_checks",
            value=float(self.stats["total_checks"]),
            unit="count"
        ))
        
        if self.stats["total_checks"] > 0:
            success_rate = self.stats["healthy_checks"] / self.stats["total_checks"]
            metrics.append(MonitoringMetric(
                name="health_check_success_rate",
                value=success_rate,
                unit="ratio",
                threshold_warning=0.9,
                threshold_critical=0.8
            ))
            
            if success_rate < 0.8:
                status = "degraded"
                alerts.append(Alert(
                    alert_id=f"system_performance_degraded_{int(time.time())}",
                    category=MonitoringCategory.SYSTEM_PERFORMANCE,
                    severity=AlertSeverity.WARNING,
                    title="System performance degraded",
                    description=f"Health check success rate {success_rate*100:.1f}% below threshold",
                    source_component="system_performance",
                    metrics={"success_rate": success_rate}
                ))
        
        details = {"monitoring_stats": self.stats}
        
        return HealthCheckResult(
            check_name="system_performance",
            status=status,
            metrics=metrics,
            alerts=alerts,
            details=details
        )
    
    async def _store_metric(self, metric: MonitoringMetric) -> None:
        """Store a monitoring metric."""
        try:
            # Store in local history
            if metric.name not in self.metrics_history:
                self.metrics_history[metric.name] = []
            
            self.metrics_history[metric.name].append(metric)
            
            # Keep only last 1000 metrics per type
            if len(self.metrics_history[metric.name]) > 1000:
                self.metrics_history[metric.name] = self.metrics_history[metric.name][-1000:]
            
            # Store in Redis for persistence
            metric_key = f"monitoring:metric:{metric.name}"
            metric_data = {
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp,
                "tags": metric.tags
            }
            
            await self.redis_client.lpush(metric_key, str(metric_data))
            await self.redis_client.ltrim(metric_key, 0, 999)  # Keep last 1000
            
        except Exception as e:
            logger.error(f"Failed to store metric {metric.name}: {e}")
    
    async def _handle_alert(self, alert: Alert) -> None:
        """Handle a new alert."""
        try:
            # Check for alert cooldown
            existing_alert = self.active_alerts.get(alert.alert_id)
            if existing_alert and not existing_alert.resolved:
                cooldown_elapsed = time.time() - existing_alert.timestamp
                if cooldown_elapsed < self.monitoring_config["alert_cooldown_seconds"]:
                    return  # Skip duplicate alert within cooldown period
            
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.stats["alerts_generated"] += 1
            
            # Log alert
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }.get(alert.severity, logging.WARNING)
            
            logger.log(log_level, f"Alert {alert.severity.value}: {alert.title} - {alert.description}")
            
            # Store in Redis
            alert_key = f"monitoring:alert:{alert.alert_id}"
            alert_data = {
                "category": alert.category.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "timestamp": alert.timestamp,
                "source_component": alert.source_component,
                "affected_assets": alert.affected_assets,
                "metrics": alert.metrics
            }
            
            await self.redis_client.setex(alert_key, 86400, str(alert_data))  # 24 hour TTL
            
            # Send webhook notification if configured
            if self.alert_webhook_url:
                await self._send_webhook_alert(alert)
                
        except Exception as e:
            logger.error(f"Failed to handle alert {alert.alert_id}: {e}")
    
    async def _send_webhook_alert(self, alert: Alert) -> None:
        """Send alert notification via webhook."""
        try:
            import aiohttp
            
            webhook_payload = {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "timestamp": alert.timestamp,
                "category": alert.category.value,
                "source": alert.source_component,
                "affected_assets": alert.affected_assets,
                "metrics": alert.metrics
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.alert_webhook_url, json=webhook_payload) as response:
                    if response.status == 200:
                        logger.info(f"Alert webhook sent successfully for {alert.alert_id}")
                    else:
                        logger.error(f"Alert webhook failed with status {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    async def _process_alerts(self) -> None:
        """Process active alerts and check for auto-resolution."""
        current_time = time.time()
        resolved_alerts = []
        
        for alert_id, alert in self.active_alerts.items():
            if alert.resolved:
                continue
            
            # Auto-resolve old INFO alerts
            if (alert.severity == AlertSeverity.INFO and 
                current_time - alert.timestamp > 3600):  # 1 hour
                alert.resolved = True
                alert.resolved_at = current_time
                resolved_alerts.append(alert_id)
                self.stats["alerts_resolved"] += 1
        
        # Clean up resolved alerts
        for alert_id in resolved_alerts:
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
    
    def _update_monitoring_stats(self) -> None:
        """Update monitoring statistics."""
        self.stats["total_checks"] += len(self.health_checks)
        
        # This would be updated during actual health check execution
        # For now, we maintain basic counters
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get current system health summary."""
        health_summary = {
            "overall_status": "healthy",
            "last_check_time": time.time(),
            "active_alerts": len([a for a in self.active_alerts.values() if not a.resolved]),
            "critical_alerts": len([a for a in self.active_alerts.values() 
                                   if a.severity == AlertSeverity.CRITICAL and not a.resolved]),
            "warning_alerts": len([a for a in self.active_alerts.values() 
                                  if a.severity == AlertSeverity.WARNING and not a.resolved]),
            "monitoring_stats": self.stats,
            "component_health": {}
        }
        
        # Determine overall status
        if health_summary["critical_alerts"] > 0:
            health_summary["overall_status"] = "unhealthy"
        elif health_summary["warning_alerts"] > 0:
            health_summary["overall_status"] = "degraded"
        
        return health_summary
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring metrics."""
        metrics_summary = {}
        
        for metric_name, metric_history in self.metrics_history.items():
            if not metric_history:
                continue
            
            recent_metrics = [m for m in metric_history if time.time() - m.timestamp < 3600]  # Last hour
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                metrics_summary[metric_name] = {
                    "current_value": recent_metrics[-1].value,
                    "average_1h": statistics.mean(values),
                    "min_1h": min(values),
                    "max_1h": max(values),
                    "unit": recent_metrics[-1].unit,
                    "sample_count": len(values),
                    "last_updated": recent_metrics[-1].timestamp
                }
        
        return metrics_summary
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        active_alerts = []
        
        for alert in self.active_alerts.values():
            if not alert.resolved:
                active_alerts.append({
                    "alert_id": alert.alert_id,
                    "category": alert.category.value,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "description": alert.description,
                    "timestamp": alert.timestamp,
                    "source_component": alert.source_component,
                    "affected_assets": alert.affected_assets,
                    "metrics": alert.metrics
                })
        
        # Sort by severity and timestamp
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.ERROR: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3
        }
        
        active_alerts.sort(key=lambda x: (severity_order.get(x["severity"], 99), -x["timestamp"]))
        
        return active_alerts
    
    async def shutdown(self) -> None:
        """Shutdown the monitoring system."""
        logger.info("Shutting down production monitoring system...")
        await self.stop_monitoring()
        logger.info("Production monitoring system shutdown complete")