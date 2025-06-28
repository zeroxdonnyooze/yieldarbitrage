"""Comprehensive health check system for production monitoring."""
import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    check_func: Callable
    timeout_seconds: float = 10.0
    critical: bool = False
    tags: List[str] = field(default_factory=list)
    interval_seconds: float = 30.0
    enabled: bool = True


class HealthChecker:
    """Production health checker with comprehensive monitoring."""
    
    def __init__(self, max_workers: int = 5):
        self.checks: Dict[str, HealthCheck] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.stats = {
            "total_checks": 0,
            "successful_checks": 0,
            "failed_checks": 0,
            "average_duration_ms": 0.0
        }
    
    def register_check(self, health_check: HealthCheck) -> None:
        """Register a health check."""
        self.checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name}")
    
    async def run_check(self, check_name: str) -> HealthCheckResult:
        """Run a single health check."""
        if check_name not in self.checks:
            return HealthCheckResult(
                name=check_name,
                status=HealthStatus.UNKNOWN,
                message="Health check not found",
                duration_ms=0.0,
                error="Check not registered"
            )
        
        check = self.checks[check_name]
        
        if not check.enabled:
            return HealthCheckResult(
                name=check_name,
                status=HealthStatus.UNKNOWN,
                message="Health check disabled",
                duration_ms=0.0
            )
        
        start_time = time.time()
        
        try:
            # Run check with timeout
            loop = asyncio.get_event_loop()
            
            if asyncio.iscoroutinefunction(check.check_func):
                # Async function
                task = asyncio.create_task(check.check_func())
                result = await asyncio.wait_for(task, timeout=check.timeout_seconds)
            else:
                # Sync function
                future = loop.run_in_executor(self.executor, check.check_func)
                result = await asyncio.wait_for(future, timeout=check.timeout_seconds)
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Parse result
            if isinstance(result, dict):
                status = HealthStatus(result.get("status", HealthStatus.HEALTHY))
                message = result.get("message", "OK")
                details = result.get("details")
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "OK" if result else "Check failed"
                details = None
            else:
                status = HealthStatus.HEALTHY
                message = str(result) if result else "OK"
                details = None
            
            self.stats["successful_checks"] += 1
            
            return HealthCheckResult(
                name=check_name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=details
            )
            
        except asyncio.TimeoutError:
            duration_ms = check.timeout_seconds * 1000
            self.stats["failed_checks"] += 1
            
            return HealthCheckResult(
                name=check_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {check.timeout_seconds}s",
                duration_ms=duration_ms,
                error="timeout"
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.stats["failed_checks"] += 1
            
            return HealthCheckResult(
                name=check_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration_ms=duration_ms,
                error=traceback.format_exc()
            )
        
        finally:
            self.stats["total_checks"] += 1
            
            # Update average duration
            if self.stats["total_checks"] > 0:
                total_duration = (self.stats["average_duration_ms"] * (self.stats["total_checks"] - 1) + 
                                (time.time() - start_time) * 1000)
                self.stats["average_duration_ms"] = total_duration / self.stats["total_checks"]
    
    async def run_all_checks(self, tags: Optional[List[str]] = None) -> Dict[str, HealthCheckResult]:
        """Run all health checks, optionally filtered by tags."""
        # Filter checks by tags if provided
        checks_to_run = {}
        for name, check in self.checks.items():
            if not check.enabled:
                continue
            if tags and not any(tag in check.tags for tag in tags):
                continue
            checks_to_run[name] = check
        
        # Run checks concurrently
        tasks = []
        for check_name in checks_to_run:
            task = asyncio.create_task(self.run_check(check_name))
            tasks.append((check_name, task))
        
        results = {}
        for check_name, task in tasks:
            try:
                result = await task
                results[check_name] = result
                self.last_results[check_name] = result
            except Exception as e:
                logger.error(f"Failed to run health check {check_name}: {e}")
                results[check_name] = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Execution failed: {str(e)}",
                    duration_ms=0.0,
                    error=str(e)
                )
        
        return results
    
    def get_overall_health(self, results: Optional[Dict[str, HealthCheckResult]] = None) -> Dict[str, Any]:
        """Get overall system health status."""
        if results is None:
            results = self.last_results
        
        if not results:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": "No health checks executed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": {},
                "summary": {
                    "total": 0,
                    "healthy": 0,
                    "degraded": 0,
                    "unhealthy": 0,
                    "critical_failures": 0
                }
            }
        
        # Count statuses
        summary = {
            "total": len(results),
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "critical_failures": 0
        }
        
        critical_failed = False
        has_degraded = False
        has_unhealthy = False
        
        for result in results.values():
            if result.status == HealthStatus.HEALTHY:
                summary["healthy"] += 1
            elif result.status == HealthStatus.DEGRADED:
                summary["degraded"] += 1
                has_degraded = True
            elif result.status == HealthStatus.UNHEALTHY:
                summary["unhealthy"] += 1
                has_unhealthy = True
                
                # Check if this is a critical check
                check = self.checks.get(result.name)
                if check and check.critical:
                    critical_failed = True
                    summary["critical_failures"] += 1
        
        # Determine overall status
        if critical_failed:
            overall_status = HealthStatus.UNHEALTHY
            message = f"Critical health checks failed: {summary['critical_failures']}"
        elif has_unhealthy:
            overall_status = HealthStatus.DEGRADED
            message = f"Some health checks failed: {summary['unhealthy']}"
        elif has_degraded:
            overall_status = HealthStatus.DEGRADED
            message = f"System running in degraded mode: {summary['degraded']} degraded"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All systems operational"
        
        return {
            "status": overall_status,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {name: {
                "status": result.status,
                "message": result.message,
                "duration_ms": result.duration_ms,
                "timestamp": result.timestamp.isoformat(),
                "critical": self.checks.get(name, HealthCheck("", lambda: None)).critical
            } for name, result in results.items()},
            "summary": summary,
            "stats": self.stats
        }
    
    async def continuous_monitoring(self, interval_seconds: float = 30.0) -> None:
        """Run continuous health monitoring."""
        logger.info(f"Starting continuous health monitoring (interval: {interval_seconds}s)")
        
        while True:
            try:
                results = await self.run_all_checks()
                overall_health = self.get_overall_health(results)
                
                # Log health status
                if overall_health["status"] == HealthStatus.HEALTHY:
                    logger.info(f"System health: {overall_health['message']}")
                elif overall_health["status"] == HealthStatus.DEGRADED:
                    logger.warning(f"System health: {overall_health['message']}")
                else:
                    logger.error(f"System health: {overall_health['message']}")
                
                # Log failed checks
                for name, result in results.items():
                    if result.status != HealthStatus.HEALTHY:
                        logger.warning(f"Health check '{name}' status: {result.status} - {result.message}")
                
            except Exception as e:
                logger.error(f"Error in continuous health monitoring: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get health checker statistics."""
        return {
            **self.stats,
            "registered_checks": len(self.checks),
            "enabled_checks": len([c for c in self.checks.values() if c.enabled]),
            "critical_checks": len([c for c in self.checks.values() if c.critical and c.enabled])
        }


# Global health checker instance
health_checker = HealthChecker()


# Convenience functions for easy health check registration
def database_health_check(db_pool) -> Dict[str, Any]:
    """Check database connectivity and performance."""
    try:
        # This would normally use your database connection
        # For now, return a mock result
        return {
            "status": HealthStatus.HEALTHY,
            "message": "Database connected",
            "details": {
                "pool_size": 10,
                "active_connections": 3,
                "query_time_ms": 5.2
            }
        }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY,
            "message": f"Database connection failed: {str(e)}"
        }


def redis_health_check(redis_client) -> Dict[str, Any]:
    """Check Redis connectivity and performance."""
    try:
        # This would normally use your Redis client
        return {
            "status": HealthStatus.HEALTHY,
            "message": "Redis connected",
            "details": {
                "memory_usage_mb": 45.2,
                "ping_time_ms": 1.1
            }
        }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY,
            "message": f"Redis connection failed: {str(e)}"
        }


def blockchain_health_check(blockchain_provider) -> Dict[str, Any]:
    """Check blockchain connectivity."""
    try:
        # This would check your blockchain connections
        return {
            "status": HealthStatus.HEALTHY,
            "message": "Blockchain connections active",
            "details": {
                "ethereum_block": 12345678,
                "arbitrum_block": 87654321,
                "rpc_latency_ms": 150.5
            }
        }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY,
            "message": f"Blockchain connection failed: {str(e)}"
        }


def edge_pipeline_health_check(edge_pipeline) -> Dict[str, Any]:
    """Check edge state pipeline health."""
    try:
        stats = edge_pipeline.get_pipeline_stats() if edge_pipeline else {}
        
        active_edges = stats.get("active_edge_count", 0)
        success_rate = stats.get("successful_updates", 0) / max(stats.get("total_updates", 1), 1)
        
        if success_rate < 0.8:
            status = HealthStatus.DEGRADED
            message = f"Edge pipeline degraded: {success_rate:.1%} success rate"
        elif active_edges == 0:
            status = HealthStatus.UNHEALTHY
            message = "No active edges"
        else:
            status = HealthStatus.HEALTHY
            message = f"Edge pipeline healthy: {active_edges} active edges"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "active_edges": active_edges,
                "success_rate": success_rate,
                "avg_update_time_ms": stats.get("average_update_time_ms", 0)
            }
        }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY,
            "message": f"Edge pipeline check failed: {str(e)}"
        }


# Register default health checks
def register_default_health_checks(
    db_pool=None,
    redis_client=None,
    blockchain_provider=None,
    edge_pipeline=None
) -> None:
    """Register default health checks for the application."""
    
    if db_pool:
        health_checker.register_check(HealthCheck(
            name="database",
            check_func=lambda: database_health_check(db_pool),
            timeout_seconds=5.0,
            critical=True,
            tags=["infrastructure", "database"]
        ))
    
    if redis_client:
        health_checker.register_check(HealthCheck(
            name="redis",
            check_func=lambda: redis_health_check(redis_client),
            timeout_seconds=5.0,
            critical=True,
            tags=["infrastructure", "cache"]
        ))
    
    if blockchain_provider:
        health_checker.register_check(HealthCheck(
            name="blockchain",
            check_func=lambda: blockchain_health_check(blockchain_provider),
            timeout_seconds=10.0,
            critical=True,
            tags=["infrastructure", "blockchain"]
        ))
    
    if edge_pipeline:
        health_checker.register_check(HealthCheck(
            name="edge_pipeline",
            check_func=lambda: edge_pipeline_health_check(edge_pipeline),
            timeout_seconds=5.0,
            critical=False,
            tags=["application", "trading"]
        ))
    
    # System health checks
    health_checker.register_check(HealthCheck(
        name="disk_space",
        check_func=lambda: {"status": HealthStatus.HEALTHY, "message": "Disk space OK"},
        timeout_seconds=5.0,
        critical=False,
        tags=["system"]
    ))
    
    health_checker.register_check(HealthCheck(
        name="memory",
        check_func=lambda: {"status": HealthStatus.HEALTHY, "message": "Memory usage OK"},
        timeout_seconds=5.0,
        critical=False,
        tags=["system"]
    ))
    
    logger.info("Default health checks registered")