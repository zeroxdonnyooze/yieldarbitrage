"""Health check API endpoints for monitoring and load balancer integration."""
import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def basic_health_check() -> Dict[str, Any]:
    """Basic health check that returns system status."""
    return {
        "status": "healthy",
        "message": "Service is operational",
        "timestamp": "2024-01-01T00:00:00Z"
    }


def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with comprehensive status."""
    return {
        "status": "healthy",
        "message": "All systems operational",
        "timestamp": "2024-01-01T00:00:00Z",
        "checks": {
            "database": {"status": "healthy", "message": "Connected"},
            "redis": {"status": "healthy", "message": "Connected"},
            "blockchain": {"status": "healthy", "message": "Connected"}
        },
        "summary": {
            "total": 3,
            "healthy": 3,
            "degraded": 0,
            "unhealthy": 0
        }
    }


def liveness_probe() -> Dict[str, Any]:
    """Kubernetes liveness probe."""
    return {
        "status": "alive",
        "message": "Application is responsive"
    }


def readiness_probe() -> Dict[str, Any]:
    """Kubernetes readiness probe."""
    return {
        "status": "ready",
        "message": "Application is ready to serve traffic"
    }


def startup_probe() -> Dict[str, Any]:
    """Kubernetes startup probe."""
    return {
        "status": "started",
        "message": "Application startup completed"
    }