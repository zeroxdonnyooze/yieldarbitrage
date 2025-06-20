"""Health check endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..database import get_db
from ..cache.redis_client import health_check as redis_health_check

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "yield-arbitrage",
        "version": "0.1.0"
    }


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """Readiness check endpoint - checks if all dependencies are available."""
    checks = {
        "database": "unknown",
        "redis": "unknown", 
        "graph_engine": "not_implemented"
    }
    
    # Check database connection
    try:
        result = await db.execute(text("SELECT 1"))
        row = result.scalar()
        if row == 1:
            checks["database"] = "healthy"
        else:
            checks["database"] = "unhealthy"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"
    
    # Check Redis connection
    try:
        redis_healthy = await redis_health_check()
        checks["redis"] = "healthy" if redis_healthy else "unhealthy"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"
    
    # TODO: Check graph engine status
    
    # Determine overall status
    overall_status = "ready" if (
        checks["database"] == "healthy" and 
        checks["redis"] == "healthy"
    ) else "not_ready"
    
    return {
        "status": overall_status,
        "checks": checks
    }