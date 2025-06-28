"""Main FastAPI application entry point."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from yield_arbitrage.api.health import router as health_router
from yield_arbitrage.config.settings import settings
from yield_arbitrage.database import shutdown_database, startup_database
from yield_arbitrage.cache import close_redis, get_redis


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan - startup and shutdown events."""
    # Startup
    print("🚀 Starting Yield Arbitrage System...")
    
    # Initialize database
    print("📊 Setting up database...")
    await startup_database()
    print("✅ Database initialized!")
    
    # Initialize Redis
    print("📦 Setting up Redis...")
    await get_redis()  # This will initialize the Redis connection
    print("✅ Redis initialized!")
    
    # TODO: Initialize graph engine
    # await initialize_graph_engine()
    
    print("✅ System startup complete!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Yield Arbitrage System...")
    
    # Close database connections
    print("📊 Closing database connections...")
    await shutdown_database()
    print("✅ Database closed!")
    
    # Close Redis connections
    print("📦 Closing Redis connections...")
    await close_redis()
    print("✅ Redis closed!")
    
    print("✅ System shutdown complete!")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Yield Arbitrage API",
        description="AI-driven DeFi yield arbitrage through graph-based strategy discovery",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Include API routers
    app.include_router(health_router, tags=["health"])
    
    # TODO: Add additional API routers
    # app.include_router(graph_router, prefix="/api/v1/graph", tags=["graph"])
    # app.include_router(opportunities_router, prefix="/api/v1/opportunities", tags=["opportunities"])
    # app.include_router(execution_router, prefix="/api/v1/execution", tags=["execution"])
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "yield_arbitrage.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
    )