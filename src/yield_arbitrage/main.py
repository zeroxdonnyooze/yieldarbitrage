"""Main FastAPI application entry point."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from yield_arbitrage.api.health import router as health_router
from yield_arbitrage.config.settings import settings
from yield_arbitrage.database import shutdown_database, startup_database
from yield_arbitrage.cache import close_redis, get_redis
from yield_arbitrage.telegram_interface.service_bot import TelegramBotService
from yield_arbitrage.graph_engine.engine import initialize_graph_engine, shutdown_graph_engine


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
    
    # Initialize Telegram bot
    print("🤖 Setting up Telegram bot...")
    try:
        telegram_service = TelegramBotService()
        await telegram_service.start()
        print("✅ Telegram bot initialized and running!")
        # Store the service in app state for shutdown
        app.state.telegram_service = telegram_service
    except Exception as e:
        print(f"⚠️  Telegram bot initialization failed: {e}")
        print("⚠️  App will continue without Telegram bot")
    
    # Initialize graph engine
    print("📊 Setting up Graph Engine...")
    try:
        graph_engine = await initialize_graph_engine()
        print("✅ Graph Engine initialized!")
        # Store in app state for shutdown
        app.state.graph_engine = graph_engine
    except Exception as e:
        print(f"⚠️  Graph Engine initialization failed: {e}")
        print("⚠️  App will continue without Graph Engine")
    
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
    
    # Stop Telegram bot
    if hasattr(app.state, 'telegram_service'):
        print("🤖 Stopping Telegram bot...")
        try:
            await app.state.telegram_service.stop()
            print("✅ Telegram bot stopped!")
        except Exception as e:
            print(f"⚠️  Error stopping Telegram bot: {e}")
    
    # Shutdown Graph Engine
    if hasattr(app.state, 'graph_engine'):
        print("📊 Shutting down Graph Engine...")
        try:
            await shutdown_graph_engine()
            print("✅ Graph Engine stopped!")
        except Exception as e:
            print(f"⚠️  Error stopping Graph Engine: {e}")
    
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