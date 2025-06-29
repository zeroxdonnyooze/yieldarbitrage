"""Enhanced database connection and session management with comprehensive pooling and monitoring."""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DisconnectionError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import Pool

from ..config.settings import settings

# Setup logging
logger = logging.getLogger(__name__)

# Create the declarative base
Base = declarative_base()

def _get_db_config() -> dict:
    """Get database configuration from settings."""
    return {
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_timeout": settings.db_pool_timeout,
        "pool_recycle": settings.db_pool_recycle,
        "pool_pre_ping": True,        # Always validate connections
        "connect_args": {
            "server_settings": {
                "application_name": "yield_arbitrage_system",
                "jit": "off",         # Disable JIT for predictable performance
            },
            "command_timeout": 30,     # Command timeout in seconds
            "prepared_statement_cache_size": 100,  # Prepared statement cache
        }
    }

# Global engine instance
engine: Optional[object] = None
AsyncSessionLocal: Optional[object] = None


def _get_database_url() -> str:
    """Get database URL with fallback."""
    if settings.database_url:
        return settings.database_url
    
    # Fallback URL for development
    fallback_url = "postgresql+asyncpg://user:pass@localhost/yieldarbitrage"
    logger.warning(f"DATABASE_URL not set, using fallback: {fallback_url}")
    return fallback_url


@event.listens_for(Pool, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set connection parameters for optimal performance."""
    if "postgresql" in str(dbapi_connection):
        # These are handled via connect_args for PostgreSQL
        pass


@event.listens_for(Pool, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log connection checkout for monitoring."""
    logger.debug(f"Connection checked out: {id(dbapi_connection)}")


@event.listens_for(Pool, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log connection checkin for monitoring."""
    logger.debug(f"Connection checked in: {id(dbapi_connection)}")


async def init_database() -> None:
    """Initialize database engine and session factory."""
    global engine, AsyncSessionLocal
    
    if engine is not None:
        logger.warning("Database already initialized")
        return
    
    database_url = _get_database_url()
    
    try:
        # Create async engine with enhanced configuration
        db_config = _get_db_config()
        engine = create_async_engine(
            database_url,
            echo=settings.debug,
            echo_pool=settings.debug,  # Pool activity logging
            future=True,               # Use SQLAlchemy 2.0 style
            **db_config
        )
        
        # Create async session factory
        AsyncSessionLocal = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
        
        # Test the connection
        await health_check()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def health_check() -> bool:
    """Check database health and connectivity."""
    if engine is None:
        logger.error("Database engine not initialized")
        return False
    
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        logger.debug("Database health check passed")
        return True
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def get_pool_status() -> dict:
    """Get detailed connection pool status."""
    if engine is None:
        return {"error": "Engine not initialized"}
    
    pool = engine.pool
    status = {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total_connections": pool.size() + pool.overflow(),
        "utilization": (pool.checkedout() / (pool.size() + pool.overflow())) * 100
        if (pool.size() + pool.overflow()) > 0 else 0
    }
    
    # Add invalid count if available (not all pool types support this)
    if hasattr(pool, 'invalid'):
        status["invalid"] = pool.invalid()
    
    return status


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Enhanced session context manager with retry logic and monitoring."""
    if AsyncSessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    retry_count = 0
    max_retries = 3
    retry_delay = 1.0
    
    while retry_count <= max_retries:
        session = None
        try:
            session = AsyncSessionLocal()
            start_time = time.time()
            
            yield session
            
            # Commit if no explicit transaction management
            if session.in_transaction():
                await session.commit()
            
            # Log session duration for monitoring
            duration = time.time() - start_time
            if duration > 1.0:  # Log slow sessions
                logger.warning(f"Slow database session: {duration:.2f}s")
            
            break  # Success, exit retry loop
            
        except (DisconnectionError, OperationalError) as e:
            logger.warning(f"Database connection error (attempt {retry_count + 1}): {e}")
            
            if session:
                await session.rollback()
                await session.close()
            
            retry_count += 1
            if retry_count <= max_retries:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Database connection failed after {max_retries} retries")
                raise
                
        except Exception as e:
            logger.error(f"Database session error: {e}")
            if session:
                await session.rollback()
            raise
            
        finally:
            if session:
                await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency to provide database sessions."""
    async with get_session() as session:
        yield session


@asynccontextmanager
async def transaction() -> AsyncGenerator[AsyncSession, None]:
    """Explicit transaction context manager."""
    async with get_session() as session:
        try:
            await session.begin()
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def execute_with_retry(query, max_retries: int = 3) -> object:
    """Execute a query with retry logic."""
    retry_count = 0
    retry_delay = 1.0
    
    while retry_count <= max_retries:
        try:
            async with get_session() as session:
                result = await session.execute(query)
                return result
                
        except (DisconnectionError, OperationalError) as e:
            logger.warning(f"Query execution error (attempt {retry_count + 1}): {e}")
            retry_count += 1
            
            if retry_count <= max_retries:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"Query execution failed after {max_retries} retries")
                raise


async def create_tables() -> None:
    """Create all database tables."""
    if engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


async def drop_tables() -> None:
    """Drop all database tables (for testing)."""
    if engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        raise


async def close_db() -> None:
    """Close database connections and cleanup."""
    global engine, AsyncSessionLocal
    
    if engine is not None:
        try:
            await engine.dispose()
            logger.info("Database connections closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
        finally:
            engine = None
            AsyncSessionLocal = None
    else:
        logger.warning("Database engine was not initialized")


# Database lifecycle management for FastAPI
async def startup_database():
    """Database startup routine for application initialization."""
    try:
        await init_database()
        
        # Verify tables exist, create if not
        async with engine.begin() as conn:
            # Check if tables exist
            result = await conn.execute(
                text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'executed_paths')")
            )
            tables_exist = result.scalar()
            
            if not tables_exist:
                logger.info("Tables not found, creating database schema...")
                await create_tables()
        
        # Log pool status
        pool_status = await get_pool_status()
        logger.info(f"Database startup complete. Pool status: {pool_status}")
        
    except Exception as e:
        logger.error(f"Database startup failed: {e}")
        raise


async def shutdown_database():
    """Database shutdown routine for application cleanup."""
    try:
        # Log final pool status
        if engine:
            pool_status = await get_pool_status()
            logger.info(f"Database shutdown initiated. Final pool status: {pool_status}")
        
        await close_db()
        
    except Exception as e:
        logger.error(f"Database shutdown error: {e}")
        raise