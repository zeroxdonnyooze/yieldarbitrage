"""Database package with enhanced connection management."""
from .connection import (
    Base,
    close_db,
    create_tables,
    drop_tables,
    execute_with_retry,
    get_db,
    get_pool_status,
    get_session,
    health_check,
    init_database,
    shutdown_database,
    startup_database,
    transaction,
)
from .models import ExecutedPath, TokenMetadata, SimulatedExecution
from .execution_logger import (
    ExecutionLogger,
    get_execution_logger,
    log_execution_start,
    log_simulation_results,
    log_execution_completion,
)

__all__ = [
    # Connection management
    "Base",
    "close_db", 
    "create_tables",
    "drop_tables",
    "execute_with_retry",
    "get_db",
    "get_pool_status",
    "get_session",
    "health_check",
    "init_database",
    "shutdown_database",
    "startup_database",
    "transaction",
    # Models
    "ExecutedPath",
    "TokenMetadata",
    "SimulatedExecution",
    # Execution logging
    "ExecutionLogger",
    "get_execution_logger",
    "log_execution_start",
    "log_simulation_results", 
    "log_execution_completion",
]