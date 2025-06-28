"""Execution and simulation package for yield arbitrage system."""
from .hybrid_simulator import (
    HybridPathSimulator,
    SimulationMode,
    SimulationResult,
    SimulatorConfig,
    TenderlyConfig,
)
from .asset_oracle import (
    AssetOracleBase,
    CachedAssetOracle,
    CoingeckoOracle,
    CompositeOracle,
    AssetPrice,
)
from .edge_validator import (
    EdgeValidator,
    EdgeValidationResult,
)
from .tenderly_client import (
    TenderlyClient,
    TenderlyTransaction,
    TenderlySimulationResult,
    TenderlyFork,
    TenderlyNetworkId,
    TenderlyAPIError,
    TenderlyAuthError,
    TenderlyRateLimitError,
    TenderlyNetworkError,
)
from .transaction_builder import (
    TransactionBuilder,
    TokenInfo,
    TokenStandard,
    SwapParams,
)
from .execution_engine import (
    ExecutionEngine,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    PreFlightCheck,
    PreFlightCheckResult,
)
from .logged_execution_engine import (
    LoggedExecutionEngine,
    create_logged_execution_engine,
)

# Import monitoring components (conditional to handle missing dependencies)
try:
    from ..monitoring.position_monitor import (
        PositionMonitor,
        PositionType,
        RiskLevel, 
        AlertSeverity,
        PositionAlert,
        MonitoringConfig,
    )
    
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

__all__ = [
    # Core simulator
    "HybridPathSimulator",
    "SimulationMode", 
    "SimulationResult",
    "SimulatorConfig",
    "TenderlyConfig",
    # Asset oracle
    "AssetOracleBase",
    "CachedAssetOracle",
    "CoingeckoOracle",
    "CompositeOracle",
    "AssetPrice",
    # Edge validator
    "EdgeValidator",
    "EdgeValidationResult",
    # Tenderly client
    "TenderlyClient",
    "TenderlyTransaction",
    "TenderlySimulationResult",
    "TenderlyFork",
    "TenderlyNetworkId",
    "TenderlyAPIError",
    "TenderlyAuthError",
    "TenderlyRateLimitError",
    "TenderlyNetworkError",
    # Transaction builder
    "TransactionBuilder",
    "TokenInfo",
    "TokenStandard",
    "SwapParams",
    # Execution engine
    "ExecutionEngine",
    "ExecutionContext",
    "ExecutionResult",
    "ExecutionStatus",
    "PreFlightCheck",
    "PreFlightCheckResult",
    # Logged execution engine
    "LoggedExecutionEngine",
    "create_logged_execution_engine",
]

# Add monitoring components to exports if available
if MONITORING_AVAILABLE:
    __all__.extend([
        "PositionMonitor",
        "PositionType",
        "RiskLevel",
        "AlertSeverity", 
        "PositionAlert",
        "MonitoringConfig",
    ])