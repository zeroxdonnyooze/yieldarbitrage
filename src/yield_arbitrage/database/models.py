"""SQLAlchemy ORM models for the yield arbitrage system."""
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import Boolean, DateTime, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .connection import Base


class SimulatedExecution(Base):
    """Model for storing simulated execution attempts and their results."""
    
    __tablename__ = "simulated_executions"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    
    # Execution identification
    execution_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
        index=True,
        comment="Unique execution identifier from ExecutionEngine"
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        comment="Session identifier for grouping related executions"
    )
    
    # Path identification
    path_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="Identifier for the arbitrage path"
    )
    path_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA256 hash of the path for deduplication"
    )
    
    # Execution parameters
    chain_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
        comment="Target blockchain chain ID"
    )
    chain_name: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        comment="Blockchain network name"
    )
    initial_amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=78, scale=18),
        nullable=False,
        comment="Initial amount for execution in wei/smallest unit"
    )
    start_asset_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Starting asset identifier"
    )
    
    # Path structure
    edge_ids: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        comment="List of edge IDs that formed this path"
    )
    edge_types: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        comment="List of edge types corresponding to edge_ids"
    )
    protocols: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        comment="List of protocols used in this path"
    )
    
    # Execution status
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        comment="Execution status (pending, completed, failed, cancelled)"
    )
    success: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether execution was successful"
    )
    
    # Pre-flight check results
    pre_flight_passed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether pre-flight checks passed"
    )
    pre_flight_warnings: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of pre-flight warnings"
    )
    pre_flight_failures: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of pre-flight failures"
    )
    pre_flight_details: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
        comment="Detailed pre-flight check results"
    )
    
    # Simulation results
    simulation_mode: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
        comment="Simulation mode used (basic, tenderly, hybrid, local)"
    )
    simulation_success: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        comment="Whether simulation was successful"
    )
    simulation_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Time taken for simulation in milliseconds"
    )
    
    # Profitability metrics
    predicted_profit_usd: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=8),
        nullable=True,
        comment="Predicted profit in USD from simulation"
    )
    predicted_profit_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=8, scale=6),
        nullable=True,
        comment="Predicted profit as percentage"
    )
    estimated_gas_cost_usd: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=8),
        nullable=True,
        comment="Estimated gas cost in USD"
    )
    estimated_output_amount: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=78, scale=18),
        nullable=True,
        comment="Estimated output amount from simulation"
    )
    
    # Risk assessment
    mev_risk_level: Mapped[Optional[str]] = mapped_column(
        String(16),
        nullable=True,
        comment="MEV risk level (low, medium, high, critical)"
    )
    position_size_usd: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=True,
        comment="Position size in USD"
    )
    max_slippage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=8, scale=6),
        nullable=True,
        comment="Maximum allowed slippage"
    )
    
    # Execution route and MEV protection
    execution_method: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
        comment="Selected execution method (public, flashbots, private_node, etc.)"
    )
    mev_protected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether MEV protection was used"
    )
    
    # Transaction details (if executed)
    transaction_hashes: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
        comment="List of transaction hashes if execution proceeded"
    )
    actual_gas_used: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Actual gas used if execution completed"
    )
    actual_gas_cost_usd: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=8),
        nullable=True,
        comment="Actual gas cost in USD"
    )
    
    # Position tracking
    position_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="Position ID in DeltaTracker if created"
    )
    delta_exposure: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Market delta exposure for this execution"
    )
    
    # Failure information
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if execution failed"
    )
    failed_at_step: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="Step where execution failed"
    )
    revert_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Revert reason from simulation or execution"
    )
    
    # Market context
    eth_price_usd: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=8),
        nullable=True,
        comment="ETH price in USD at time of execution"
    )
    gas_price_gwei: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=9),
        nullable=True,
        comment="Gas price in Gwei at time of execution"
    )
    block_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Block number at time of execution"
    )
    
    # Performance metrics
    execution_time_seconds: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=10, scale=3),
        nullable=True,
        comment="Total execution time in seconds"
    )
    pre_flight_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Time spent on pre-flight checks in milliseconds"
    )
    
    # Additional metadata
    warnings: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
        comment="List of warnings generated during execution"
    )
    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional metadata about the execution attempt"
    )
    
    # User context
    user_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        comment="User identifier if applicable"
    )
    api_key_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="API key identifier for tracking usage"
    )
    request_source: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="Source of the execution request (api, ui, automated)"
    )
    
    # Timestamps
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="When execution was started"
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When execution was completed or failed"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Record creation timestamp"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Record last update timestamp"
    )

    def __repr__(self) -> str:
        return (
            f"<SimulatedExecution(id={self.id}, execution_id={self.execution_id}, "
            f"status={self.status}, success={self.success})>"
        )


class ExecutedPath(Base):
    """Model for storing executed arbitrage paths and their results."""
    
    __tablename__ = "executed_paths"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    
    # Path identification
    path_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA256 hash of the path for deduplication"
    )
    
    # Execution details
    transaction_hash: Mapped[str] = mapped_column(
        String(66),
        nullable=False,
        unique=True,
        index=True,
        comment="Blockchain transaction hash"
    )
    block_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
        comment="Block number where transaction was included"
    )
    chain_name: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        comment="Blockchain network name (e.g., ethereum, arbitrum)"
    )
    
    # Path structure
    edge_ids: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        comment="List of edge IDs that formed this path"
    )
    edge_types: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        comment="List of edge types corresponding to edge_ids"
    )
    protocols: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        comment="List of protocols used in this path"
    )
    
    # Financial results
    input_amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=78, scale=18),
        nullable=False,
        comment="Input amount in wei/smallest unit"
    )
    output_amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=78, scale=18),
        nullable=False,
        comment="Output amount in wei/smallest unit"
    )
    input_token: Mapped[str] = mapped_column(
        String(42),
        nullable=False,
        comment="Input token contract address"
    )
    output_token: Mapped[str] = mapped_column(
        String(42),
        nullable=False,
        comment="Output token contract address"
    )
    
    # Profit analysis
    profit_usd: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=8),
        nullable=False,
        comment="Profit in USD"
    )
    profit_percentage: Mapped[Decimal] = mapped_column(
        Numeric(precision=8, scale=6),
        nullable=False,
        comment="Profit as percentage (0.05 = 5%)"
    )
    effective_rate: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=8),
        nullable=False,
        comment="Effective exchange rate (output/input)"
    )
    
    # Execution costs
    gas_used: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Gas consumed by the transaction"
    )
    gas_price_gwei: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=9),
        nullable=False,
        comment="Gas price in Gwei"
    )
    gas_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=8),
        nullable=False,
        comment="Total gas cost in USD"
    )
    
    # Execution properties
    execution_time_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Time taken for execution in milliseconds"
    )
    slippage_actual: Mapped[Decimal] = mapped_column(
        Numeric(precision=8, scale=6),
        nullable=False,
        comment="Actual slippage experienced (0.01 = 1%)"
    )
    mev_protected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether transaction was sent via private mempool"
    )
    flash_loan_used: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether flash loan was used for capital"
    )
    
    # Market conditions at execution
    market_volatility: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=8, scale=6),
        nullable=True,
        comment="Market volatility at time of execution"
    )
    total_liquidity_usd: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=True,
        comment="Total liquidity across all edges in path"
    )
    
    # ML features and scores
    ml_confidence_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=5, scale=4),
        nullable=True,
        comment="ML model confidence score (0-1)"
    )
    predicted_profit: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=8),
        nullable=True,
        comment="ML predicted profit in USD"
    )
    prediction_error: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=8),
        nullable=True,
        comment="Difference between predicted and actual profit"
    )
    
    # Risk metrics
    max_exposure_delta: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=8),
        nullable=True,
        comment="Maximum market exposure during path execution"
    )
    risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=5, scale=4),
        nullable=True,
        comment="Calculated risk score for this path (0-1)"
    )
    
    # Status and metadata
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="success",
        comment="Execution status: success, failed, reverted"
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if execution failed"
    )
    
    # Additional metadata
    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional metadata about execution"
    )
    
    # Timestamps
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="When the path was discovered"
    )
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="When the path was executed"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Record creation timestamp"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Record last update timestamp"
    )

    def __repr__(self) -> str:
        return (
            f"<ExecutedPath(id={self.id}, tx={self.transaction_hash[:8]}..., "
            f"profit=${self.profit_usd}, status={self.status})>"
        )


class TokenMetadata(Base):
    """Model for storing token metadata and pricing information."""
    
    __tablename__ = "token_metadata"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    
    # Token identification
    asset_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
        index=True,
        comment="Unique asset identifier (e.g., ETH_MAINNET_0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2)"
    )
    chain_name: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        comment="Blockchain network name"
    )
    contract_address: Mapped[str] = mapped_column(
        String(42),
        nullable=False,
        index=True,
        comment="Token contract address"
    )
    
    # Basic token information
    symbol: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        comment="Token symbol (e.g., WETH, USDC)"
    )
    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Full token name"
    )
    decimals: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of decimal places"
    )
    
    # Token type and properties
    token_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="ERC20",
        comment="Token standard (ERC20, ERC721, etc.)"
    )
    is_stable: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether this is a stablecoin"
    )
    is_wrapped: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether this is a wrapped token"
    )
    is_yield_bearing: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether this token earns yield over time"
    )
    
    # Pricing information
    price_usd: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=8),
        nullable=True,
        comment="Current price in USD"
    )
    price_eth: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=28, scale=18),
        nullable=True,
        comment="Current price in ETH"
    )
    market_cap_usd: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=True,
        comment="Market capitalization in USD"
    )
    
    # Trading metrics
    volume_24h_usd: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=True,
        comment="24-hour trading volume in USD"
    )
    liquidity_usd: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=True,
        comment="Total liquidity across all venues in USD"
    )
    volatility_24h: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=8, scale=6),
        nullable=True,
        comment="24-hour price volatility (standard deviation)"
    )
    
    # Supply information
    total_supply: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=78, scale=18),
        nullable=True,
        comment="Total token supply in wei/smallest unit"
    )
    circulating_supply: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=78, scale=18),
        nullable=True,
        comment="Circulating token supply"
    )
    
    # Risk and quality metrics
    security_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=5, scale=4),
        nullable=True,
        comment="Security assessment score (0-1)"
    )
    liquidity_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=5, scale=4),
        nullable=True,
        comment="Liquidity quality score (0-1)"
    )
    reliability_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=5, scale=4),
        nullable=True,
        comment="Price oracle reliability score (0-1)"
    )
    
    # Protocol relationships
    protocols: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
        comment="List of protocols that support this token"
    )
    primary_venue: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="Primary trading venue for this token"
    )
    
    # Yield information (for yield-bearing tokens)
    base_apr: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=8, scale=6),
        nullable=True,
        comment="Base APR for yield-bearing tokens"
    )
    reward_tokens: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
        comment="List of reward tokens for yield-bearing assets"
    )
    
    # Status and flags
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether token is actively tracked"
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether token has been verified as legitimate"
    )
    risk_level: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="medium",
        comment="Risk assessment: low, medium, high"
    )
    
    # External identifiers
    coingecko_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="CoinGecko API identifier"
    )
    coinmarketcap_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="CoinMarketCap identifier"
    )
    
    # Additional metadata
    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional token metadata"
    )
    tags: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
        comment="Tags for categorization (e.g., ['defi', 'stablecoin'])"
    )
    
    # Data freshness
    price_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When price data was last updated"
    )
    metadata_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When metadata was last updated"
    )
    
    # Timestamps
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="When token was first discovered"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Record creation timestamp"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Record last update timestamp"
    )

    def __repr__(self) -> str:
        return (
            f"<TokenMetadata(asset_id={self.asset_id}, symbol={self.symbol}, "
            f"price_usd={self.price_usd}, active={self.is_active})>"
        )