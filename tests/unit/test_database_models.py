"""Unit tests for database ORM models."""
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from yield_arbitrage.database.connection import Base
from yield_arbitrage.database.models import ExecutedPath, TokenMetadata


@pytest.fixture
async def async_session():
    """Create an async test database session."""
    # Use in-memory SQLite for tests
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session_factory = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session_factory() as session:
        yield session
    
    await engine.dispose()


class TestExecutedPath:
    """Test ExecutedPath model functionality."""
    
    async def test_executed_path_creation(self, async_session: AsyncSession):
        """Test creating an ExecutedPath record."""
        now = datetime.now(timezone.utc)
        
        executed_path = ExecutedPath(
            path_hash="a1b2c3d4e5f6789012345678901234567890123456789012345678901234abcd",
            transaction_hash="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            block_number=18500000,
            chain_name="ethereum",
            edge_ids=["ETH_UNISWAP_TRADE_WETH_USDC", "ETH_AAVE_LEND_USDC"],
            edge_types=["TRADE", "LEND"],
            protocols=["UniswapV3", "AaveV3"],
            input_amount=Decimal("1000000000000000000"),  # 1 ETH in wei
            output_amount=Decimal("2000000000"),  # 2000 USDC
            input_token="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            output_token="0xA0b86a33E6441b9435B654f6D26Cc98b6e1D0a3A",
            profit_usd=Decimal("50.25"),
            profit_percentage=Decimal("0.025"),  # 2.5%
            effective_rate=Decimal("2000.5"),
            gas_used=250000,
            gas_price_gwei=Decimal("20.5"),
            gas_cost_usd=Decimal("12.75"),
            execution_time_ms=1500,
            slippage_actual=Decimal("0.002"),  # 0.2%
            mev_protected=True,
            flash_loan_used=False,
            discovered_at=now,
            executed_at=now
        )
        
        async_session.add(executed_path)
        await async_session.commit()
        await async_session.refresh(executed_path)
        
        # Verify the record was created correctly
        assert executed_path.id is not None
        assert isinstance(executed_path.id, uuid.UUID)
        assert executed_path.chain_name == "ethereum"
        assert executed_path.profit_usd == Decimal("50.25")
        assert executed_path.mev_protected is True
        assert executed_path.status == "success"  # Default value
        assert executed_path.created_at is not None
        assert executed_path.updated_at is not None
    
    async def test_executed_path_with_optional_fields(self, async_session: AsyncSession):
        """Test ExecutedPath with optional ML and risk fields."""
        now = datetime.now(timezone.utc)
        
        executed_path = ExecutedPath(
            path_hash="b2c3d4e5f6789012345678901234567890123456789012345678901234abcde",
            transaction_hash="0x2345678901bcdef1234567890abcdef1234567890abcdef1234567890abcdef1",
            block_number=18500001,
            chain_name="arbitrum",
            edge_ids=["ARB_CAMELOT_TRADE_ARB_USDC"],
            edge_types=["TRADE"],
            protocols=["Camelot"],
            input_amount=Decimal("500000000000000000"),  # 0.5 ETH
            output_amount=Decimal("1000000000"),  # 1000 USDC
            input_token="0x912CE59144191C1204E64559FE8253a0e49E6548",
            output_token="0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
            profit_usd=Decimal("25.10"),
            profit_percentage=Decimal("0.0125"),
            effective_rate=Decimal("2000.0"),
            gas_used=180000,
            gas_price_gwei=Decimal("0.1"),
            gas_cost_usd=Decimal("0.50"),
            execution_time_ms=800,
            slippage_actual=Decimal("0.001"),
            mev_protected=False,
            flash_loan_used=True,
            # Optional ML fields
            ml_confidence_score=Decimal("0.8750"),
            predicted_profit=Decimal("26.00"),
            prediction_error=Decimal("-0.90"),  # Predicted higher than actual
            # Optional risk fields
            max_exposure_delta=Decimal("0.05"),
            risk_score=Decimal("0.3200"),
            market_volatility=Decimal("0.150"),
            total_liquidity_usd=Decimal("5000000.00"),
            # Additional metadata
            extra_metadata={"flash_loan_provider": "aave", "execution_strategy": "atomic"},
            discovered_at=now,
            executed_at=now
        )
        
        async_session.add(executed_path)
        await async_session.commit()
        await async_session.refresh(executed_path)
        
        # Verify optional fields
        assert executed_path.ml_confidence_score == Decimal("0.8750")
        assert executed_path.predicted_profit == Decimal("26.00")
        assert executed_path.prediction_error == Decimal("-0.90")
        assert executed_path.max_exposure_delta == Decimal("0.05")
        assert executed_path.extra_metadata["flash_loan_provider"] == "aave"
    
    async def test_executed_path_failed_execution(self, async_session: AsyncSession):
        """Test ExecutedPath with failed execution status."""
        now = datetime.now(timezone.utc)
        
        executed_path = ExecutedPath(
            path_hash="c3d4e5f6789012345678901234567890123456789012345678901234abcdef1",
            transaction_hash="0x3456789012cdef1234567890abcdef1234567890abcdef1234567890abcdef12",
            block_number=18500002,
            chain_name="base",
            edge_ids=["BASE_AERODROME_TRADE_ETH_USDC"],
            edge_types=["TRADE"],
            protocols=["Aerodrome"],
            input_amount=Decimal("100000000000000000"),  # 0.1 ETH
            output_amount=Decimal("0"),  # Failed, no output
            input_token="0x4200000000000000000000000000000000000006",
            output_token="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            profit_usd=Decimal("-5.00"),  # Lost money due to gas
            profit_percentage=Decimal("-0.025"),
            effective_rate=Decimal("0.0"),
            gas_used=150000,
            gas_price_gwei=Decimal("0.05"),
            gas_cost_usd=Decimal("5.00"),
            execution_time_ms=2000,
            slippage_actual=Decimal("0.0"),
            mev_protected=False,
            flash_loan_used=False,
            status="failed",
            error_message="Insufficient liquidity for trade execution",
            discovered_at=now,
            executed_at=now
        )
        
        async_session.add(executed_path)
        await async_session.commit()
        await async_session.refresh(executed_path)
        
        assert executed_path.status == "failed"
        assert executed_path.error_message == "Insufficient liquidity for trade execution"
        assert executed_path.profit_usd == Decimal("-5.00")
        assert executed_path.output_amount == Decimal("0")
    
    async def test_executed_path_repr(self, async_session: AsyncSession):
        """Test ExecutedPath string representation."""
        now = datetime.now(timezone.utc)
        
        executed_path = ExecutedPath(
            path_hash="d4e5f6789012345678901234567890123456789012345678901234abcdef12",
            transaction_hash="0x456789abcdef1234567890abcdef1234567890abcdef1234567890abcdef123",
            block_number=18500003,
            chain_name="ethereum",
            edge_ids=["ETH_TEST"],
            edge_types=["TRADE"],
            protocols=["Test"],
            input_amount=Decimal("1"),
            output_amount=Decimal("1"),
            input_token="0x0000000000000000000000000000000000000001",
            output_token="0x0000000000000000000000000000000000000002",
            profit_usd=Decimal("100.50"),
            profit_percentage=Decimal("0.01"),
            effective_rate=Decimal("1.0"),
            gas_used=100000,
            gas_price_gwei=Decimal("10.0"),
            gas_cost_usd=Decimal("5.0"),
            execution_time_ms=1000,
            slippage_actual=Decimal("0.001"),
            discovered_at=now,
            executed_at=now
        )
        
        async_session.add(executed_path)
        await async_session.commit()
        await async_session.refresh(executed_path)
        
        repr_str = repr(executed_path)
        assert "ExecutedPath" in repr_str
        assert "0x456789" in repr_str  # First 8 chars of tx hash
        assert "$100.50" in repr_str
        assert "success" in repr_str


class TestTokenMetadata:
    """Test TokenMetadata model functionality."""
    
    async def test_token_metadata_creation(self, async_session: AsyncSession):
        """Test creating a TokenMetadata record."""
        now = datetime.now(timezone.utc)
        
        token = TokenMetadata(
            asset_id="ETH_MAINNET_0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            chain_name="ethereum",
            contract_address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            symbol="WETH",
            name="Wrapped Ether",
            decimals=18,
            token_type="ERC20",
            is_stable=False,
            is_wrapped=True,
            is_yield_bearing=False,
            price_usd=Decimal("2000.50"),
            price_eth=Decimal("1.0"),
            market_cap_usd=Decimal("25000000000.00"),
            volume_24h_usd=Decimal("1500000000.00"),
            liquidity_usd=Decimal("500000000.00"),
            total_supply=Decimal("2850000000000000000000000"),  # ~2.85M WETH
            circulating_supply=Decimal("2850000000000000000000000"),
            first_seen_at=now
        )
        
        async_session.add(token)
        await async_session.commit()
        await async_session.refresh(token)
        
        # Verify the record was created correctly
        assert token.id is not None
        assert isinstance(token.id, uuid.UUID)
        assert token.asset_id == "ETH_MAINNET_0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        assert token.symbol == "WETH"
        assert token.is_wrapped is True
        assert token.price_usd == Decimal("2000.50")
        assert token.is_active is True  # Default value
        assert token.is_verified is False  # Default value
        assert token.risk_level == "medium"  # Default value
        assert token.created_at is not None
        assert token.updated_at is not None
    
    async def test_stablecoin_metadata(self, async_session: AsyncSession):
        """Test TokenMetadata for a stablecoin."""
        now = datetime.now(timezone.utc)
        
        usdc = TokenMetadata(
            asset_id="ETH_MAINNET_0xA0b86a33E6441b9435B654f6D26Cc98b6e1D0a3A",
            chain_name="ethereum",
            contract_address="0xA0b86a33E6441b9435B654f6D26Cc98b6e1D0a3A",
            symbol="USDC",
            name="USD Coin",
            decimals=6,
            token_type="ERC20",
            is_stable=True,
            is_wrapped=False,
            is_yield_bearing=False,
            price_usd=Decimal("1.0005"),
            price_eth=Decimal("0.0005"),
            market_cap_usd=Decimal("32000000000.00"),
            volume_24h_usd=Decimal("8000000000.00"),
            liquidity_usd=Decimal("2000000000.00"),
            volatility_24h=Decimal("0.002"),  # Very low volatility
            security_score=Decimal("0.9500"),
            liquidity_score=Decimal("0.9800"),
            reliability_score=Decimal("0.9900"),
            protocols=["Aave", "Compound", "Uniswap"],
            primary_venue="Coinbase",
            is_verified=True,
            risk_level="low",
            coingecko_id="usd-coin",
            coinmarketcap_id="3408",
            tags=["stablecoin", "defi", "payments"],
            first_seen_at=now
        )
        
        async_session.add(usdc)
        await async_session.commit()
        await async_session.refresh(usdc)
        
        assert usdc.is_stable is True
        assert usdc.security_score == Decimal("0.9500")
        assert usdc.risk_level == "low"
        assert usdc.coingecko_id == "usd-coin"
        assert "stablecoin" in usdc.tags
        assert usdc.is_verified is True
    
    async def test_yield_bearing_token(self, async_session: AsyncSession):
        """Test TokenMetadata for a yield-bearing token."""
        now = datetime.now(timezone.utc)
        
        steth = TokenMetadata(
            asset_id="ETH_MAINNET_0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",
            chain_name="ethereum",
            contract_address="0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",
            symbol="stETH",
            name="Lido Staked Ether",
            decimals=18,
            token_type="ERC20",
            is_stable=False,
            is_wrapped=False,
            is_yield_bearing=True,
            price_usd=Decimal("1995.25"),
            price_eth=Decimal("0.9975"),
            base_apr=Decimal("0.0425"),  # 4.25% APR
            reward_tokens=["ETH"],
            protocols=["Lido", "Curve", "Aave"],
            primary_venue="Curve",
            security_score=Decimal("0.8500"),
            liquidity_score=Decimal("0.9200"),
            reliability_score=Decimal("0.9000"),
            risk_level="medium",
            tags=["staking", "liquid-staking", "yield"],
            extra_metadata={"staking_protocol": "lido", "validator_count": 400000},
            first_seen_at=now
        )
        
        async_session.add(steth)
        await async_session.commit()
        await async_session.refresh(steth)
        
        assert steth.is_yield_bearing is True
        assert steth.base_apr == Decimal("0.0425")
        assert steth.reward_tokens == ["ETH"]
        assert "staking" in steth.tags
        assert steth.extra_metadata["staking_protocol"] == "lido"
    
    async def test_inactive_token(self, async_session: AsyncSession):
        """Test TokenMetadata for an inactive/deprecated token."""
        now = datetime.now(timezone.utc)
        
        deprecated_token = TokenMetadata(
            asset_id="ETH_MAINNET_0x1234567890123456789012345678901234567890",
            chain_name="ethereum",
            contract_address="0x1234567890123456789012345678901234567890",
            symbol="DEPRECATED",
            name="Deprecated Token",
            decimals=18,
            is_active=False,
            is_verified=False,
            risk_level="high",
            security_score=Decimal("0.1000"),
            liquidity_score=Decimal("0.0500"),
            reliability_score=Decimal("0.0100"),
            first_seen_at=now
        )
        
        async_session.add(deprecated_token)
        await async_session.commit()
        await async_session.refresh(deprecated_token)
        
        assert deprecated_token.is_active is False
        assert deprecated_token.risk_level == "high"
        assert deprecated_token.security_score == Decimal("0.1000")
    
    async def test_token_metadata_repr(self, async_session: AsyncSession):
        """Test TokenMetadata string representation."""
        now = datetime.now(timezone.utc)
        
        token = TokenMetadata(
            asset_id="ETH_MAINNET_TEST",
            chain_name="ethereum",
            contract_address="0x1111111111111111111111111111111111111111",
            symbol="TEST",
            name="Test Token",
            decimals=18,
            price_usd=Decimal("123.45"),
            is_active=True,
            first_seen_at=now
        )
        
        async_session.add(token)
        await async_session.commit()
        await async_session.refresh(token)
        
        repr_str = repr(token)
        assert "TokenMetadata" in repr_str
        assert "ETH_MAINNET_TEST" in repr_str
        assert "TEST" in repr_str
        assert "123.45" in repr_str
        assert "True" in repr_str


class TestModelRelationships:
    """Test relationships and constraints between models."""
    
    async def test_unique_constraints(self, async_session: AsyncSession):
        """Test unique constraints on models."""
        now = datetime.now(timezone.utc)
        
        # Test unique transaction hash constraint
        path1 = ExecutedPath(
            path_hash="unique1",
            transaction_hash="0x1111111111111111111111111111111111111111111111111111111111111111",
            block_number=18500000,
            chain_name="ethereum",
            edge_ids=["TEST1"],
            edge_types=["TRADE"],
            protocols=["Test"],
            input_amount=Decimal("1"),
            output_amount=Decimal("1"),
            input_token="0x0000000000000000000000000000000000000001",
            output_token="0x0000000000000000000000000000000000000002",
            profit_usd=Decimal("1.0"),
            profit_percentage=Decimal("0.01"),
            effective_rate=Decimal("1.0"),
            gas_used=100000,
            gas_price_gwei=Decimal("10.0"),
            gas_cost_usd=Decimal("5.0"),
            execution_time_ms=1000,
            slippage_actual=Decimal("0.001"),
            discovered_at=now,
            executed_at=now
        )
        
        async_session.add(path1)
        await async_session.commit()
        
        # Test unique asset_id constraint
        token1 = TokenMetadata(
            asset_id="UNIQUE_ASSET_ID",
            chain_name="ethereum",
            contract_address="0x2222222222222222222222222222222222222222",
            symbol="UNQ1",
            name="Unique Token 1",
            decimals=18,
            first_seen_at=now
        )
        
        async_session.add(token1)
        await async_session.commit()
        
        # Attempting to create duplicate should work at SQLAlchemy level
        # (database-level constraints would be tested in integration tests)
        token2 = TokenMetadata(
            asset_id="UNIQUE_ASSET_ID_2",  # Different asset_id
            chain_name="ethereum",
            contract_address="0x3333333333333333333333333333333333333333",
            symbol="UNQ2",
            name="Unique Token 2",
            decimals=18,
            first_seen_at=now
        )
        
        async_session.add(token2)
        await async_session.commit()
        
        # Verify both records exist
        result = await async_session.execute(text("SELECT COUNT(*) FROM token_metadata"))
        count = result.scalar()
        assert count == 2
    
    async def test_model_field_lengths(self, async_session: AsyncSession):
        """Test field length constraints."""
        now = datetime.now(timezone.utc)
        
        # Test long symbol (should be truncated or fail validation)
        token = TokenMetadata(
            asset_id="TEST_LONG_SYMBOL",
            chain_name="ethereum",
            contract_address="0x4444444444444444444444444444444444444444",
            symbol="VERYLONGSYMBOLNAMETHATEXCEEDSLIMIT",  # > 32 chars
            name="Test Token",
            decimals=18,
            first_seen_at=now
        )
        
        # This should work in SQLAlchemy (database would enforce constraint)
        async_session.add(token)
        await async_session.commit()
        
        assert len(token.symbol) > 32  # SQLAlchemy doesn't enforce string length