"""
Unit tests for Execution Logging functionality.

Tests the ExecutionLogger, SimulatedExecution model, and LoggedExecutionEngine
integration with PostgreSQL logging.
"""
import pytest
import asyncio
import time
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from yield_arbitrage.database.execution_logger import ExecutionLogger, get_execution_logger
from yield_arbitrage.database.models import SimulatedExecution
from yield_arbitrage.execution.logged_execution_engine import LoggedExecutionEngine
from yield_arbitrage.execution.execution_engine import ExecutionContext, ExecutionStatus, PreFlightCheck, PreFlightCheckResult
from yield_arbitrage.execution.hybrid_simulator import SimulationResult, SimulationMode
from yield_arbitrage.pathfinding.path_models import YieldPath
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType


class TestExecutionLogger:
    """Test ExecutionLogger functionality."""
    
    @pytest.fixture
    def execution_logger(self):
        """Create ExecutionLogger instance."""
        return ExecutionLogger()
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = Mock()
        session.add = Mock()
        session.commit = AsyncMock()
        session.execute = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()
        return session
    
    @pytest.fixture
    def sample_execution_context(self):
        """Create sample execution context."""
        edges = [
            Mock(spec=YieldGraphEdge,
                 edge_id="edge_1",
                 source_asset_id="ETH_MAINNET_WETH",
                 target_asset_id="ETH_MAINNET_USDC",
                 edge_type=EdgeType.TRADE,
                 protocol_name="uniswapv3"),
            Mock(spec=YieldGraphEdge,
                 edge_id="edge_2",
                 source_asset_id="ETH_MAINNET_USDC", 
                 target_asset_id="ETH_MAINNET_WETH",
                 edge_type=EdgeType.TRADE,
                 protocol_name="sushiswap")
        ]
        
        path = YieldPath(path_id="test_path", edges=edges, expected_yield=0.025)
        
        return ExecutionContext(
            execution_id="test_exec_123",
            path=path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            chain_id=1
        )
    
    @pytest.fixture
    def sample_pre_flight_checks(self):
        """Create sample pre-flight checks."""
        return [
            PreFlightCheck(
                check_name="path_validity",
                result=PreFlightCheckResult.PASS,
                message="Path is valid"
            ),
            PreFlightCheck(
                check_name="oracle_health",
                result=PreFlightCheckResult.WARNING,
                message="Oracle slightly slow"
            ),
            PreFlightCheck(
                check_name="gas_price",
                result=PreFlightCheckResult.PASS,
                message="Gas price acceptable"
            )
        ]
    
    @pytest.fixture
    def sample_simulation_result(self):
        """Create sample simulation result."""
        return SimulationResult(
            success=True,
            simulation_mode=SimulationMode.HYBRID.value,
            profit_usd=25.50,
            profit_percentage=2.5,
            gas_cost_usd=8.75,
            output_amount=1000.0,
            simulation_time_ms=1500.0
        )
    
    def test_execution_logger_initialization(self, execution_logger):
        """Test ExecutionLogger initialization."""
        assert execution_logger.stats["records_created"] == 0
        assert execution_logger.stats["records_updated"] == 0
        assert execution_logger.stats["write_errors"] == 0
        assert execution_logger.stats["last_write_time"] is None
    
    @pytest.mark.asyncio
    @patch('yield_arbitrage.database.execution_logger.get_session')
    async def test_log_execution_start(self, mock_get_session, execution_logger, 
                                      sample_execution_context, mock_session):
        """Test logging execution start."""
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success = await execution_logger.log_execution_start(
            context=sample_execution_context,
            session_id="session_123",
            user_id="user_456",
            request_source="api"
        )
        
        assert success
        assert execution_logger.stats["records_created"] == 1
        assert execution_logger.stats["last_write_time"] is not None
        
        # Verify session.add was called with SimulatedExecution
        mock_session.add.assert_called_once()
        added_record = mock_session.add.call_args[0][0]
        assert isinstance(added_record, SimulatedExecution)
        assert added_record.execution_id == "test_exec_123"
        assert added_record.session_id == "session_123"
        assert added_record.user_id == "user_456"
        assert added_record.request_source == "api"
        assert added_record.chain_id == 1
        assert len(added_record.edge_ids) == 2
        assert "uniswapv3" in added_record.protocols
        assert "sushiswap" in added_record.protocols
    
    @pytest.mark.asyncio
    @patch('yield_arbitrage.database.execution_logger.get_session')
    async def test_log_pre_flight_results(self, mock_get_session, execution_logger,
                                         sample_pre_flight_checks, mock_session):
        """Test logging pre-flight check results."""
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success = await execution_logger.log_pre_flight_results(
            execution_id="test_exec_123",
            pre_flight_checks=sample_pre_flight_checks,
            pre_flight_time_ms=250
        )
        
        assert success
        assert execution_logger.stats["records_updated"] == 1
        
        # Verify update statement
        mock_session.execute.assert_called_once()
        
    @pytest.mark.asyncio
    @patch('yield_arbitrage.database.execution_logger.get_session')
    async def test_log_simulation_results(self, mock_get_session, execution_logger,
                                         sample_simulation_result, mock_session):
        """Test logging simulation results."""
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        market_context = {
            "eth_price_usd": 2000.0,
            "gas_price_gwei": 25.0,
            "block_number": 18500000
        }
        
        success = await execution_logger.log_simulation_results(
            execution_id="test_exec_123",
            simulation_result=sample_simulation_result,
            market_context=market_context
        )
        
        assert success
        assert execution_logger.stats["records_updated"] == 1
        
        # Verify update was called
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('yield_arbitrage.database.execution_logger.get_session')
    async def test_log_execution_update(self, mock_get_session, execution_logger, mock_session):
        """Test logging execution status update."""
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        success = await execution_logger.log_execution_update(
            execution_id="test_exec_123",
            status=ExecutionStatus.EXECUTING,
            additional_data={"test_field": "test_value"}
        )
        
        assert success
        assert execution_logger.stats["records_updated"] == 1
    
    def test_path_hash_calculation(self, execution_logger, sample_execution_context):
        """Test path hash calculation."""
        hash1 = execution_logger._calculate_path_hash(sample_execution_context.path.edges)
        hash2 = execution_logger._calculate_path_hash(sample_execution_context.path.edges)
        
        # Same path should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64-character hex string
        assert isinstance(hash1, str)
    
    def test_chain_name_mapping(self, execution_logger):
        """Test chain ID to name mapping."""
        assert execution_logger._get_chain_name(1) == "ethereum"
        assert execution_logger._get_chain_name(56) == "bsc"
        assert execution_logger._get_chain_name(137) == "polygon"
        assert execution_logger._get_chain_name(42161) == "arbitrum"
        assert execution_logger._get_chain_name(999999) == "chain_999999"  # Unknown chain
    
    def test_protocol_analysis(self, execution_logger):
        """Test protocol usage analysis."""
        mock_executions = [
            Mock(protocols=["uniswapv3", "sushiswap"]),
            Mock(protocols=["uniswapv3", "aave"]),
            Mock(protocols=["curve", "uniswapv3"]),
            Mock(protocols=["sushiswap"])
        ]
        
        result = execution_logger._analyze_protocols(mock_executions)
        
        # uniswapv3 should be most common (3 occurrences)
        assert result["uniswapv3"] == 3
        assert result["sushiswap"] == 2
        assert result["aave"] == 1
        assert result["curve"] == 1
    
    def test_chain_analysis(self, execution_logger):
        """Test chain distribution analysis."""
        mock_executions = [
            Mock(chain_name="ethereum"),
            Mock(chain_name="ethereum"),
            Mock(chain_name="arbitrum"),
            Mock(chain_name="polygon")
        ]
        
        result = execution_logger._analyze_chains(mock_executions)
        
        assert result["ethereum"] == 2
        assert result["arbitrum"] == 1
        assert result["polygon"] == 1


class TestLoggedExecutionEngine:
    """Test LoggedExecutionEngine functionality."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for LoggedExecutionEngine."""
        return {
            "simulator": Mock(),
            "transaction_builder": Mock(),
            "mev_router": Mock(),
            "delta_tracker": Mock(),
            "mev_assessor": Mock(),
            "asset_oracle": Mock()
        }
    
    @pytest.fixture
    def logged_execution_engine(self, mock_components):
        """Create LoggedExecutionEngine instance with mocks."""
        return LoggedExecutionEngine(
            **mock_components,
            router_address="0x" + "1" * 40,
            chain_id=1,
            enable_logging=True
        )
    
    @pytest.fixture
    def sample_path(self):
        """Create sample yield path."""
        edges = [
            Mock(spec=YieldGraphEdge,
                 edge_id="edge_1",
                 source_asset_id="ETH_MAINNET_WETH",
                 target_asset_id="ETH_MAINNET_USDC",
                 edge_type=EdgeType.TRADE,
                 protocol_name="uniswapv3")
        ]
        
        return YieldPath(
            path_id="test_path",
            edges=edges,
            expected_yield=0.025
        )
    
    def test_logged_execution_engine_initialization(self, logged_execution_engine):
        """Test LoggedExecutionEngine initialization."""
        assert logged_execution_engine.enable_logging
        assert logged_execution_engine.execution_logger is not None
        assert logged_execution_engine.logging_stats["logs_written"] == 0
        assert logged_execution_engine.logging_stats["log_failures"] == 0
    
    def test_disabled_logging_initialization(self, mock_components):
        """Test LoggedExecutionEngine with logging disabled."""
        engine = LoggedExecutionEngine(
            **mock_components,
            router_address="0x" + "1" * 40,
            chain_id=1,
            enable_logging=False
        )
        
        assert not engine.enable_logging
        assert engine.execution_logger is None
    
    @pytest.mark.asyncio
    @patch('yield_arbitrage.execution.logged_execution_engine.get_execution_logger')
    async def test_gather_market_context(self, mock_get_logger, logged_execution_engine):
        """Test market context gathering."""
        # Mock asset oracle response
        logged_execution_engine.asset_oracle.get_price_usd = AsyncMock(return_value=2500.0)
        
        context = await logged_execution_engine._gather_market_context()
        
        assert "eth_price_usd" in context
        assert context["eth_price_usd"] == 2500.0
        assert "gas_price_gwei" in context
        assert "block_number" in context
    
    @pytest.mark.asyncio
    @patch('yield_arbitrage.execution.logged_execution_engine.get_execution_logger')
    async def test_log_status_update(self, mock_get_logger, logged_execution_engine):
        """Test status update logging."""
        mock_logger = Mock()
        mock_logger.log_execution_update = AsyncMock(return_value=True)
        mock_get_logger.return_value = mock_logger
        logged_execution_engine.execution_logger = mock_logger
        
        await logged_execution_engine._log_status_update(
            execution_id="test_123",
            status=ExecutionStatus.SIMULATING,
            additional_data={"test": "data"}
        )
        
        mock_logger.log_execution_update.assert_called_once_with(
            execution_id="test_123",
            status=ExecutionStatus.SIMULATING,
            additional_data={"test": "data"}
        )
        
        assert logged_execution_engine.logging_stats["logs_written"] == 1
    
    def test_logging_stats(self, logged_execution_engine):
        """Test logging statistics retrieval."""
        # Simulate some logging activity
        logged_execution_engine.logging_stats["logs_written"] = 5
        logged_execution_engine.logging_stats["log_failures"] = 1
        logged_execution_engine.logging_stats["last_log_time"] = time.time()
        
        stats = logged_execution_engine.get_logging_stats()
        
        assert stats["logging_enabled"]
        assert stats["logs_written"] == 5
        assert stats["log_failures"] == 1
        assert "last_log_time_iso" in stats
        assert stats["last_log_time_iso"] is not None
    
    @pytest.mark.asyncio
    async def test_enable_disable_logging(self, logged_execution_engine):
        """Test enabling and disabling logging."""
        # Start with logging enabled
        assert logged_execution_engine.enable_logging
        
        # Disable logging
        logged_execution_engine.disable_database_logging()
        assert not logged_execution_engine.enable_logging
        assert logged_execution_engine.execution_logger is None
        
        # Re-enable logging
        with patch('yield_arbitrage.execution.logged_execution_engine.get_execution_logger') as mock_get:
            mock_logger = Mock()
            mock_get.return_value = mock_logger
            
            success = await logged_execution_engine.enable_database_logging()
            assert success
            assert logged_execution_engine.enable_logging
            assert logged_execution_engine.execution_logger == mock_logger


class TestSimulatedExecutionModel:
    """Test SimulatedExecution database model."""
    
    def test_simulated_execution_creation(self):
        """Test SimulatedExecution model creation."""
        now = datetime.now(timezone.utc)
        
        execution = SimulatedExecution(
            execution_id="test_exec_456",
            path_id="test_path_789",
            path_hash="abc123def456",
            chain_id=1,
            chain_name="ethereum",
            initial_amount=Decimal("1000000000000000000"),  # 1 ETH in wei
            start_asset_id="ETH_MAINNET_WETH",
            edge_ids=["edge_1", "edge_2"],
            edge_types=["TRADE", "TRADE"],
            protocols=["uniswapv3", "sushiswap"],
            status="pending",
            success=False,
            started_at=now
        )
        
        assert execution.execution_id == "test_exec_456"
        assert execution.path_id == "test_path_789"
        assert execution.chain_name == "ethereum"
        assert execution.initial_amount == Decimal("1000000000000000000")
        assert len(execution.edge_ids) == 2
        assert len(execution.protocols) == 2
        assert execution.status == "pending"
        assert not execution.success
    
    def test_simulated_execution_repr(self):
        """Test SimulatedExecution string representation."""
        execution = SimulatedExecution(
            execution_id="test_repr",
            path_id="path_repr",
            path_hash="hash_repr",
            chain_id=1,
            chain_name="ethereum",
            initial_amount=Decimal("1.0"),
            start_asset_id="ETH_MAINNET_WETH",
            edge_ids=["edge_1"],
            edge_types=["TRADE"],
            protocols=["uniswap"],
            status="completed",
            success=True,
            started_at=datetime.now(timezone.utc)
        )
        
        repr_str = repr(execution)
        assert "test_repr" in repr_str
        assert "completed" in repr_str
        assert "True" in repr_str


class TestIntegrationLogging:
    """Test integration between ExecutionEngine and logging system."""
    
    @pytest.mark.asyncio
    @patch('yield_arbitrage.database.execution_logger.get_session')
    async def test_end_to_end_logging_flow(self, mock_get_session):
        """Test complete logging flow from execution start to completion."""
        # Setup mocks
        mock_session = Mock()
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        
        # Create execution logger
        logger = ExecutionLogger()
        
        # Create sample execution context
        edges = [Mock(edge_id="test_edge", edge_type=EdgeType.TRADE, protocol_name="uniswap")]
        path = Mock(path_id="test_path", edges=edges)
        context = ExecutionContext(
            execution_id="integration_test",
            path=path,
            initial_amount=1.0,
            start_asset_id="ETH_MAINNET_WETH",
            chain_id=1
        )
        
        # 1. Log execution start
        success1 = await logger.log_execution_start(context, session_id="session_1")
        assert success1
        
        # 2. Log pre-flight results
        checks = [PreFlightCheck("test", PreFlightCheckResult.PASS, "OK")]
        success2 = await logger.log_pre_flight_results("integration_test", checks, 100)
        assert success2
        
        # 3. Log simulation results
        sim_result = SimulationResult(True, "hybrid", profit_usd=10.0)
        success3 = await logger.log_simulation_results("integration_test", sim_result)
        assert success3
        
        # Verify all logging operations succeeded
        assert logger.stats["records_created"] == 1
        assert logger.stats["records_updated"] == 2
        assert logger.stats["write_errors"] == 0


if __name__ == "__main__":
    # Run basic functionality test
    print("🧪 Testing Execution Logging")
    print("=" * 50)
    
    # Test ExecutionLogger initialization
    logger = ExecutionLogger()
    print(f"✅ ExecutionLogger initialized")
    print(f"   - Stats: {logger.get_stats()}")
    
    # Test path hash calculation
    from unittest.mock import Mock
    
    mock_edges = [
        Mock(edge_id="edge_1"),
        Mock(edge_id="edge_2")
    ]
    
    hash1 = logger._calculate_path_hash(mock_edges)
    hash2 = logger._calculate_path_hash(mock_edges)
    
    print(f"\n✅ Path hash calculation:")
    print(f"   - Hash: {hash1[:16]}...")
    print(f"   - Consistent: {hash1 == hash2}")
    
    # Test chain name mapping
    print(f"\n✅ Chain name mapping:")
    print(f"   - Chain 1: {logger._get_chain_name(1)}")
    print(f"   - Chain 137: {logger._get_chain_name(137)}")
    print(f"   - Chain 999: {logger._get_chain_name(999)}")
    
    print("\n✅ Execution Logging test passed!")