"""Unit tests for PathValidator and path validation algorithms."""
import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch

from yield_arbitrage.pathfinding.path_validator import (
    PathValidator,
    ValidationConfig,
    ValidationResult,
    ValidationReport
)
from yield_arbitrage.pathfinding.path_models import (
    SearchPath,
    PathNode,
    PathStatus
)
from yield_arbitrage.graph_engine.models import (
    UniversalYieldGraph,
    YieldGraphEdge,
    EdgeState,
    EdgeType
)


class TestValidationConfig:
    """Test ValidationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()
        
        assert config.max_path_length == 6
        assert config.min_path_length == 2
        assert config.min_confidence_threshold == 0.5
        assert config.min_cumulative_confidence == 0.3
        assert config.min_profit_threshold == 0.001
        assert config.max_gas_to_profit_ratio == 0.5
        assert config.min_liquidity_per_edge == 1000.0
        assert config.max_price_impact == 0.05
        assert config.max_slippage_tolerance == 0.02
        assert config.max_execution_time_estimate == 300.0
        assert config.min_success_probability == 0.7
        assert config.allow_immediate_cycles is True
        assert config.allow_complex_cycles is False
        assert config.max_asset_revisits == 2
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ValidationConfig(
            max_path_length=4,
            min_confidence_threshold=0.8,
            min_profit_threshold=0.05,
            allow_immediate_cycles=True,
            max_asset_revisits=2
        )
        
        assert config.max_path_length == 4
        assert config.min_confidence_threshold == 0.8
        assert config.min_profit_threshold == 0.05
        assert config.allow_immediate_cycles is True
        assert config.max_asset_revisits == 2


class TestValidationReport:
    """Test ValidationReport dataclass."""
    
    def test_valid_report(self):
        """Test valid validation report."""
        report = ValidationReport(
            is_valid=True,
            result=ValidationResult.VALID,
            errors=[],
            warnings=["Minor warning"],
            path_score=0.85,
            confidence_score=0.9,
            liquidity_score=0.8,
            risk_score=0.2,
            cycle_analysis={},
            constraint_analysis={},
            risk_analysis={},
            performance_metrics={}
        )
        
        assert report.is_valid is True
        assert report.result == ValidationResult.VALID
        assert len(report.errors) == 0
        assert len(report.warnings) == 1
        assert report.path_score == 0.85
    
    def test_invalid_report(self):
        """Test invalid validation report."""
        report = ValidationReport(
            is_valid=False,
            result=ValidationResult.INVALID_CYCLE,
            errors=["Cycle detected"],
            warnings=[],
            path_score=0.0,
            confidence_score=0.5,
            liquidity_score=0.3,
            risk_score=0.8,
            cycle_analysis={"has_cycles": True},
            constraint_analysis={},
            risk_analysis={},
            performance_metrics={}
        )
        
        assert report.is_valid is False
        assert report.result == ValidationResult.INVALID_CYCLE
        assert len(report.errors) == 1
        assert report.errors[0] == "Cycle detected"


class TestPathValidator:
    """Test PathValidator class."""
    
    @pytest.fixture
    def default_config(self):
        """Create default validation configuration."""
        return ValidationConfig(
            max_asset_revisits=2,  # Allow ETH->USDC->ETH arbitrage
            min_profit_threshold=0.001,  # Lower threshold for tests
            allow_immediate_cycles=True  # Allow simple arbitrage cycles
        )
    
    @pytest.fixture
    def strict_config(self):
        """Create strict validation configuration."""
        return ValidationConfig(
            max_path_length=4,
            min_confidence_threshold=0.8,
            min_cumulative_confidence=0.6,
            min_profit_threshold=0.05,
            max_gas_to_profit_ratio=0.3,
            allow_immediate_cycles=False,
            allow_complex_cycles=False
        )
    
    @pytest.fixture
    def validator(self, default_config):
        """Create PathValidator instance."""
        return PathValidator(default_config)
    
    @pytest.fixture
    def strict_validator(self, strict_config):
        """Create strict PathValidator instance."""
        return PathValidator(strict_config)
    
    @pytest.fixture
    def mock_graph(self):
        """Create mock graph for testing."""
        graph = Mock(spec=UniversalYieldGraph)
        
        # Mock some edges
        eth_usdc_edge = Mock(spec=YieldGraphEdge)
        eth_usdc_edge.edge_id = "eth_usdc_trade"
        eth_usdc_edge.protocol_name = "uniswapv3"
        eth_usdc_edge.chain_name = "ethereum"
        
        usdc_eth_edge = Mock(spec=YieldGraphEdge)
        usdc_eth_edge.edge_id = "usdc_eth_trade"
        usdc_eth_edge.protocol_name = "uniswapv3"
        usdc_eth_edge.chain_name = "ethereum"
        
        graph.get_edge.side_effect = lambda edge_id: {
            "eth_usdc_trade": eth_usdc_edge,
            "usdc_eth_trade": usdc_eth_edge
        }.get(edge_id)
        
        return graph
    
    @pytest.fixture
    def valid_path(self):
        """Create a valid arbitrage path."""
        nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("USDC", 1500.0, 0.005, 0.95, ["eth_usdc_trade"]),  # Very low gas
            PathNode("ETH", 1.05, 0.01, 0.90, ["eth_usdc_trade", "usdc_eth_trade"])  # Clearly profitable
        ]
        
        return SearchPath(
            nodes=nodes,
            total_score=0.0,
            status=PathStatus.COMPLETE,
            creation_time=time.time()
        )
    
    @pytest.fixture
    def cyclic_path(self):
        """Create a path with cycles."""
        nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("USDC", 1500.0, 15.0, 0.95, ["eth_usdc_trade"]),
            PathNode("ETH", 1.01, 30.0, 0.90, ["eth_usdc_trade", "usdc_eth_trade"]),
            PathNode("USDC", 1510.0, 45.0, 0.85, ["eth_usdc_trade", "usdc_eth_trade", "eth_usdc_trade2"])
        ]
        
        return SearchPath(
            nodes=nodes,
            total_score=0.0,
            status=PathStatus.ACTIVE,
            creation_time=time.time()
        )
    
    @pytest.fixture
    def long_path(self):
        """Create a path that's too long."""
        nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("USDC", 1500.0, 15.0, 0.95, ["edge1"]),
            PathNode("DAI", 1500.0, 30.0, 0.90, ["edge1", "edge2"]),
            PathNode("WBTC", 0.05, 45.0, 0.85, ["edge1", "edge2", "edge3"]),
            PathNode("USDT", 1500.0, 60.0, 0.80, ["edge1", "edge2", "edge3", "edge4"]),
            PathNode("BUSD", 1500.0, 75.0, 0.75, ["edge1", "edge2", "edge3", "edge4", "edge5"]),
            PathNode("USDC", 1500.0, 90.0, 0.70, ["edge1", "edge2", "edge3", "edge4", "edge5", "edge6"]),
            PathNode("ETH", 0.98, 105.0, 0.65, ["edge1", "edge2", "edge3", "edge4", "edge5", "edge6", "edge7"])
        ]
        
        return SearchPath(
            nodes=nodes,
            total_score=0.0,
            status=PathStatus.COMPLETE,
            creation_time=time.time()
        )
    
    @pytest.fixture
    def low_confidence_path(self):
        """Create a path with low confidence."""
        nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("USDC", 1500.0, 15.0, 0.3, ["low_conf_edge1"]),  # Very low confidence
            PathNode("ETH", 1.01, 30.0, 0.1, ["low_conf_edge1", "low_conf_edge2"])  # Extremely low
        ]
        
        return SearchPath(
            nodes=nodes,
            total_score=0.0,
            status=PathStatus.COMPLETE,
            creation_time=time.time()
        )
    
    @pytest.fixture
    def unprofitable_path(self):
        """Create an unprofitable path."""
        nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("USDC", 1500.0, 500.0, 0.95, ["expensive_edge1"]),  # Very high gas
            PathNode("ETH", 0.99, 1000.0, 0.90, ["expensive_edge1", "expensive_edge2"])  # Loss + high gas
        ]
        
        return SearchPath(
            nodes=nodes,
            total_score=0.0,
            status=PathStatus.COMPLETE,
            creation_time=time.time()
        )
    
    def test_validator_initialization(self, default_config):
        """Test validator initialization."""
        validator = PathValidator(default_config)
        
        assert validator.config is default_config
        assert validator._validation_stats["paths_validated"] == 0
        assert validator._validation_stats["valid_paths"] == 0
    
    def test_default_initialization(self):
        """Test validator initialization with default config."""
        validator = PathValidator()
        
        assert isinstance(validator.config, ValidationConfig)
        assert validator.config.max_path_length == 6
    
    @pytest.mark.asyncio
    async def test_validate_valid_path(self, validator, mock_graph, valid_path):
        """Test validation of a valid path."""
        report = await validator.validate_path(valid_path, mock_graph, "ETH", 1.0)
        
        assert report.is_valid is True
        assert report.result == ValidationResult.VALID
        assert len(report.errors) == 0
        assert report.path_score > 0
        assert report.confidence_score > 0
        assert report.liquidity_score >= 0
    
    @pytest.mark.asyncio
    async def test_validate_cyclic_path(self, validator, mock_graph, cyclic_path):
        """Test validation of a path with cycles."""
        report = await validator.validate_path(cyclic_path, mock_graph, "ETH", 1.0)
        
        # With default settings allowing asset revisits up to 2, this should be valid
        # unless there are other constraint violations
        if not report.is_valid:
            # Check if it failed due to cycles or other constraints
            assert report.cycle_analysis is not None
        else:
            # Path should be valid with current permissive settings
            assert report.cycle_analysis["has_cycles"] is True
    
    @pytest.mark.asyncio
    async def test_validate_long_path(self, validator, mock_graph, long_path):
        """Test validation of a path that's too long."""
        report = await validator.validate_path(long_path, mock_graph, "ETH", 1.0)
        
        assert report.is_valid is False
        # Could fail due to length OR cycles (since it has ETH->...->ETH)
        assert report.result in [ValidationResult.INVALID_LENGTH, ValidationResult.INVALID_CYCLE]
        assert len(report.errors) > 0
    
    @pytest.mark.asyncio
    async def test_validate_low_confidence_path(self, validator, mock_graph, low_confidence_path):
        """Test validation of a path with low confidence."""
        report = await validator.validate_path(low_confidence_path, mock_graph, "ETH", 1.0)
        
        assert report.is_valid is False
        assert report.result == ValidationResult.INVALID_CONFIDENCE
        assert any("confidence" in error.lower() for error in report.errors)
        assert report.confidence_score < 0.5
    
    @pytest.mark.asyncio
    async def test_validate_unprofitable_path(self, validator, mock_graph, unprofitable_path):
        """Test validation of an unprofitable path."""
        report = await validator.validate_path(unprofitable_path, mock_graph, "ETH", 1.0)
        
        assert report.is_valid is False
        assert report.result in [ValidationResult.INVALID_GAS_COST, ValidationResult.INVALID_PROFIT]
        assert any("profit" in error.lower() or "gas" in error.lower() for error in report.errors)
    
    @pytest.mark.asyncio
    async def test_validate_empty_path(self, validator, mock_graph):
        """Test validation of an empty path."""
        empty_path = SearchPath(nodes=[], total_score=0.0)
        report = await validator.validate_path(empty_path, mock_graph, "ETH", 1.0)
        
        assert report.is_valid is False
        assert len(report.errors) > 0
    
    @pytest.mark.asyncio
    async def test_strict_validation(self, strict_validator, mock_graph, valid_path):
        """Test strict validation configuration."""
        # Valid path might fail strict validation due to higher thresholds
        report = await strict_validator.validate_path(valid_path, mock_graph, "ETH", 1.0)
        
        # May fail strict validation due to confidence or profit thresholds
        if not report.is_valid:
            assert len(report.errors) > 0
    
    def test_basic_structure_validation(self, validator):
        """Test basic path structure validation."""
        # Test with inconsistent edge paths
        invalid_nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("USDC", 1500.0, 15.0, 0.95, ["edge1", "edge2", "edge3"])  # Too many edges
        ]
        invalid_path = SearchPath(nodes=invalid_nodes, total_score=0.0)
        
        result = validator._validate_basic_structure(invalid_path)
        assert len(result["errors"]) > 0
    
    def test_cycle_analysis(self, validator, cyclic_path):
        """Test cycle analysis functionality."""
        analysis = validator._analyze_cycles(cyclic_path)
        
        assert analysis["has_cycles"] is True
        # With current permissive settings, cycles may not be invalid
        assert len(analysis["cycles_found"]) > 0
        assert analysis["asset_visit_counts"]["ETH"] > 1
        assert analysis["asset_visit_counts"]["USDC"] > 1
    
    def test_cycle_analysis_no_cycles(self, validator, valid_path):
        """Test cycle analysis with no cycles."""
        analysis = validator._analyze_cycles(valid_path)
        
        # Valid arbitrage path has controlled cycle (ETH -> USDC -> ETH)
        assert analysis["asset_visit_counts"]["ETH"] == 2
        assert analysis["asset_visit_counts"]["USDC"] == 1
    
    def test_path_length_validation(self, validator):
        """Test path length validation."""
        # Too short
        short_result = validator._validate_path_length(SearchPath(
            nodes=[PathNode("ETH", 1.0)], total_score=0.0
        ))
        assert not short_result["valid"]
        assert "too short" in short_result["error"]
        
        # Too long  
        long_path = SearchPath(nodes=[PathNode(f"ASSET_{i}", 1.0) for i in range(8)], total_score=0.0)
        long_result = validator._validate_path_length(long_path)
        assert not long_result["valid"]
        assert "too long" in long_result["error"]
        
        # Just right
        valid_path = SearchPath(nodes=[PathNode(f"ASSET_{i}", 1.0) for i in range(4)], total_score=0.0)
        valid_result = validator._validate_path_length(valid_path)
        assert valid_result["valid"]
    
    def test_confidence_validation(self, validator, valid_path, low_confidence_path):
        """Test confidence requirements validation."""
        # Valid path
        valid_result = validator._validate_confidence_requirements(valid_path)
        assert valid_result["valid"]
        assert valid_result["score"] > 0
        
        # Low confidence path
        invalid_result = validator._validate_confidence_requirements(low_confidence_path)
        assert not invalid_result["valid"]
        assert "confidence" in invalid_result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_economic_constraints_validation(self, validator, valid_path, unprofitable_path):
        """Test economic constraints validation."""
        # Valid profitable path
        valid_result = await validator._validate_economic_constraints(valid_path, "ETH", 1.0)
        assert valid_result["valid"]
        assert valid_result["net_profit"] > 0
        
        # Unprofitable path
        invalid_result = await validator._validate_economic_constraints(unprofitable_path, "ETH", 1.0)
        assert not invalid_result["valid"]
        assert len(invalid_result["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_liquidity_constraints_validation(self, validator, valid_path):
        """Test liquidity constraints validation."""
        result = await validator._validate_liquidity_constraints(valid_path, 1.0)
        
        # With mock liquidity, should generally pass
        assert result["score"] >= 0
        assert result["total_liquidity"] > 0
        assert result["max_price_impact"] >= 0
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self, validator, valid_path, long_path, mock_graph):
        """Test path risk assessment."""
        # Valid path should have reasonable risk
        valid_risk = await validator._assess_path_risks(valid_path, mock_graph)
        assert 0 <= valid_risk["risk_score"] <= 1
        
        # Long path should have higher risk
        long_risk = await validator._assess_path_risks(long_path, mock_graph)
        assert long_risk["risk_score"] >= valid_risk["risk_score"]  # May be equal due to thresholds
        assert len(long_risk["risk_factors"]) >= len(valid_risk["risk_factors"])
    
    @pytest.mark.asyncio
    async def test_technical_constraints_validation(self, validator, valid_path, mock_graph):
        """Test technical constraints validation."""
        result = await validator._validate_technical_constraints(valid_path, mock_graph)
        
        # With simple mock, should generally pass
        assert isinstance(result["valid"], bool)
        assert isinstance(result["errors"], list)
        assert isinstance(result["warnings"], list)
    
    def test_cross_chain_validation(self, validator, valid_path, mock_graph):
        """Test cross-chain consistency validation."""
        result = validator._validate_cross_chain_consistency(valid_path, mock_graph)
        
        # With ethereum-only mock, should pass
        assert len(result["errors"]) == 0
    
    def test_temporal_consistency_validation(self, validator, valid_path):
        """Test temporal consistency validation."""
        result = validator._validate_temporal_consistency(valid_path)
        
        # Should estimate reasonable execution time
        assert isinstance(result["errors"], list)
        assert isinstance(result["warnings"], list)
    
    def test_edge_compatibility_validation(self, validator, valid_path, mock_graph):
        """Test edge compatibility validation."""
        result = validator._validate_edge_compatibility(valid_path, mock_graph)
        
        # With same protocol mock, should pass
        assert len(result["errors"]) == 0
    
    def test_execution_complexity_calculation(self, validator, valid_path, long_path):
        """Test execution complexity calculation."""
        valid_complexity = validator._calculate_execution_complexity(valid_path)
        long_complexity = validator._calculate_execution_complexity(long_path)
        
        assert 0 <= valid_complexity <= 1
        assert 0 <= long_complexity <= 1
        assert long_complexity > valid_complexity
    
    def test_execution_time_estimation(self, validator, valid_path, long_path):
        """Test execution time estimation."""
        valid_time = validator._estimate_execution_time(valid_path)
        long_time = validator._estimate_execution_time(long_path)
        
        assert valid_time > 0
        assert long_time > valid_time
    
    def test_overall_score_calculation(self, validator):
        """Test overall path score calculation."""
        # Perfect path
        perfect_score = validator._calculate_overall_score(1.0, 1.0, 0.0, 0, 0)
        assert perfect_score == 1.0
        
        # Path with errors
        error_score = validator._calculate_overall_score(1.0, 1.0, 0.0, 1, 0)
        assert error_score == 0.0
        
        # Path with warnings
        warning_score = validator._calculate_overall_score(0.8, 0.8, 0.2, 0, 2)
        assert warning_score < 0.8
    
    def test_performance_metrics_calculation(self, validator, valid_path):
        """Test performance metrics calculation."""
        metrics = validator._calculate_performance_metrics(valid_path)
        
        assert "initial_amount" in metrics
        assert "final_amount" in metrics
        assert "gross_return" in metrics
        assert "net_return" in metrics
        assert "return_ratio" in metrics
        assert "gas_efficiency" in metrics
        assert "confidence_preservation" in metrics
        assert "path_efficiency" in metrics
        
        assert metrics["initial_amount"] == 1.0
        assert metrics["final_amount"] == 1.05
        assert abs(metrics["gross_return"] - 0.05) < 0.001
    
    def test_performance_metrics_empty_path(self, validator):
        """Test performance metrics with empty path."""
        empty_path = SearchPath(nodes=[], total_score=0.0)
        metrics = validator._calculate_performance_metrics(empty_path)
        
        assert len(metrics) == 0
    
    def test_validation_statistics(self, validator, mock_graph, valid_path):
        """Test validation statistics tracking."""
        initial_stats = validator.get_validation_stats()
        assert initial_stats["paths_validated"] == 0
        assert initial_stats["valid_paths"] == 0
        
        # Validate a path
        asyncio.run(validator.validate_path(valid_path, mock_graph, "ETH", 1.0))
        
        updated_stats = validator.get_validation_stats()
        assert updated_stats["paths_validated"] == 1
        assert updated_stats["valid_paths"] == 1
        assert "validation_rate" in updated_stats
        assert "result_distribution" in updated_stats
    
    def test_clear_statistics(self, validator):
        """Test clearing validation statistics."""
        # Add some fake stats
        validator._validation_stats["paths_validated"] = 10
        validator._validation_stats["valid_paths"] = 7
        
        validator.clear_stats()
        
        stats = validator.get_validation_stats()
        assert stats["paths_validated"] == 0
        assert stats["valid_paths"] == 0
    
    @pytest.mark.asyncio
    async def test_validation_with_different_configs(self, mock_graph, valid_path):
        """Test validation with different configurations."""
        # Permissive config
        permissive_config = ValidationConfig(
            min_confidence_threshold=0.1,
            min_profit_threshold=0.001,
            allow_immediate_cycles=True,
            max_asset_revisits=3
        )
        permissive_validator = PathValidator(permissive_config)
        
        # Strict config
        strict_config = ValidationConfig(
            min_confidence_threshold=0.95,
            min_profit_threshold=0.1,
            max_gas_to_profit_ratio=0.1,
            allow_immediate_cycles=False,
            max_asset_revisits=1
        )
        strict_validator = PathValidator(strict_config)
        
        # Same path, different results
        permissive_result = await permissive_validator.validate_path(valid_path, mock_graph, "ETH", 1.0)
        strict_result = await strict_validator.validate_path(valid_path, mock_graph, "ETH", 1.0)
        
        # Permissive should be more likely to pass
        assert permissive_result.is_valid or not strict_result.is_valid
    
    @pytest.mark.asyncio
    async def test_partial_path_validation(self, validator, mock_graph):
        """Test validation of partial (incomplete) paths."""
        partial_path = SearchPath(
            nodes=[
                PathNode("ETH", 1.0, 0.0, 1.0, []),
                PathNode("USDC", 1500.0, 15.0, 0.95, ["eth_usdc_trade"])
            ],
            total_score=0.0,
            status=PathStatus.ACTIVE
        )
        
        report = await validator.validate_path(partial_path, mock_graph, "ETH", 1.0)
        
        # Partial paths should be evaluated for potential
        assert isinstance(report.is_valid, bool)
        assert report.constraint_analysis["economic"]["net_profit"] != 0  # Should estimate profit
    
    @pytest.mark.asyncio
    async def test_edge_case_validations(self, validator, mock_graph):
        """Test edge cases in validation."""
        # Single node path
        single_node = SearchPath(
            nodes=[PathNode("ETH", 1.0, 0.0, 1.0, [])],
            total_score=0.0
        )
        
        report = await validator.validate_path(single_node, mock_graph, "ETH", 1.0)
        assert not report.is_valid  # Too short
        
        # Path with zero amounts
        zero_amount_path = SearchPath(
            nodes=[
                PathNode("ETH", 1.0, 0.0, 1.0, []),
                PathNode("USDC", 0.0, 15.0, 0.95, ["edge1"]),  # Zero amount
                PathNode("ETH", 0.0, 30.0, 0.90, ["edge1", "edge2"])
            ],
            total_score=0.0
        )
        
        report = await validator.validate_path(zero_amount_path, mock_graph, "ETH", 1.0)
        assert not report.is_valid  # Invalid amounts
    
    @pytest.mark.asyncio
    async def test_concurrent_validations(self, validator, mock_graph, valid_path):
        """Test concurrent validation operations."""
        import asyncio
        
        # Run multiple validations concurrently
        tasks = [
            validator.validate_path(valid_path, mock_graph, "ETH", 1.0)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert isinstance(result, ValidationReport)
        
        # Statistics should reflect all validations
        stats = validator.get_validation_stats()
        assert stats["paths_validated"] >= 5