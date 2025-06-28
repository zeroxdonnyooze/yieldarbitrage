"""Unit tests for NonMLPathScorer and path scoring algorithms."""
import pytest
import time
import math
from unittest.mock import Mock, AsyncMock, patch

from yield_arbitrage.pathfinding.path_scorer import (
    NonMLPathScorer,
    ScoringConfig,
    ScoringMethod,
    PathScoreBreakdown
)
from yield_arbitrage.pathfinding.path_models import (
    SearchPath,
    PathNode,
    PathStatus
)
from yield_arbitrage.graph_engine.models import EdgeState


class TestScoringConfig:
    """Test ScoringConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ScoringConfig()
        
        assert config.method == ScoringMethod.COMPOSITE
        assert config.profitability_weight == 0.4
        assert config.liquidity_weight == 0.25
        assert config.confidence_weight == 0.15
        assert config.gas_efficiency_weight == 0.15
        assert config.time_penalty_weight == 0.05
        assert config.max_acceptable_slippage == 0.02
        assert config.min_liquidity_threshold == 10_000.0
        assert config.gas_price_gwei == 20.0
        assert config.max_optimal_length == 4
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ScoringConfig(
            method=ScoringMethod.RISK_ADJUSTED,
            profitability_weight=0.5,
            liquidity_weight=0.3,
            max_acceptable_slippage=0.01,
            min_liquidity_threshold=50_000.0
        )
        
        assert config.method == ScoringMethod.RISK_ADJUSTED
        assert config.profitability_weight == 0.5
        assert config.liquidity_weight == 0.3
        assert config.max_acceptable_slippage == 0.01
        assert config.min_liquidity_threshold == 50_000.0


class TestPathScoreBreakdown:
    """Test PathScoreBreakdown dataclass."""
    
    def test_breakdown_initialization(self):
        """Test PathScoreBreakdown initialization."""
        breakdown = PathScoreBreakdown(
            total_score=0.75,
            profitability_score=0.8,
            liquidity_score=0.9,
            confidence_score=0.85,
            gas_efficiency_score=0.7,
            time_penalty=0.05,
            risk_adjustment=-0.1,
            expected_profit_usd=100.0,
            expected_gas_cost_usd=15.0,
            total_liquidity_usd=1_000_000.0,
            min_edge_confidence=0.9,
            max_slippage_estimate=0.01,
            path_length=3,
            warnings=["Low liquidity warning"],
            risk_flags=["High gas cost"]
        )
        
        assert breakdown.total_score == 0.75
        assert breakdown.expected_profit_usd == 100.0
        assert len(breakdown.warnings) == 1
        assert len(breakdown.risk_flags) == 1


class TestNonMLPathScorer:
    """Test NonMLPathScorer class."""
    
    @pytest.fixture
    def default_config(self):
        """Create default scoring configuration."""
        return ScoringConfig()
    
    @pytest.fixture
    def scorer(self, default_config):
        """Create NonMLPathScorer instance."""
        return NonMLPathScorer(default_config)
    
    @pytest.fixture
    def simple_path(self):
        """Create a simple profitable path for testing."""
        nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("USDC", 1500.0, 15.0, 0.95, ["eth_usdc_trade"]),
            PathNode("ETH", 1.01, 30.0, 0.90, ["eth_usdc_trade", "usdc_eth_trade"])
        ]
        
        return SearchPath(
            nodes=nodes,
            total_score=0.0,
            status=PathStatus.COMPLETE,
            creation_time=time.time()
        )
    
    @pytest.fixture
    def partial_path(self):
        """Create a partial path for testing."""
        nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("USDC", 1500.0, 15.0, 0.95, ["eth_usdc_trade"])
        ]
        
        return SearchPath(
            nodes=nodes,
            total_score=0.0,
            status=PathStatus.ACTIVE,
            creation_time=time.time()
        )
    
    @pytest.fixture
    def long_path(self):
        """Create a long path for testing length penalties."""
        nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("USDC", 1500.0, 15.0, 0.95, ["edge1"]),
            PathNode("DAI", 1500.0, 30.0, 0.90, ["edge1", "edge2"]),
            PathNode("WBTC", 0.05, 45.0, 0.85, ["edge1", "edge2", "edge3"]),
            PathNode("USDT", 1500.0, 60.0, 0.80, ["edge1", "edge2", "edge3", "edge4"]),
            PathNode("USDC", 1500.0, 75.0, 0.75, ["edge1", "edge2", "edge3", "edge4", "edge5"]),
            PathNode("ETH", 0.98, 90.0, 0.70, ["edge1", "edge2", "edge3", "edge4", "edge5", "edge6"])
        ]
        
        return SearchPath(
            nodes=nodes,
            total_score=0.0,
            status=PathStatus.COMPLETE,
            creation_time=time.time()
        )
    
    @pytest.fixture
    def low_confidence_path(self):
        """Create a path with low confidence for testing."""
        nodes = [
            PathNode("ETH", 1.0, 0.0, 1.0, []),
            PathNode("USDC", 1500.0, 15.0, 0.4, ["edge1"]),  # Low confidence
            PathNode("ETH", 1.01, 30.0, 0.3, ["edge1", "edge2"])  # Very low confidence
        ]
        
        return SearchPath(
            nodes=nodes,
            total_score=0.0,
            status=PathStatus.COMPLETE,
            creation_time=time.time()
        )
    
    def test_scorer_initialization(self, default_config):
        """Test scorer initialization."""
        scorer = NonMLPathScorer(default_config)
        
        assert scorer.config is default_config
        assert len(scorer._score_cache) == 0
        assert scorer._scoring_stats["paths_scored"] == 0
    
    def test_default_initialization(self):
        """Test scorer initialization with default config."""
        scorer = NonMLPathScorer()
        
        assert scorer.config.method == ScoringMethod.COMPOSITE
        assert isinstance(scorer.config, ScoringConfig)
    
    @pytest.mark.asyncio
    async def test_score_simple_profit_method(self, simple_path):
        """Test simple profit scoring method."""
        config = ScoringConfig(method=ScoringMethod.SIMPLE_PROFIT)
        scorer = NonMLPathScorer(config)
        
        breakdown = await scorer.score_path(simple_path, "ETH", 1.0)
        
        assert breakdown.total_score > 0
        assert breakdown.expected_profit_usd > 0
        assert breakdown.profitability_score > 0
        # Simple method should have basic values for other metrics
        assert breakdown.liquidity_score == 1.0
        assert breakdown.confidence_score == 1.0
    
    @pytest.mark.asyncio
    async def test_score_composite_method(self, simple_path, scorer):
        """Test composite scoring method."""
        breakdown = await scorer.score_path(simple_path, "ETH", 1.0)
        
        assert breakdown.total_score > 0
        assert breakdown.profitability_score >= 0
        assert breakdown.liquidity_score >= 0
        assert breakdown.confidence_score >= 0
        assert breakdown.gas_efficiency_score >= 0
        assert breakdown.time_penalty >= 0
        assert isinstance(breakdown.warnings, list)
        assert isinstance(breakdown.risk_flags, list)
    
    @pytest.mark.asyncio
    async def test_score_risk_adjusted_method(self, simple_path):
        """Test risk-adjusted scoring method."""
        config = ScoringConfig(method=ScoringMethod.RISK_ADJUSTED)
        scorer = NonMLPathScorer(config)
        
        breakdown = await scorer.score_path(simple_path, "ETH", 1.0)
        
        assert breakdown.total_score >= 0
        # Risk-adjusted should typically have lower scores than composite
        assert breakdown.risk_adjustment <= 0
    
    @pytest.mark.asyncio
    async def test_score_kelly_criterion_method(self, simple_path):
        """Test Kelly Criterion scoring method."""
        config = ScoringConfig(method=ScoringMethod.KELLY_CRITERION)
        scorer = NonMLPathScorer(config)
        
        breakdown = await scorer.score_path(simple_path, "ETH", 1.0)
        
        assert breakdown.total_score >= 0
        # Kelly criterion should apply position sizing adjustments
        assert breakdown.risk_adjustment != 0
    
    @pytest.mark.asyncio
    async def test_score_partial_path(self, partial_path, scorer):
        """Test scoring of partial paths."""
        breakdown = await scorer.score_path(partial_path, "ETH", 1.0)
        
        assert breakdown.total_score >= 0
        assert breakdown.path_length == 1
        # Partial paths should have estimated profitability
        assert breakdown.expected_profit_usd != 0
    
    @pytest.mark.asyncio
    async def test_score_long_path_penalties(self, long_path, scorer):
        """Test that long paths receive appropriate penalties."""
        breakdown = await scorer.score_path(long_path, "ETH", 1.0)
        
        assert breakdown.path_length == 6  # 7 nodes = 6 edges
        assert breakdown.total_score >= 0
        # Long paths should have penalties
        assert breakdown.time_penalty >= 0
        # Should have complexity penalty in risk adjustment
        assert breakdown.risk_adjustment < 0
    
    @pytest.mark.asyncio
    async def test_score_low_confidence_path(self, low_confidence_path, scorer):
        """Test scoring of paths with low confidence."""
        breakdown = await scorer.score_path(low_confidence_path, "ETH", 1.0)
        
        assert breakdown.min_edge_confidence == 0.3  # Lowest confidence
        assert "Low confidence" in str(breakdown.warnings)
        # Low confidence should reduce overall score
        assert breakdown.confidence_score < 0.5
    
    @pytest.mark.asyncio
    async def test_profitability_metrics_calculation(self, simple_path, scorer):
        """Test profitability metrics calculation."""
        # Test completed path
        metrics = await scorer._calculate_profitability_metrics(simple_path, 1.0, True)
        
        assert "gross_profit_usd" in metrics
        assert "net_profit_usd" in metrics
        assert "profit_ratio" in metrics
        assert "gas_cost_usd" in metrics
        
        # Net profit should account for gas costs
        assert metrics["net_profit_usd"] == metrics["gross_profit_usd"] - metrics["gas_cost_usd"]
        assert metrics["profit_ratio"] == metrics["net_profit_usd"] / 1.0
    
    @pytest.mark.asyncio
    async def test_profitability_metrics_partial_path(self, partial_path, scorer):
        """Test profitability metrics for partial path."""
        metrics = await scorer._calculate_profitability_metrics(partial_path, 1.0, False)
        
        # Partial paths should have estimated completion factor applied
        gross_profit_expected = (1500.0 - 1.0) * 0.7  # 0.7 completion factor
        assert abs(metrics["gross_profit_usd"] - gross_profit_expected) < 0.01
    
    @pytest.mark.asyncio
    async def test_liquidity_metrics_calculation(self, simple_path, scorer):
        """Test liquidity metrics calculation."""
        metrics = await scorer._calculate_liquidity_metrics(simple_path, 1.0)
        
        assert "total_liquidity" in metrics
        assert "min_liquidity" in metrics
        assert "max_slippage" in metrics
        assert "normalized_score" in metrics
        
        assert metrics["total_liquidity"] > 0
        assert 0 <= metrics["normalized_score"] <= 1
    
    def test_confidence_metrics_calculation(self, simple_path, scorer):
        """Test confidence metrics calculation."""
        metrics = scorer._calculate_confidence_metrics(simple_path)
        
        assert "min_confidence" in metrics
        assert "avg_confidence" in metrics
        assert "normalized_score" in metrics
        
        assert metrics["min_confidence"] == 0.90  # Lowest confidence in path
        assert 0 <= metrics["normalized_score"] <= 1
    
    def test_confidence_metrics_empty_path(self, scorer):
        """Test confidence metrics with empty path."""
        empty_path = SearchPath(nodes=[], total_score=0.0)
        metrics = scorer._calculate_confidence_metrics(empty_path)
        
        assert metrics["min_confidence"] == 0.0
        assert metrics["avg_confidence"] == 0.0
        assert metrics["normalized_score"] == 0.0
    
    @pytest.mark.asyncio
    async def test_gas_efficiency_metrics(self, simple_path, scorer):
        """Test gas efficiency metrics calculation."""
        expected_profit = 10.0  # $10 profit
        metrics = await scorer._calculate_gas_efficiency_metrics(simple_path, expected_profit)
        
        assert "total_gas_cost" in metrics
        assert "gas_to_profit_ratio" in metrics
        assert "efficiency" in metrics
        assert "normalized_score" in metrics
        
        # Gas to profit ratio should be reasonable
        expected_ratio = simple_path.total_gas_cost / expected_profit
        assert abs(metrics["gas_to_profit_ratio"] - expected_ratio) < 0.01
    
    def test_time_penalty_calculation(self, scorer):
        """Test time-based penalty calculation."""
        # Create path with specific timestamp
        old_time = time.time() - 600  # 10 minutes ago
        old_path = SearchPath(
            nodes=[PathNode("ETH", 1.0)],
            total_score=0.0,
            creation_time=old_time
        )
        
        penalty = scorer._calculate_time_penalty(old_path)
        
        assert penalty > 0  # Should have penalty for old path
        assert penalty < 1.0  # Should not be too severe
    
    def test_length_penalty_calculation(self, scorer):
        """Test path length penalty calculation."""
        # Short path (within optimal length)
        short_penalty = scorer._calculate_length_penalty(3)
        assert short_penalty == 0.0
        
        # Long path (beyond optimal length)
        long_penalty = scorer._calculate_length_penalty(8)
        assert long_penalty > 0
        assert long_penalty == (8 - scorer.config.max_optimal_length) * scorer.config.length_penalty_factor
    
    @pytest.mark.asyncio
    async def test_risk_adjustments_low_liquidity(self, scorer):
        """Test risk adjustments for low liquidity."""
        # Mock liquidity metrics with low liquidity
        liquidity_metrics = {
            "min_liquidity": 5_000.0,  # Below threshold
            "max_slippage": 0.01  # Normal slippage
        }
        
        simple_path = SearchPath(nodes=[PathNode("ETH", 1.0)], total_score=0.0)
        adjustment = await scorer._calculate_risk_adjustments(simple_path, liquidity_metrics)
        
        assert adjustment < 0  # Should be negative (penalty)
    
    @pytest.mark.asyncio
    async def test_risk_adjustments_high_slippage(self, scorer):
        """Test risk adjustments for high slippage."""
        liquidity_metrics = {
            "min_liquidity": 50_000.0,  # Good liquidity
            "max_slippage": 0.05  # High slippage (> 2% threshold)
        }
        
        simple_path = SearchPath(nodes=[PathNode("ETH", 1.0)], total_score=0.0)
        adjustment = await scorer._calculate_risk_adjustments(simple_path, liquidity_metrics)
        
        assert adjustment < 0  # Should be negative (penalty)
    
    def test_volatility_penalty_stable_assets(self, scorer):
        """Test volatility penalty with stable assets."""
        # Path with mostly stable assets
        stable_path = SearchPath(
            nodes=[
                PathNode("USDC", 1000.0),
                PathNode("USDT", 1000.0),
                PathNode("DAI", 1000.0)
            ],
            total_score=0.0
        )
        
        penalty = scorer._calculate_volatility_penalty(stable_path)
        assert penalty < 0.05  # Should have low penalty for stable assets
    
    def test_volatility_penalty_volatile_assets(self, scorer):
        """Test volatility penalty with volatile assets."""
        # Path with mostly volatile assets
        volatile_path = SearchPath(
            nodes=[
                PathNode("ETH", 1.0),
                PathNode("WBTC", 0.05),
                PathNode("LINK", 100.0)
            ],
            total_score=0.0
        )
        
        penalty = scorer._calculate_volatility_penalty(volatile_path)
        assert penalty > 0.05  # Should have higher penalty for volatile assets
    
    def test_execution_risk_penalty(self, long_path, scorer):
        """Test execution risk penalty calculation."""
        penalty = scorer._calculate_execution_risk_penalty(long_path)
        
        assert penalty > 0  # Long path should have execution risk
        # Should account for length, confidence, and gas factors
    
    def test_win_probability_estimation(self, simple_path, scorer):
        """Test win probability estimation."""
        probability = scorer._estimate_win_probability(simple_path)
        
        assert 0.1 <= probability <= 0.99
        # Good path should have high probability
        assert probability > 0.5
    
    def test_win_probability_low_confidence(self, low_confidence_path, scorer):
        """Test win probability with low confidence path."""
        probability = scorer._estimate_win_probability(low_confidence_path)
        
        assert 0.1 <= probability <= 0.99
        # Low confidence should reduce probability
        assert probability < 0.7
    
    def test_profitability_score_normalization(self, scorer):
        """Test profitability score normalization."""
        # Test various profit levels
        high_profit_score = scorer._normalize_profitability_score(100.0, 1000.0)  # 10% profit
        low_profit_score = scorer._normalize_profitability_score(1.0, 1000.0)    # 0.1% profit
        zero_profit_score = scorer._normalize_profitability_score(0.0, 1000.0)   # 0% profit
        
        assert high_profit_score > low_profit_score > zero_profit_score
        assert 0 <= high_profit_score <= 1
        assert 0 <= low_profit_score <= 1
        assert zero_profit_score >= 0
    
    def test_sigmoid_normalization(self, scorer):
        """Test sigmoid normalization function."""
        # Test sigmoid with different parameters
        low_value = scorer._sigmoid_normalize(0.0, 0.5, 0.1)
        mid_value = scorer._sigmoid_normalize(0.5, 0.5, 0.1)
        high_value = scorer._sigmoid_normalize(1.0, 0.5, 0.1)
        
        assert 0 <= low_value <= 1
        assert 0 <= mid_value <= 1
        assert 0 <= high_value <= 1
        assert low_value < mid_value < high_value
        assert abs(mid_value - 0.5) < 0.1  # Should be close to midpoint
    
    def test_path_hash_computation(self, simple_path, scorer):
        """Test path hash computation for caching."""
        hash1 = scorer._compute_path_hash(simple_path)
        hash2 = scorer._compute_path_hash(simple_path)
        
        assert hash1 == hash2  # Same path should produce same hash
        assert isinstance(hash1, str)
    
    @pytest.mark.asyncio
    async def test_score_caching(self, simple_path, scorer):
        """Test score caching functionality."""
        # First call should compute score
        breakdown1 = await scorer.score_path(simple_path, "ETH", 1.0)
        
        # Second call should use cache
        breakdown2 = await scorer.score_path(simple_path, "ETH", 1.0)
        
        # Cache hit should be recorded
        stats = scorer.get_scoring_stats()
        assert stats["cache_hits"] > 0
        assert len(scorer._score_cache) > 0
    
    def test_cache_expiry(self, simple_path, scorer):
        """Test cache expiry functionality."""
        # Set very short cache TTL
        scorer._cache_ttl = 0.1  # 0.1 seconds
        
        path_hash = scorer._compute_path_hash(simple_path)
        
        # Store in cache
        test_breakdown = PathScoreBreakdown(
            total_score=1.0, profitability_score=0.0, liquidity_score=0.0,
            confidence_score=0.0, gas_efficiency_score=0.0, time_penalty=0.0,
            risk_adjustment=0.0, expected_profit_usd=0.0, expected_gas_cost_usd=0.0,
            total_liquidity_usd=0.0, min_edge_confidence=0.0, max_slippage_estimate=0.0,
            path_length=0, warnings=[], risk_flags=[]
        )
        scorer._cache_score(path_hash, test_breakdown)
        
        # Should be in cache
        cached = scorer._get_cached_score(path_hash)
        assert cached is not None
        
        # Wait for expiry
        time.sleep(0.2)
        
        # Should be expired
        cached_expired = scorer._get_cached_score(path_hash)
        assert cached_expired is None
    
    def test_cache_size_limit(self, scorer):
        """Test cache size limiting."""
        # Fill cache beyond limit
        for i in range(1100):  # More than 1000 limit
            path_hash = f"test_path_{i}"
            test_breakdown = PathScoreBreakdown(
                total_score=1.0, profitability_score=0.0, liquidity_score=0.0,
                confidence_score=0.0, gas_efficiency_score=0.0, time_penalty=0.0,
                risk_adjustment=0.0, expected_profit_usd=0.0, expected_gas_cost_usd=0.0,
                total_liquidity_usd=0.0, min_edge_confidence=0.0, max_slippage_estimate=0.0,
                path_length=0, warnings=[], risk_flags=[]
            )
            scorer._cache_score(path_hash, test_breakdown)
        
        # Cache should be limited
        assert len(scorer._score_cache) <= 1000
    
    def test_get_scoring_stats(self, scorer):
        """Test scoring statistics retrieval."""
        stats = scorer.get_scoring_stats()
        
        assert "paths_scored" in stats
        assert "cache_hits" in stats
        assert "cache_hit_rate" in stats
        assert "cache_size" in stats
        assert "method_distribution" in stats
        
        assert isinstance(stats["cache_hit_rate"], str)
        assert "%" in stats["cache_hit_rate"]
    
    def test_clear_cache(self, simple_path, scorer):
        """Test cache clearing functionality."""
        # Add something to cache
        path_hash = scorer._compute_path_hash(simple_path)
        test_breakdown = PathScoreBreakdown(
            total_score=1.0, profitability_score=0.0, liquidity_score=0.0,
            confidence_score=0.0, gas_efficiency_score=0.0, time_penalty=0.0,
            risk_adjustment=0.0, expected_profit_usd=0.0, expected_gas_cost_usd=0.0,
            total_liquidity_usd=0.0, min_edge_confidence=0.0, max_slippage_estimate=0.0,
            path_length=0, warnings=[], risk_flags=[]
        )
        scorer._cache_score(path_hash, test_breakdown)
        
        assert len(scorer._score_cache) > 0
        
        scorer.clear_cache()
        
        assert len(scorer._score_cache) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_scoring(self, scorer):
        """Test error handling in scoring methods."""
        # Create invalid path that might cause errors
        invalid_path = SearchPath(nodes=[], total_score=0.0)
        
        # Should handle gracefully without crashing
        breakdown = await scorer.score_path(invalid_path, "ETH", 1.0)
        
        assert isinstance(breakdown, PathScoreBreakdown)
        assert breakdown.total_score >= 0
    
    @pytest.mark.asyncio
    async def test_scoring_method_distribution_tracking(self, simple_path):
        """Test that scoring method usage is tracked."""
        # Test different methods
        methods_to_test = [
            ScoringMethod.SIMPLE_PROFIT,
            ScoringMethod.COMPOSITE,
            ScoringMethod.RISK_ADJUSTED
        ]
        
        for method in methods_to_test:
            config = ScoringConfig(method=method)
            scorer = NonMLPathScorer(config)
            
            await scorer.score_path(simple_path, "ETH", 1.0)
            
            stats = scorer.get_scoring_stats()
            assert stats["method_distribution"][method.value] == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_scoring_operations(self, simple_path, partial_path, scorer):
        """Test concurrent scoring operations."""
        import asyncio
        
        # Create multiple scoring tasks
        tasks = [
            scorer.score_path(simple_path, "ETH", 1.0),
            scorer.score_path(partial_path, "ETH", 1.0),
            scorer.score_path(simple_path, "USDC", 1.0)
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, PathScoreBreakdown)
            assert result.total_score >= 0
    
    @pytest.mark.asyncio
    async def test_different_initial_amounts(self, simple_path, scorer):
        """Test scoring with different initial amounts."""
        amounts = [0.1, 1.0, 10.0, 100.0]
        
        for amount in amounts:
            breakdown = await scorer.score_path(simple_path, "ETH", amount)
            
            assert breakdown.total_score >= 0
            # Profitability should scale with amount
            assert breakdown.expected_profit_usd != 0
    
    @pytest.mark.asyncio
    async def test_gas_efficiency_edge_cases(self, scorer):
        """Test gas efficiency calculation edge cases."""
        # High gas cost path
        high_gas_path = SearchPath(
            nodes=[
                PathNode("ETH", 1.0, 0.0, 1.0, []),
                PathNode("ETH", 1.001, 500.0, 0.9, ["expensive_edge"])  # Very high gas
            ],
            total_score=0.0,
            status=PathStatus.COMPLETE
        )
        
        breakdown = await scorer.score_path(high_gas_path, "ETH", 1.0)
        
        # Should have very low gas efficiency score
        assert breakdown.gas_efficiency_score < 0.5
        assert "High gas cost" in str(breakdown.risk_flags)
    
    @pytest.mark.asyncio
    async def test_zero_profit_path_scoring(self, scorer):
        """Test scoring of zero-profit paths."""
        zero_profit_path = SearchPath(
            nodes=[
                PathNode("ETH", 1.0, 0.0, 1.0, []),
                PathNode("ETH", 1.0, 15.0, 0.9, ["neutral_edge"])  # No profit, just gas cost
            ],
            total_score=0.0,
            status=PathStatus.COMPLETE
        )
        
        breakdown = await scorer.score_path(zero_profit_path, "ETH", 1.0)
        
        # Should have low score due to no profit
        assert breakdown.total_score < 0.5
        assert breakdown.expected_profit_usd <= 0