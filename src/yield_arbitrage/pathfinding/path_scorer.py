"""Non-ML path scoring algorithms for yield arbitrage pathfinding."""
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

from ..graph_engine.models import YieldGraphEdge, EdgeState, EdgeType
from .path_models import SearchPath, PathNode

logger = logging.getLogger(__name__)


class ScoringMethod(str, Enum):
    """Different scoring methods available."""
    SIMPLE_PROFIT = "simple_profit"
    RISK_ADJUSTED = "risk_adjusted"
    COMPOSITE = "composite"
    KELLY_CRITERION = "kelly_criterion"


@dataclass
class ScoringConfig:
    """Configuration for path scoring algorithms."""
    # Scoring method
    method: ScoringMethod = ScoringMethod.COMPOSITE
    
    # Weight factors for composite scoring
    profitability_weight: float = 0.4
    liquidity_weight: float = 0.25
    confidence_weight: float = 0.15
    gas_efficiency_weight: float = 0.15
    time_penalty_weight: float = 0.05
    
    # Risk parameters
    max_acceptable_slippage: float = 0.02  # 2%
    min_liquidity_threshold: float = 10_000.0  # $10k
    confidence_decay_factor: float = 0.95  # Per edge confidence decay
    
    # Gas efficiency parameters
    gas_price_gwei: float = 20.0
    eth_price_usd: float = 2000.0  # Dynamic in production
    
    # Path length penalties
    length_penalty_factor: float = 0.02  # 2% penalty per edge
    max_optimal_length: int = 4  # Optimal path length
    
    # Time-based scoring
    time_preference_half_life: float = 300.0  # 5 minutes
    
    # Liquidity impact parameters
    liquidity_impact_threshold: float = 0.01  # 1% of pool


@dataclass
class PathScoreBreakdown:
    """Detailed breakdown of path scoring components."""
    total_score: float
    profitability_score: float
    liquidity_score: float
    confidence_score: float
    gas_efficiency_score: float
    time_penalty: float
    risk_adjustment: float
    
    # Additional metrics
    expected_profit_usd: float
    expected_gas_cost_usd: float
    total_liquidity_usd: float
    min_edge_confidence: float
    max_slippage_estimate: float
    path_length: int
    
    # Warnings and flags
    warnings: List[str]
    risk_flags: List[str]


class NonMLPathScorer:
    """
    Non-ML path scoring system for yield arbitrage opportunities.
    
    This class implements various scoring algorithms that evaluate paths
    based on fundamental metrics without machine learning:
    - Profitability potential
    - Liquidity depth and impact
    - Gas efficiency
    - Confidence and reliability
    - Time preferences
    - Risk adjustments
    """
    
    def __init__(self, config: ScoringConfig = None):
        """
        Initialize the path scorer.
        
        Args:
            config: Scoring configuration parameters
        """
        self.config = config or ScoringConfig()
        
        # Scoring cache for performance
        self._score_cache: Dict[str, Tuple[float, float]] = {}  # path_hash -> (score, timestamp)
        self._cache_ttl = 60.0  # 1 minute cache
        
        # Statistics tracking
        self._scoring_stats = {
            "paths_scored": 0,
            "cache_hits": 0,
            "method_distribution": {method.value: 0 for method in ScoringMethod}
        }
    
    async def score_path(
        self, 
        path: SearchPath, 
        target_asset_id: str,
        initial_amount: float = 1.0
    ) -> PathScoreBreakdown:
        """
        Score a complete or partial path.
        
        Args:
            path: The path to score
            target_asset_id: Target asset for arbitrage completion
            initial_amount: Initial amount for profitability calculations
            
        Returns:
            Detailed scoring breakdown
        """
        self._scoring_stats["paths_scored"] += 1
        self._scoring_stats["method_distribution"][self.config.method.value] += 1
        
        # Check cache first
        path_hash = self._compute_path_hash(path)
        cached_result = self._get_cached_score(path_hash)
        if cached_result:
            self._scoring_stats["cache_hits"] += 1
            return cached_result
        
        # Calculate score based on method
        if self.config.method == ScoringMethod.SIMPLE_PROFIT:
            breakdown = await self._score_simple_profit(path, target_asset_id, initial_amount)
        elif self.config.method == ScoringMethod.RISK_ADJUSTED:
            breakdown = await self._score_risk_adjusted(path, target_asset_id, initial_amount)
        elif self.config.method == ScoringMethod.KELLY_CRITERION:
            breakdown = await self._score_kelly_criterion(path, target_asset_id, initial_amount)
        else:  # COMPOSITE
            breakdown = await self._score_composite(path, target_asset_id, initial_amount)
        
        # Cache result
        self._cache_score(path_hash, breakdown)
        
        return breakdown
    
    async def _score_composite(
        self, 
        path: SearchPath, 
        target_asset_id: str, 
        initial_amount: float
    ) -> PathScoreBreakdown:
        """
        Composite scoring using multiple weighted factors.
        
        This is the most sophisticated non-ML scoring method.
        """
        warnings = []
        risk_flags = []
        
        # Basic path metrics
        path_length = path.path_length
        is_complete = (path.end_asset == target_asset_id)
        
        # 1. Profitability Score
        profitability_metrics = await self._calculate_profitability_metrics(
            path, initial_amount, is_complete
        )
        
        expected_profit = profitability_metrics["net_profit_usd"]
        profitability_score = self._normalize_profitability_score(expected_profit, initial_amount)
        
        # 2. Liquidity Score
        liquidity_metrics = await self._calculate_liquidity_metrics(path, initial_amount)
        liquidity_score = liquidity_metrics["normalized_score"]
        
        if liquidity_metrics["min_liquidity"] < self.config.min_liquidity_threshold:
            warnings.append(f"Low liquidity: {liquidity_metrics['min_liquidity']:.0f} USD")
        
        # 3. Confidence Score
        confidence_metrics = self._calculate_confidence_metrics(path)
        confidence_score = confidence_metrics["normalized_score"]
        
        if confidence_metrics["min_confidence"] < 0.7:
            warnings.append(f"Low confidence: {confidence_metrics['min_confidence']:.2f}")
        
        # 4. Gas Efficiency Score
        gas_metrics = await self._calculate_gas_efficiency_metrics(path, expected_profit)
        gas_efficiency_score = gas_metrics["normalized_score"]
        
        if gas_metrics["gas_to_profit_ratio"] > 0.1:  # Gas > 10% of profit
            risk_flags.append("High gas cost relative to profit")
        
        # 5. Time Penalty
        time_penalty = self._calculate_time_penalty(path)
        
        # 6. Length Penalty
        length_penalty = self._calculate_length_penalty(path_length)
        
        # 7. Risk Adjustments
        risk_adjustment = await self._calculate_risk_adjustments(path, liquidity_metrics)
        
        # Combine scores with weights
        total_score = (
            profitability_score * self.config.profitability_weight +
            liquidity_score * self.config.liquidity_weight +
            confidence_score * self.config.confidence_weight +
            gas_efficiency_score * self.config.gas_efficiency_weight -
            time_penalty * self.config.time_penalty_weight -
            length_penalty * 0.1 +
            risk_adjustment
        )
        
        # Ensure non-negative score
        total_score = max(0.0, total_score)
        
        return PathScoreBreakdown(
            total_score=total_score,
            profitability_score=profitability_score,
            liquidity_score=liquidity_score,
            confidence_score=confidence_score,
            gas_efficiency_score=gas_efficiency_score,
            time_penalty=time_penalty,
            risk_adjustment=risk_adjustment,
            expected_profit_usd=expected_profit,
            expected_gas_cost_usd=gas_metrics["total_gas_cost"],
            total_liquidity_usd=liquidity_metrics["total_liquidity"],
            min_edge_confidence=confidence_metrics["min_confidence"],
            max_slippage_estimate=liquidity_metrics["max_slippage"],
            path_length=path_length,
            warnings=warnings,
            risk_flags=risk_flags
        )
    
    async def _score_simple_profit(
        self, 
        path: SearchPath, 
        target_asset_id: str, 
        initial_amount: float
    ) -> PathScoreBreakdown:
        """Simple profit-based scoring."""
        # Basic profit calculation
        final_amount = path.final_amount
        gas_cost_usd = path.total_gas_cost
        
        if path.end_asset == target_asset_id:
            # Complete arbitrage path
            profit_usd = final_amount - initial_amount - gas_cost_usd
        else:
            # Partial path - estimate potential
            profit_usd = (final_amount - initial_amount) * 0.5 - gas_cost_usd
        
        # Simple scoring: profit with basic penalties
        score = profit_usd - (path.path_length * 0.001)
        
        return PathScoreBreakdown(
            total_score=max(0.0, score),
            profitability_score=profit_usd,
            liquidity_score=1.0,
            confidence_score=1.0,
            gas_efficiency_score=1.0,
            time_penalty=0.0,
            risk_adjustment=0.0,
            expected_profit_usd=profit_usd,
            expected_gas_cost_usd=gas_cost_usd,
            total_liquidity_usd=0.0,
            min_edge_confidence=1.0,
            max_slippage_estimate=0.0,
            path_length=path.path_length,
            warnings=[],
            risk_flags=[]
        )
    
    async def _score_risk_adjusted(
        self, 
        path: SearchPath, 
        target_asset_id: str, 
        initial_amount: float
    ) -> PathScoreBreakdown:
        """Risk-adjusted scoring with volatility and confidence factors."""
        composite_score = await self._score_composite(path, target_asset_id, initial_amount)
        
        # Apply additional risk adjustments
        volatility_penalty = self._calculate_volatility_penalty(path)
        execution_risk_penalty = self._calculate_execution_risk_penalty(path)
        
        # Adjust total score
        risk_adjusted_score = (
            composite_score.total_score * 
            (1 - volatility_penalty) * 
            (1 - execution_risk_penalty)
        )
        
        # Update risk adjustment
        total_risk_adjustment = (
            composite_score.risk_adjustment - 
            volatility_penalty - 
            execution_risk_penalty
        )
        
        composite_score.total_score = max(0.0, risk_adjusted_score)
        composite_score.risk_adjustment = total_risk_adjustment
        
        return composite_score
    
    async def _score_kelly_criterion(
        self, 
        path: SearchPath, 
        target_asset_id: str, 
        initial_amount: float
    ) -> PathScoreBreakdown:
        """Kelly Criterion-based scoring for optimal position sizing."""
        # Get base scoring
        base_score = await self._score_composite(path, target_asset_id, initial_amount)
        
        # Calculate Kelly fraction
        win_probability = self._estimate_win_probability(path)
        win_amount = base_score.expected_profit_usd
        loss_amount = base_score.expected_gas_cost_usd + (initial_amount * 0.1)  # Potential loss
        
        if loss_amount <= 0:
            kelly_fraction = 1.0
        else:
            kelly_fraction = max(0.0, (win_probability * win_amount - (1 - win_probability) * loss_amount) / win_amount)
        
        # Adjust score by Kelly fraction
        kelly_adjusted_score = base_score.total_score * kelly_fraction
        
        base_score.total_score = kelly_adjusted_score
        base_score.risk_adjustment += kelly_fraction - 1.0
        
        return base_score
    
    async def _calculate_profitability_metrics(
        self, 
        path: SearchPath, 
        initial_amount: float, 
        is_complete: bool
    ) -> Dict[str, float]:
        """Calculate profitability-related metrics."""
        final_amount = path.final_amount
        gas_cost_usd = path.total_gas_cost
        
        if is_complete:
            # Complete arbitrage path
            gross_profit = final_amount - initial_amount
            net_profit = gross_profit - gas_cost_usd
            profit_ratio = net_profit / initial_amount if initial_amount > 0 else 0.0
        else:
            # Partial path - estimate completion potential
            estimated_completion_factor = 0.7  # Conservative estimate
            gross_profit = (final_amount - initial_amount) * estimated_completion_factor
            net_profit = gross_profit - gas_cost_usd
            profit_ratio = net_profit / initial_amount if initial_amount > 0 else 0.0
        
        return {
            "gross_profit_usd": gross_profit,
            "net_profit_usd": net_profit,
            "profit_ratio": profit_ratio,
            "gas_cost_usd": gas_cost_usd
        }
    
    async def _calculate_liquidity_metrics(
        self, 
        path: SearchPath, 
        trade_amount: float
    ) -> Dict[str, float]:
        """Calculate liquidity-related metrics."""
        total_liquidity = 0.0
        min_liquidity = float('inf')
        max_slippage = 0.0
        liquidity_scores = []
        
        for i, node in enumerate(path.nodes[:-1]):  # Skip last node
            edge_id = node.edge_path[-1] if node.edge_path else None
            if not edge_id:
                continue
            
            # Mock edge state - in production, would get from EdgeStateManager
            edge_liquidity = 1_000_000.0  # Default liquidity
            
            total_liquidity += edge_liquidity
            min_liquidity = min(min_liquidity, edge_liquidity)
            
            # Estimate slippage based on trade size vs liquidity
            trade_impact = trade_amount / edge_liquidity if edge_liquidity > 0 else 1.0
            slippage = min(trade_impact * 0.1, 0.1)  # Cap at 10%
            max_slippage = max(max_slippage, slippage)
            
            # Liquidity score (0-1)
            liquidity_score = self._sigmoid_normalize(edge_liquidity, 100_000, 10_000_000)
            liquidity_scores.append(liquidity_score)
        
        if min_liquidity == float('inf'):
            min_liquidity = 0.0
        
        # Overall liquidity score (geometric mean)
        if liquidity_scores:
            normalized_score = math.pow(math.prod(liquidity_scores), 1.0 / len(liquidity_scores))
        else:
            normalized_score = 0.0
        
        return {
            "total_liquidity": total_liquidity,
            "min_liquidity": min_liquidity,
            "max_slippage": max_slippage,
            "normalized_score": normalized_score
        }
    
    def _calculate_confidence_metrics(self, path: SearchPath) -> Dict[str, float]:
        """Calculate confidence-related metrics."""
        if not path.nodes:
            return {"min_confidence": 0.0, "avg_confidence": 0.0, "normalized_score": 0.0}
        
        # Get confidence from path nodes
        confidences = [node.confidence_accumulated for node in path.nodes if node.confidence_accumulated > 0]
        
        if not confidences:
            return {"min_confidence": 0.0, "avg_confidence": 0.0, "normalized_score": 0.0}
        
        min_confidence = min(confidences)
        avg_confidence = sum(confidences) / len(confidences)
        
        # Apply confidence decay for path length
        length_decay = math.pow(self.config.confidence_decay_factor, path.path_length)
        adjusted_confidence = min_confidence * length_decay
        
        return {
            "min_confidence": min_confidence,
            "avg_confidence": avg_confidence,
            "normalized_score": adjusted_confidence
        }
    
    async def _calculate_gas_efficiency_metrics(
        self, 
        path: SearchPath, 
        expected_profit: float
    ) -> Dict[str, float]:
        """Calculate gas efficiency metrics."""
        total_gas_cost = path.total_gas_cost
        
        # Gas to profit ratio
        if expected_profit > 0:
            gas_to_profit_ratio = total_gas_cost / expected_profit
        else:
            gas_to_profit_ratio = float('inf')
        
        # Gas efficiency score (inverse of gas cost, normalized)
        if total_gas_cost > 0:
            efficiency = 1.0 / (1.0 + gas_to_profit_ratio)
        else:
            efficiency = 1.0
        
        # Normalize efficiency score
        normalized_score = self._sigmoid_normalize(efficiency, 0.5, 0.9)
        
        return {
            "total_gas_cost": total_gas_cost,
            "gas_to_profit_ratio": gas_to_profit_ratio,
            "efficiency": efficiency,
            "normalized_score": normalized_score
        }
    
    def _calculate_time_penalty(self, path: SearchPath) -> float:
        """Calculate time-based penalty for path freshness."""
        if not path.creation_time:
            return 0.0
        
        age_seconds = time.time() - path.creation_time
        
        # Exponential decay based on half-life
        decay_factor = math.pow(0.5, age_seconds / self.config.time_preference_half_life)
        penalty = 1.0 - decay_factor
        
        return penalty
    
    def _calculate_length_penalty(self, path_length: int) -> float:
        """Calculate penalty for path length."""
        if path_length <= self.config.max_optimal_length:
            return 0.0
        
        excess_length = path_length - self.config.max_optimal_length
        penalty = excess_length * self.config.length_penalty_factor
        
        return penalty
    
    async def _calculate_risk_adjustments(
        self, 
        path: SearchPath, 
        liquidity_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall risk adjustments."""
        adjustments = 0.0
        
        # Low liquidity penalty
        if liquidity_metrics["min_liquidity"] < self.config.min_liquidity_threshold:
            liquidity_penalty = -0.2  # 20% penalty
            adjustments += liquidity_penalty
        
        # High slippage penalty
        if liquidity_metrics["max_slippage"] > self.config.max_acceptable_slippage:
            slippage_penalty = -liquidity_metrics["max_slippage"] * 2.0
            adjustments += slippage_penalty
        
        # Path complexity penalty
        if path.path_length > 6:
            complexity_penalty = -(path.path_length - 6) * 0.05
            adjustments += complexity_penalty
        
        return adjustments
    
    def _calculate_volatility_penalty(self, path: SearchPath) -> float:
        """Calculate volatility-based penalty."""
        # Simplified volatility estimation based on asset types
        # In production, would use historical volatility data
        volatile_assets = 0
        stable_assets = 0
        
        for node in path.nodes:
            asset_id = node.asset_id.lower()
            if any(stable in asset_id for stable in ['usdc', 'usdt', 'dai', 'busd']):
                stable_assets += 1
            else:
                volatile_assets += 1
        
        total_assets = volatile_assets + stable_assets
        if total_assets == 0:
            return 0.0
        
        volatility_ratio = volatile_assets / total_assets
        penalty = volatility_ratio * 0.1  # Up to 10% penalty
        
        return penalty
    
    def _calculate_execution_risk_penalty(self, path: SearchPath) -> float:
        """Calculate execution risk penalty."""
        # Factors that increase execution risk
        risk_factors = []
        
        # Long paths are riskier
        if path.path_length > 5:
            risk_factors.append(0.02 * (path.path_length - 5))
        
        # Low confidence is risky
        min_confidence = min([node.confidence_accumulated for node in path.nodes], default=1.0)
        if min_confidence < 0.8:
            risk_factors.append(0.05 * (0.8 - min_confidence))
        
        # High gas cost relative to amount is risky
        if path.nodes and path.total_gas_cost > path.nodes[0].amount * 0.05:
            risk_factors.append(0.03)
        
        return sum(risk_factors)
    
    def _estimate_win_probability(self, path: SearchPath) -> float:
        """Estimate probability of successful execution."""
        # Base probability
        base_prob = 0.8
        
        # Adjust based on confidence
        min_confidence = min([node.confidence_accumulated for node in path.nodes], default=1.0)
        confidence_factor = min_confidence
        
        # Adjust based on path length (longer paths are riskier)
        length_factor = math.pow(0.95, path.path_length)
        
        # Adjust based on gas efficiency
        gas_factor = 1.0
        if path.nodes and path.total_gas_cost > 0:
            initial_amount = path.nodes[0].amount
            if path.total_gas_cost / initial_amount > 0.02:  # Gas > 2% of amount
                gas_factor = 0.9
        
        probability = base_prob * confidence_factor * length_factor * gas_factor
        return max(0.1, min(0.99, probability))
    
    def _normalize_profitability_score(self, profit_usd: float, initial_amount: float) -> float:
        """Normalize profitability score to 0-1 range."""
        if initial_amount <= 0:
            return 0.0
        
        profit_ratio = profit_usd / initial_amount
        
        # Use sigmoid to normalize profit ratio
        return self._sigmoid_normalize(profit_ratio, 0.01, 0.1)  # 1% to 10% profit
    
    def _sigmoid_normalize(self, value: float, midpoint: float, scale: float) -> float:
        """Normalize value using sigmoid function."""
        if scale <= 0:
            return 0.5
        
        normalized = 1.0 / (1.0 + math.exp(-(value - midpoint) / scale))
        return max(0.0, min(1.0, normalized))
    
    def _compute_path_hash(self, path: SearchPath) -> str:
        """Compute hash for path caching."""
        # Simple hash based on path structure
        path_str = "_".join([
            path.start_asset,
            path.end_asset,
            str(path.path_length),
            str(int(path.final_amount * 1000))  # Round for caching
        ])
        return str(hash(path_str))
    
    def _get_cached_score(self, path_hash: str) -> Optional[PathScoreBreakdown]:
        """Get cached score if valid."""
        if path_hash not in self._score_cache:
            return None
        
        score, timestamp = self._score_cache[path_hash]
        
        if time.time() - timestamp > self._cache_ttl:
            del self._score_cache[path_hash]
            return None
        
        # Return cached score (simplified for caching)
        return PathScoreBreakdown(
            total_score=score,
            profitability_score=0.0,
            liquidity_score=0.0,
            confidence_score=0.0,
            gas_efficiency_score=0.0,
            time_penalty=0.0,
            risk_adjustment=0.0,
            expected_profit_usd=0.0,
            expected_gas_cost_usd=0.0,
            total_liquidity_usd=0.0,
            min_edge_confidence=0.0,
            max_slippage_estimate=0.0,
            path_length=0,
            warnings=[],
            risk_flags=[]
        )
    
    def _cache_score(self, path_hash: str, breakdown: PathScoreBreakdown) -> None:
        """Cache score result."""
        self._score_cache[path_hash] = (breakdown.total_score, time.time())
        
        # Limit cache size
        if len(self._score_cache) > 1000:
            # Remove oldest entries
            oldest_entries = sorted(
                self._score_cache.items(),
                key=lambda x: x[1][1]
            )[:100]
            
            for key, _ in oldest_entries:
                del self._score_cache[key]
    
    def get_scoring_stats(self) -> Dict[str, Any]:
        """Get scoring performance statistics."""
        total_scored = self._scoring_stats["paths_scored"]
        cache_hit_rate = (
            (self._scoring_stats["cache_hits"] / total_scored * 100) 
            if total_scored > 0 else 0.0
        )
        
        return {
            "paths_scored": total_scored,
            "cache_hits": self._scoring_stats["cache_hits"],
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "cache_size": len(self._score_cache),
            "method_distribution": self._scoring_stats["method_distribution"]
        }
    
    def clear_cache(self) -> None:
        """Clear scoring cache."""
        self._score_cache.clear()
        logger.info("Scoring cache cleared")