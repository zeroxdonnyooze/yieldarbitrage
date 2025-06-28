"""
MEV Opportunity Detector.

This module combines mempool monitoring and transaction analysis to detect
and validate MEV opportunities in real-time.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from decimal import Decimal

from .mempool_monitor import MempoolMonitor, TransactionEvent, TransactionEventType
from .transaction_analyzer import TransactionAnalyzer, TransactionImpact, TransactionCategory
from .opportunity_models import (
    MEVOpportunity, BackRunOpportunity, SandwichOpportunity, ArbitrageOpportunity,
    MEVOpportunityType, OpportunityStatus, OpportunityQueue
)

# Import existing MEV infrastructure
from yield_arbitrage.mev_protection.mev_risk_assessor import MEVRiskAssessor, MEVRiskLevel
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType, BackRunEdge

logger = logging.getLogger(__name__)


@dataclass
class OpportunityDetectionConfig:
    """Configuration for MEV opportunity detection."""
    
    # Detection thresholds
    min_profit_usd: float = 50.0
    min_confidence_score: float = 0.6
    max_gas_cost_ratio: float = 0.3  # Max 30% of profit can be gas
    
    # Opportunity types to detect
    detect_back_runs: bool = True
    detect_sandwich_attacks: bool = True
    detect_arbitrage: bool = True
    detect_liquidations: bool = True
    
    # Risk parameters
    max_capital_requirement: float = 1000000.0  # $1M max capital
    min_mev_risk_score: float = 0.5
    max_execution_time_ms: int = 2000  # 2 seconds max execution time
    
    # Performance settings
    max_opportunities_per_block: int = 10
    opportunity_ttl_seconds: int = 60
    batch_analysis_size: int = 50


@dataclass
class DetectedOpportunity:
    """Wrapper for detected MEV opportunity with additional detection metadata."""
    opportunity: MEVOpportunity
    detection_confidence: float
    detection_timestamp: float
    source_transaction: TransactionImpact
    estimated_execution_time: float = 0.0
    
    # Path information for execution
    suggested_path: Optional[List[YieldGraphEdge]] = None
    execution_strategy: Optional[Dict[str, Any]] = None


class MEVOpportunityDetector:
    """
    Real-time MEV opportunity detector that combines mempool monitoring,
    transaction analysis, and path finding to identify profitable MEV opportunities.
    """
    
    def __init__(
        self,
        config: OpportunityDetectionConfig,
        chain_id: int = 1
    ):
        """Initialize MEV opportunity detector."""
        self.config = config
        self.chain_id = chain_id
        
        # Core components
        self.mempool_monitor: Optional[MempoolMonitor] = None
        self.transaction_analyzer = TransactionAnalyzer(chain_id)
        self.mev_risk_assessor = MEVRiskAssessor()
        
        # Opportunity management
        self.opportunity_queue = OpportunityQueue(max_size=1000)
        self.detected_opportunities: Dict[str, DetectedOpportunity] = {}
        
        # Event handlers
        self.opportunity_handlers: List[Callable[[DetectedOpportunity], None]] = []
        
        # Statistics
        self.stats = {
            "opportunities_detected": 0,
            "opportunities_executed": 0,
            "total_profit_detected": 0.0,
            "total_profit_captured": 0.0,
            "detection_accuracy": 0.0,
            "average_detection_time": 0.0
        }
        
        # Running state
        self.is_running = False
        self.detection_tasks: List[asyncio.Task] = []
    
    async def start(self, mempool_monitor: MempoolMonitor):
        """Start MEV opportunity detection."""
        if self.is_running:
            logger.warning("MEV opportunity detector already running")
            return
        
        logger.info(f"Starting MEV opportunity detector for chain {self.chain_id}")
        self.mempool_monitor = mempool_monitor
        self.is_running = True
        
        # Register for mempool events
        self.mempool_monitor.add_event_handler(
            TransactionEventType.PENDING,
            self._handle_pending_transaction
        )
        
        # Start detection tasks
        self.detection_tasks = [
            asyncio.create_task(self._process_detection_queue()),
            asyncio.create_task(self._cleanup_expired_opportunities()),
            asyncio.create_task(self._update_statistics())
        ]
        
        logger.info("MEV opportunity detector started")
    
    async def stop(self):
        """Stop MEV opportunity detection."""
        logger.info("Stopping MEV opportunity detector")
        self.is_running = False
        
        # Cancel detection tasks
        for task in self.detection_tasks:
            task.cancel()
        
        await asyncio.gather(*self.detection_tasks, return_exceptions=True)
        self.detection_tasks.clear()
    
    def add_opportunity_handler(self, handler: Callable[[DetectedOpportunity], None]):
        """Add handler for detected opportunities."""
        self.opportunity_handlers.append(handler)
        logger.debug("Added opportunity handler")
    
    async def _handle_pending_transaction(self, event: TransactionEvent):
        """Handle new pending transaction from mempool."""
        try:
            # Analyze transaction impact
            impact = await self.transaction_analyzer.analyze_transaction(event.transaction_data)
            
            # Check if transaction has MEV potential
            mev_risk_score = impact.metadata.get("mev_risk_score", 0.0)
            if mev_risk_score < self.config.min_mev_risk_score:
                return
            
            # Detect specific MEV opportunities
            opportunities = await self._detect_opportunities(impact)
            
            # Process and queue opportunities
            for opportunity in opportunities:
                detected_opportunity = DetectedOpportunity(
                    opportunity=opportunity,
                    detection_confidence=self._calculate_detection_confidence(impact, opportunity),
                    detection_timestamp=time.time(),
                    source_transaction=impact
                )
                
                # Validate opportunity
                if await self._validate_opportunity(detected_opportunity):
                    # Add to queue and tracking
                    self.opportunity_queue.add_opportunity(opportunity)
                    self.detected_opportunities[opportunity.opportunity_id] = detected_opportunity
                    
                    # Trigger handlers
                    await self._trigger_opportunity_handlers(detected_opportunity)
                    
                    self.stats["opportunities_detected"] += 1
                    self.stats["total_profit_detected"] += opportunity.estimated_profit_usd
                    
                    logger.info(
                        f"Detected {opportunity.opportunity_type} opportunity: "
                        f"${opportunity.estimated_profit_usd:.2f} profit "
                        f"(confidence: {detected_opportunity.detection_confidence:.2f})"
                    )
        
        except Exception as e:
            logger.error(f"Error handling pending transaction {event.transaction_hash}: {e}")
    
    async def _detect_opportunities(self, impact: TransactionImpact) -> List[MEVOpportunity]:
        """Detect MEV opportunities from transaction impact."""
        opportunities = []
        
        try:
            # Back-run opportunities
            if self.config.detect_back_runs and impact.creates_arbitrage_opportunity:
                back_run = await self._detect_back_run_opportunity(impact)
                if back_run:
                    opportunities.append(back_run)
            
            # Sandwich opportunities
            if self.config.detect_sandwich_attacks and impact.sandwich_vulnerable:
                sandwich = await self._detect_sandwich_opportunity(impact)
                if sandwich:
                    opportunities.append(sandwich)
            
            # Cross-DEX arbitrage opportunities
            if self.config.detect_arbitrage and impact.category == TransactionCategory.DEX_TRADE:
                arbitrage = await self._detect_arbitrage_opportunity(impact)
                if arbitrage:
                    opportunities.append(arbitrage)
            
            # Liquidation opportunities
            if self.config.detect_liquidations and impact.liquidation_opportunity:
                liquidation = await self._detect_liquidation_opportunity(impact)
                if liquidation:
                    opportunities.append(liquidation)
        
        except Exception as e:
            logger.error(f"Error detecting opportunities for {impact.transaction_hash}: {e}")
        
        return opportunities
    
    async def _detect_back_run_opportunity(self, impact: TransactionImpact) -> Optional[BackRunOpportunity]:
        """Detect back-run opportunity from transaction impact."""
        try:
            # Estimate profit from price movement
            estimated_profit = impact.max_price_impact * impact.total_value_usd * 0.1  # Conservative estimate
            
            if estimated_profit < self.config.min_profit_usd:
                return None
            
            # Determine optimal assets for back-run
            affected_pools = impact.affected_pools
            if not affected_pools:
                return None
            
            pool = affected_pools[0]  # Use first affected pool
            
            opportunity_id = f"backrun_{impact.transaction_hash}_{int(time.time())}"
            
            return BackRunOpportunity(
                opportunity_id=opportunity_id,
                target_transaction_hash=impact.transaction_hash,
                estimated_profit_usd=estimated_profit,
                confidence_score=0.7,  # Default confidence for back-runs
                required_capital_usd=estimated_profit * 10,  # Rough capital requirement
                max_gas_price=int(200e9),  # 200 gwei max
                execution_deadline=time.time() + 60,  # 1 minute deadline
                chain_id=self.chain_id,
                involved_protocols=[pool.protocol],
                source_asset="WETH",  # Simplified - would determine optimal asset
                target_asset="USDC",  # Simplified - would determine optimal asset
                optimal_amount=impact.total_value_usd * 0.1,
                expected_price_movement=impact.max_price_impact,
                pool_address=pool.pool_address
            )
        
        except Exception as e:
            logger.error(f"Error detecting back-run opportunity: {e}")
            return None
    
    async def _detect_sandwich_opportunity(self, impact: TransactionImpact) -> Optional[SandwichOpportunity]:
        """Detect sandwich attack opportunity."""
        try:
            # Calculate extractable value from victim transaction
            victim_slippage = 0.005  # Assume 0.5% slippage tolerance
            extractable_value = impact.total_value_usd * victim_slippage * 0.5  # Conservative
            
            if extractable_value < self.config.min_profit_usd:
                return None
            
            opportunity_id = f"sandwich_{impact.transaction_hash}_{int(time.time())}"
            
            return SandwichOpportunity(
                opportunity_id=opportunity_id,
                target_transaction_hash=impact.transaction_hash,
                estimated_profit_usd=extractable_value,
                confidence_score=0.6,  # Lower confidence for sandwich attacks
                required_capital_usd=impact.total_value_usd,
                max_gas_price=int(300e9),  # Higher gas for sandwich
                execution_deadline=time.time() + 30,  # Shorter deadline
                chain_id=self.chain_id,
                frontrun_transaction={
                    "type": "frontrun",
                    "estimated_gas": 150000,
                    "priority_fee_multiplier": 1.1
                },
                backrun_transaction={
                    "type": "backrun", 
                    "estimated_gas": 150000,
                    "priority_fee_multiplier": 1.0
                },
                victim_slippage_tolerance=victim_slippage,
                extractable_value=extractable_value,
                pool_liquidity=impact.total_value_usd * 100,  # Estimate
                optimal_sandwich_size=impact.total_value_usd * 0.5
            )
        
        except Exception as e:
            logger.error(f"Error detecting sandwich opportunity: {e}")
            return None
    
    async def _detect_arbitrage_opportunity(self, impact: TransactionImpact) -> Optional[ArbitrageOpportunity]:
        """Detect cross-DEX arbitrage opportunity."""
        try:
            # Estimate price difference created by transaction
            price_diff = impact.max_price_impact
            if price_diff < 0.002:  # Less than 0.2% difference
                return None
            
            estimated_profit = impact.total_value_usd * price_diff * 0.5  # Conservative
            
            if estimated_profit < self.config.min_profit_usd:
                return None
            
            opportunity_id = f"arbitrage_{impact.transaction_hash}_{int(time.time())}"
            
            return ArbitrageOpportunity(
                opportunity_id=opportunity_id,
                target_transaction_hash=impact.transaction_hash,
                estimated_profit_usd=estimated_profit,
                confidence_score=0.8,  # Higher confidence for arbitrage
                required_capital_usd=impact.total_value_usd,
                max_gas_price=int(150e9),  # 150 gwei
                execution_deadline=time.time() + 45,
                chain_id=self.chain_id,
                source_dex="uniswap_v3",  # Would determine actual DEXs
                target_dex="sushiswap",
                arbitrage_token="WETH",  # Would determine actual token
                price_difference=price_diff,
                optimal_arbitrage_amount=impact.total_value_usd * 0.3,
                arbitrage_path=["WETH", "USDC", "WETH"],  # Simplified path
                execution_complexity=2
            )
        
        except Exception as e:
            logger.error(f"Error detecting arbitrage opportunity: {e}")
            return None
    
    async def _detect_liquidation_opportunity(self, impact: TransactionImpact) -> Optional[MEVOpportunity]:
        """Detect liquidation opportunity."""
        # This would integrate with lending protocol analysis
        # For now, return None as it requires more complex integration
        return None
    
    def _calculate_detection_confidence(
        self, 
        impact: TransactionImpact, 
        opportunity: MEVOpportunity
    ) -> float:
        """Calculate confidence score for detected opportunity."""
        base_confidence = opportunity.confidence_score
        
        # Adjust based on transaction analysis quality
        if impact.metadata.get("is_contract_interaction", False):
            base_confidence += 0.1
        
        # Adjust based on value size
        if impact.total_value_usd > 100000:  # $100k+
            base_confidence += 0.1
        
        # Adjust based on time sensitivity
        if impact.time_sensitivity > 0.8:
            base_confidence += 0.1
        
        # Cap at 1.0
        return min(1.0, base_confidence)
    
    async def _validate_opportunity(self, detected_opportunity: DetectedOpportunity) -> bool:
        """Validate detected opportunity meets criteria."""
        opportunity = detected_opportunity.opportunity
        
        # Check minimum profit
        if opportunity.estimated_profit_usd < self.config.min_profit_usd:
            return False
        
        # Check confidence score
        if detected_opportunity.detection_confidence < self.config.min_confidence_score:
            return False
        
        # Check capital requirements
        if opportunity.required_capital_usd > self.config.max_capital_requirement:
            return False
        
        # Estimate gas cost and check ratio
        estimated_gas_cost = 200000 * 50e9 / 1e18 * 2000  # Rough estimate
        if estimated_gas_cost / opportunity.estimated_profit_usd > self.config.max_gas_cost_ratio:
            return False
        
        # Check if opportunity hasn't expired
        if opportunity.is_expired():
            return False
        
        return True
    
    async def _trigger_opportunity_handlers(self, detected_opportunity: DetectedOpportunity):
        """Trigger registered opportunity handlers."""
        for handler in self.opportunity_handlers:
            try:
                await handler(detected_opportunity)
            except Exception as e:
                logger.error(f"Error in opportunity handler: {e}")
    
    async def _process_detection_queue(self):
        """Process opportunities in the detection queue."""
        while self.is_running:
            try:
                # Get next opportunity
                opportunity = self.opportunity_queue.get_next_opportunity()
                
                if opportunity:
                    # Additional processing could happen here
                    # For now, just log the opportunity
                    logger.debug(f"Processing opportunity: {opportunity.opportunity_id}")
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error processing detection queue: {e}")
                await asyncio.sleep(1)
    
    async def _cleanup_expired_opportunities(self):
        """Clean up expired opportunities."""
        while self.is_running:
            try:
                current_time = time.time()
                expired_ids = []
                
                for opp_id, detected_opp in self.detected_opportunities.items():
                    if (current_time - detected_opp.detection_timestamp > 
                        self.config.opportunity_ttl_seconds):
                        expired_ids.append(opp_id)
                
                # Remove expired opportunities
                for opp_id in expired_ids:
                    del self.detected_opportunities[opp_id]
                
                # Cleanup opportunity queue
                self.opportunity_queue.cleanup_expired()
                
                if expired_ids:
                    logger.debug(f"Cleaned up {len(expired_ids)} expired opportunities")
                
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in opportunity cleanup: {e}")
                await asyncio.sleep(30)
    
    async def _update_statistics(self):
        """Update detection statistics."""
        while self.is_running:
            try:
                # Update basic stats
                queue_stats = self.opportunity_queue.get_stats()
                self.stats.update(queue_stats)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error updating statistics: {e}")
                await asyncio.sleep(10)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            **self.stats,
            "detected_opportunities_count": len(self.detected_opportunities),
            "queue_size": len(self.opportunity_queue.opportunities),
            "is_running": self.is_running
        }
    
    def get_recent_opportunities(self, limit: int = 10) -> List[DetectedOpportunity]:
        """Get most recent detected opportunities."""
        opportunities = list(self.detected_opportunities.values())
        opportunities.sort(key=lambda x: x.detection_timestamp, reverse=True)
        return opportunities[:limit]
    
    def get_opportunities_by_type(
        self, 
        opportunity_type: MEVOpportunityType
    ) -> List[DetectedOpportunity]:
        """Get opportunities of a specific type."""
        return [
            detected_opp for detected_opp in self.detected_opportunities.values()
            if detected_opp.opportunity.opportunity_type == opportunity_type
        ]


# Convenience functions

async def create_mev_detector(
    chain_id: int = 1,
    mempool_monitor: Optional[MempoolMonitor] = None,
    **config_kwargs
) -> MEVOpportunityDetector:
    """Create and configure MEV opportunity detector."""
    config = OpportunityDetectionConfig(**config_kwargs)
    detector = MEVOpportunityDetector(config, chain_id)
    
    if mempool_monitor:
        await detector.start(mempool_monitor)
    
    return detector


async def start_mev_detection_pipeline(
    chain_id: int = 1,
    rpc_urls: Optional[List[str]] = None
) -> tuple[MempoolMonitor, MEVOpportunityDetector]:
    """Start complete MEV detection pipeline."""
    from .mempool_monitor import create_mempool_monitor
    
    # Create mempool monitor
    monitor = await create_mempool_monitor(chain_id, rpc_urls)
    
    # Create opportunity detector
    detector = await create_mev_detector(chain_id, monitor)
    
    # Start both components
    asyncio.create_task(monitor.start())
    
    return monitor, detector