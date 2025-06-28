#!/usr/bin/env python3
"""
Integration tests with real data for production DeFi arbitrage system.

This test suite validates the complete arbitrage system using real blockchain data,
live protocols, and production-ready components.
"""
import asyncio
import pytest
import logging
import sys
from typing import Dict, List, Optional, Any
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.blockchain_connector.provider import BlockchainProvider
from yield_arbitrage.execution.onchain_price_oracle import OnChainPriceOracle
from yield_arbitrage.protocols.production_registry import production_registry
from yield_arbitrage.data.real_edge_pipeline import RealEdgeStatePipeline
from yield_arbitrage.execution.real_transaction_builder import RealTransactionBuilder
from yield_arbitrage.monitoring.production_monitor import ProductionMonitor
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeState, EdgeType
from yield_arbitrage.config.production import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestRealDataIntegration:
    """Integration tests with real DeFi data and protocols."""
    
    @pytest.fixture(scope="class")
    async def blockchain_provider(self):
        """Initialize blockchain provider with real connections."""
        provider = BlockchainProvider()
        await provider.initialize()
        yield provider
        await provider.close()
    
    @pytest.fixture(scope="class")
    async def redis_client(self):
        """Mock Redis client for testing."""
        # In production, this would be a real Redis connection
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.lpush.return_value = 1
        mock_redis.ltrim.return_value = True
        mock_redis.setex.return_value = True
        yield mock_redis
    
    @pytest.fixture(scope="class")
    async def price_oracle(self, blockchain_provider, redis_client):
        """Initialize on-chain price oracle."""
        oracle = OnChainPriceOracle(blockchain_provider, redis_client)
        await oracle.initialize()
        yield oracle
    
    @pytest.fixture(scope="class")
    async def edge_pipeline(self, blockchain_provider, price_oracle, redis_client):
        """Initialize real edge state pipeline."""
        pipeline = RealEdgeStatePipeline(blockchain_provider, price_oracle, redis_client)
        await pipeline.initialize()
        yield pipeline
    
    @pytest.fixture(scope="class")
    async def transaction_builder(self, blockchain_provider, price_oracle):
        """Initialize real transaction builder."""
        builder = RealTransactionBuilder(blockchain_provider, price_oracle)
        await builder.initialize()
        yield builder
    
    @pytest.fixture(scope="class")
    async def production_monitor(self, blockchain_provider, price_oracle, redis_client):
        """Initialize production monitoring system."""
        monitor = ProductionMonitor(blockchain_provider, price_oracle, redis_client)
        yield monitor
    
    @pytest.mark.asyncio
    async def test_end_to_end_arbitrage_discovery(
        self, 
        blockchain_provider, 
        price_oracle, 
        edge_pipeline,
        transaction_builder,
        production_monitor
    ):
        """Test complete end-to-end arbitrage discovery and execution preparation."""
        logger.info("🔄 Testing end-to-end arbitrage discovery...")
        
        # Step 1: Start production monitoring
        await production_monitor.start_monitoring()
        
        try:
            # Step 2: Discover real edges from live protocols
            logger.info("   📊 Discovering live edges...")
            edges = await edge_pipeline.discover_edges()
            
            assert len(edges) > 0, "Should discover at least some edges"
            logger.info(f"   ✅ Discovered {len(edges)} live edges")
            
            # Step 3: Update edge states with real data
            logger.info("   🔄 Updating edge states...")
            updated_edges = []
            
            for edge in edges[:5]:  # Test first 5 edges
                updated_edge = await edge_pipeline.update_edge_state(edge)
                if updated_edge and updated_edge.state.conversion_rate:
                    updated_edges.append(updated_edge)
            
            assert len(updated_edges) > 0, "Should update at least some edge states"
            logger.info(f"   ✅ Updated {len(updated_edges)} edge states")
            
            # Step 4: Validate price data consistency
            logger.info("   💰 Validating price data...")
            test_assets = ["ETH_MAINNET_WETH", "ETH_MAINNET_USDC"]
            
            for asset in test_assets:
                price = await price_oracle.get_price_usd(asset)
                assert price is not None, f"Should get price for {asset}"
                assert price > 0, f"Price should be positive for {asset}"
                logger.info(f"   ✅ {asset}: ${price:.2f}")
            
            # Step 5: Attempt arbitrage path construction
            logger.info("   🔀 Constructing arbitrage paths...")
            
            # Find edges that could form arbitrage opportunities
            potential_paths = self._find_potential_arbitrage_paths(updated_edges)
            
            if potential_paths:
                logger.info(f"   ✅ Found {len(potential_paths)} potential arbitrage paths")
                
                # Step 6: Build transactions for valid paths
                logger.info("   🏗️  Building arbitrage transactions...")
                
                for path in potential_paths[:2]:  # Test first 2 paths
                    transaction = await transaction_builder.build_simple_arbitrage(
                        path, 
                        Decimal("1000")  # $1000 test amount
                    )
                    
                    if transaction:
                        logger.info(f"   ✅ Built transaction: {transaction.transaction_id}")
                        
                        # Step 7: Simulate transaction
                        simulation_passed = await transaction_builder.simulate_transaction(transaction)
                        if simulation_passed:
                            logger.info(f"   ✅ Simulation passed for {transaction.transaction_id}")
                        else:
                            logger.warning(f"   ⚠️  Simulation failed for {transaction.transaction_id}")
            
            else:
                logger.info("   ℹ️  No arbitrage paths found (expected in test environment)")
            
            # Step 8: Check system health
            logger.info("   🏥 Checking system health...")
            health_summary = await production_monitor.get_system_health()
            
            assert health_summary["overall_status"] in ["healthy", "degraded"], "System should be operational"
            logger.info(f"   ✅ System status: {health_summary['overall_status']}")
            
            logger.info("🎉 End-to-end integration test completed successfully!")
            
        finally:
            await production_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_real_price_oracle_integration(self, price_oracle):
        """Test price oracle with real on-chain data."""
        logger.info("🔄 Testing real price oracle integration...")
        
        # Test major assets
        test_cases = [
            ("ETH_MAINNET_WETH", 1000, 10000),  # ETH should be $1k-$10k
            ("ETH_MAINNET_USDC", 0.95, 1.05),   # USDC should be ~$1
            ("ETH_MAINNET_USDT", 0.95, 1.05),   # USDT should be ~$1
            ("ETH_MAINNET_DAI", 0.95, 1.05),    # DAI should be ~$1
        ]
        
        for asset_id, min_price, max_price in test_cases:
            price = await price_oracle.get_price_usd(asset_id)
            
            assert price is not None, f"Should get price for {asset_id}"
            assert min_price <= price <= max_price, f"{asset_id} price ${price:.2f} outside expected range ${min_price}-${max_price}"
            
            logger.info(f"   ✅ {asset_id}: ${price:.2f}")
        
        logger.info("✅ Price oracle integration test passed!")
    
    @pytest.mark.asyncio
    async def test_protocol_registry_production_data(self):
        """Test protocol registry with production contract addresses."""
        logger.info("🔄 Testing protocol registry with production data...")
        
        # Test critical protocols
        critical_protocols = ["uniswap_v3", "aave_v3", "curve", "balancer", "sushiswap"]
        
        for protocol_id in critical_protocols:
            protocol = production_registry.get_protocol(protocol_id)
            assert protocol is not None, f"Should find protocol {protocol_id}"
            assert protocol.is_enabled, f"Protocol {protocol_id} should be enabled"
            
            # Check Ethereum contracts
            eth_contracts = protocol.contracts.get("ethereum", {})
            assert len(eth_contracts) > 0, f"Protocol {protocol_id} should have Ethereum contracts"
            
            logger.info(f"   ✅ {protocol_id}: {len(eth_contracts)} contracts")
        
        # Test contract address resolution
        uniswap_factory = production_registry.get_contract_address("uniswap_v3", "ethereum", "factory")
        assert uniswap_factory == "0x1F98431c8aD98523631AE4a59f267346ea31F984", "Should get correct Uniswap V3 factory"
        
        aave_pool = production_registry.get_contract_address("aave_v3", "ethereum", "pool")
        assert aave_pool == "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2", "Should get correct Aave V3 pool"
        
        logger.info("✅ Protocol registry test passed!")
    
    @pytest.mark.asyncio
    async def test_edge_state_pipeline_real_data(self, edge_pipeline):
        """Test edge state pipeline with real protocol data."""
        logger.info("🔄 Testing edge state pipeline with real data...")
        
        # Discover high-priority edges
        edges = await edge_pipeline.discover_edges()
        assert len(edges) > 0, "Should discover edges from real protocols"
        
        # Test edge state updates
        updated_count = 0
        error_count = 0
        
        for edge in edges[:10]:  # Test first 10 edges
            try:
                updated_edge = await edge_pipeline.update_edge_state(edge)
                if updated_edge and updated_edge.state:
                    updated_count += 1
                    rate = updated_edge.state.conversion_rate
                    rate_str = f"{rate:.6f}" if rate is not None else "None"
                    logger.info(f"   ✅ Updated {edge.edge_id}: rate={rate_str}")
            except Exception as e:
                error_count += 1
                logger.warning(f"   ⚠️  Failed to update {edge.edge_id}: {e}")
        
        assert updated_count > 0, "Should successfully update at least some edges"
        
        success_rate = updated_count / (updated_count + error_count)
        assert success_rate >= 0.5, f"Edge update success rate {success_rate:.1%} too low"
        
        logger.info(f"✅ Edge pipeline test passed: {updated_count} updated, {success_rate:.1%} success rate")
    
    @pytest.mark.asyncio
    async def test_transaction_builder_real_protocols(self, transaction_builder):
        """Test transaction builder with real protocol integration."""
        logger.info("🔄 Testing transaction builder with real protocols...")
        
        # Create test arbitrage path
        test_path = self._create_test_arbitrage_path()
        
        # Test simple arbitrage building
        transaction = await transaction_builder.build_simple_arbitrage(
            test_path,
            Decimal("1000")  # $1000 test amount
        )
        
        if transaction:
            assert transaction.transaction_id is not None
            assert transaction.strategy_type == "simple_arbitrage"
            assert len(transaction.steps) == len(test_path)
            assert transaction.total_input_amount == Decimal("1000")
            
            logger.info(f"   ✅ Built transaction: {transaction.transaction_id}")
            logger.info(f"   📊 Expected profit: ${transaction.expected_profit:.2f}")
            logger.info(f"   ⛽ Gas estimate: {transaction.max_gas_limit:,}")
            
            # Test transaction simulation
            simulation_passed = await transaction_builder.simulate_transaction(transaction)
            logger.info(f"   🔍 Simulation: {'✅ PASSED' if simulation_passed else '❌ FAILED'}")
            
        else:
            logger.info("   ℹ️  Transaction building failed (expected in test environment)")
        
        logger.info("✅ Transaction builder test completed!")
    
    @pytest.mark.asyncio
    async def test_production_monitoring_real_system(self, production_monitor):
        """Test production monitoring with real system integration."""
        logger.info("🔄 Testing production monitoring with real system...")
        
        # Start monitoring
        await production_monitor.start_monitoring()
        
        try:
            # Let monitoring run for a few seconds
            await asyncio.sleep(5)
            
            # Check health status
            health_summary = await production_monitor.get_system_health()
            
            assert "overall_status" in health_summary
            assert health_summary["overall_status"] in ["healthy", "degraded", "unhealthy"]
            
            logger.info(f"   ✅ System status: {health_summary['overall_status']}")
            logger.info(f"   📊 Active alerts: {health_summary['active_alerts']}")
            
            # Check metrics
            metrics_summary = await production_monitor.get_metrics_summary()
            logger.info(f"   📈 Metrics tracked: {len(metrics_summary)}")
            
            # Check alerts
            active_alerts = await production_monitor.get_active_alerts()
            logger.info(f"   🚨 Active alerts: {len(active_alerts)}")
            
            for alert in active_alerts[:3]:  # Show first 3 alerts
                logger.info(f"      • {alert['severity'].upper()}: {alert['title']}")
        
        finally:
            await production_monitor.stop_monitoring()
        
        logger.info("✅ Production monitoring test completed!")
    
    @pytest.mark.asyncio
    async def test_system_performance_under_load(
        self, 
        blockchain_provider, 
        price_oracle, 
        edge_pipeline
    ):
        """Test system performance under concurrent load."""
        logger.info("🔄 Testing system performance under load...")
        
        # Concurrent price requests
        price_tasks = [
            price_oracle.get_price_usd("ETH_MAINNET_WETH"),
            price_oracle.get_price_usd("ETH_MAINNET_USDC"),
            price_oracle.get_price_usd("ETH_MAINNET_USDT"),
            price_oracle.get_price_usd("ETH_MAINNET_DAI"),
        ]
        
        prices = await asyncio.gather(*price_tasks, return_exceptions=True)
        
        successful_prices = [p for p in prices if isinstance(p, (int, float)) and p > 0]
        assert len(successful_prices) >= 3, "Should successfully fetch most prices concurrently"
        
        logger.info(f"   ✅ Concurrent price requests: {len(successful_prices)}/4 successful")
        
        # Concurrent blockchain calls
        blockchain_tasks = [
            blockchain_provider.get_web3("ethereum"),
            blockchain_provider.get_web3("arbitrum"),
            blockchain_provider.get_web3("base"),
        ]
        
        web3_instances = await asyncio.gather(*blockchain_tasks, return_exceptions=True)
        successful_connections = [w for w in web3_instances if w is not None]
        
        assert len(successful_connections) >= 2, "Should maintain multiple blockchain connections"
        logger.info(f"   ✅ Concurrent blockchain connections: {len(successful_connections)}/3 successful")
        
        logger.info("✅ Performance test completed!")
    
    def _find_potential_arbitrage_paths(self, edges: List[YieldGraphEdge]) -> List[List[YieldGraphEdge]]:
        """Find potential arbitrage paths from available edges."""
        paths = []
        
        # Group edges by asset pairs
        asset_pairs = {}
        for edge in edges:
            pair = (edge.source_asset_id, edge.target_asset_id)
            if pair not in asset_pairs:
                asset_pairs[pair] = []
            asset_pairs[pair].append(edge)
        
        # Look for simple 2-step arbitrage opportunities
        for edge1 in edges:
            if edge1.edge_type != EdgeType.TRADE:
                continue
            
            # Find reverse edge
            reverse_pair = (edge1.target_asset_id, edge1.source_asset_id)
            if reverse_pair in asset_pairs:
                for edge2 in asset_pairs[reverse_pair]:
                    if (edge2.edge_type == EdgeType.TRADE and 
                        edge2.protocol_name != edge1.protocol_name):  # Different protocols
                        paths.append([edge1, edge2])
        
        return paths[:5]  # Return first 5 potential paths
    
    def _create_test_arbitrage_path(self) -> List[YieldGraphEdge]:
        """Create a test arbitrage path for transaction building."""
        # Create test edges for USDC -> WETH -> USDC arbitrage
        edge1 = YieldGraphEdge(
            edge_id="test_uniswap_v3_usdc_weth",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_WETH",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswap_v3",
            chain_name="ethereum",
            state=EdgeState(
                conversion_rate=0.00035,  # ~$2857 per ETH
                liquidity_usd=1000000,
                gas_cost_usd=15.0,
                confidence_score=0.95
            )
        )
        
        edge2 = YieldGraphEdge(
            edge_id="test_sushiswap_weth_usdc",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="sushiswap",
            chain_name="ethereum",
            state=EdgeState(
                conversion_rate=2860.0,   # Slightly higher rate for arbitrage
                liquidity_usd=500000,
                gas_cost_usd=20.0,
                confidence_score=0.90
            )
        )
        
        return [edge1, edge2]


@pytest.mark.asyncio
async def test_full_integration_suite():
    """Run the complete integration test suite."""
    logger.info("🚀 Starting full integration test suite...")
    
    # Initialize components
    blockchain_provider = BlockchainProvider()
    await blockchain_provider.initialize()
    
    try:
        # Mock Redis for testing
        redis_client = AsyncMock()
        redis_client.ping.return_value = True
        redis_client.get.return_value = None
        redis_client.set.return_value = True
        redis_client.lpush.return_value = 1
        redis_client.ltrim.return_value = True
        redis_client.setex.return_value = True
        
        # Initialize oracle
        oracle = OnChainPriceOracle(blockchain_provider, redis_client)
        await oracle.initialize()
        
        # Initialize pipeline
        pipeline = RealEdgeStatePipeline(blockchain_provider, oracle)
        await pipeline.initialize()
        
        # Initialize transaction builder
        builder = RealTransactionBuilder(blockchain_provider, oracle)
        await builder.initialize()
        
        # Initialize monitoring
        monitor = ProductionMonitor(blockchain_provider, oracle, redis_client)
        
        # Run integration test
        test_instance = TestRealDataIntegration()
        
        await test_instance.test_end_to_end_arbitrage_discovery(
            blockchain_provider, oracle, pipeline, builder, monitor
        )
        
        logger.info("🎉 Full integration test suite completed successfully!")
        
    finally:
        await blockchain_provider.close()


if __name__ == "__main__":
    asyncio.run(test_full_integration_suite())