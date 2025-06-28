#!/usr/bin/env python3
"""
Script to run integration tests with real DeFi data.

This script runs comprehensive integration tests that validate the complete
arbitrage system using real blockchain connections and live protocol data.
"""
import asyncio
import sys
import logging
from typing import Dict, Any, List
import time

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.blockchain_connector.provider import BlockchainProvider
from yield_arbitrage.execution.onchain_price_oracle import OnChainPriceOracle
from yield_arbitrage.protocols.production_registry import production_registry
from yield_arbitrage.data.real_edge_pipeline import RealEdgeStatePipeline
from yield_arbitrage.execution.real_transaction_builder import RealTransactionBuilder
from yield_arbitrage.monitoring.production_monitor import ProductionMonitor
from yield_arbitrage.config.production import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Runner for comprehensive integration tests with real data."""
    
    def __init__(self):
        self.results = {}
        self.blockchain_provider = None
        self.oracle = None
        self.edge_pipeline = None
        self.transaction_builder = None
        self.production_monitor = None
        self.redis_client = None
    
    async def initialize_components(self) -> bool:
        """Initialize all system components."""
        try:
            logger.info("🚀 Initializing system components...")
            
            # Initialize blockchain provider
            self.blockchain_provider = BlockchainProvider()
            await self.blockchain_provider.initialize()
            logger.info("   ✅ Blockchain provider initialized")
            
            # Mock Redis client for testing
            from unittest.mock import AsyncMock
            self.redis_client = AsyncMock()
            self.redis_client.ping.return_value = True
            self.redis_client.get.return_value = None
            self.redis_client.set.return_value = True
            self.redis_client.lpush.return_value = 1
            self.redis_client.ltrim.return_value = True
            self.redis_client.setex.return_value = True
            logger.info("   ✅ Redis client (mock) initialized")
            
            # Initialize price oracle
            self.oracle = OnChainPriceOracle(self.blockchain_provider, self.redis_client)
            await self.oracle.initialize()
            logger.info("   ✅ On-chain price oracle initialized")
            
            # Initialize edge pipeline
            self.edge_pipeline = RealEdgeStatePipeline(
                self.blockchain_provider, 
                self.oracle, 
                self.redis_client
            )
            await self.edge_pipeline.initialize()
            logger.info("   ✅ Real edge state pipeline initialized")
            
            # Initialize transaction builder
            self.transaction_builder = RealTransactionBuilder(self.blockchain_provider, self.oracle)
            await self.transaction_builder.initialize()
            logger.info("   ✅ Real transaction builder initialized")
            
            # Initialize production monitor
            self.production_monitor = ProductionMonitor(
                self.blockchain_provider, 
                self.oracle, 
                self.redis_client
            )
            logger.info("   ✅ Production monitor initialized")
            
            logger.info("🎉 All components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    async def test_blockchain_connectivity(self) -> Dict[str, Any]:
        """Test blockchain connectivity across all supported chains."""
        logger.info("🔗 Testing blockchain connectivity...")
        
        test_results = {
            "test_name": "blockchain_connectivity",
            "chains_tested": 0,
            "chains_connected": 0,
            "chain_details": {},
            "success": False
        }
        
        try:
            chains = ["ethereum", "arbitrum", "base", "sonic", "berachain"]
            
            for chain in chains:
                test_results["chains_tested"] += 1
                
                try:
                    web3 = await self.blockchain_provider.get_web3(chain)
                    if web3:
                        block_number = await web3.eth.block_number
                        chain_id = await web3.eth.chain_id
                        gas_price = await web3.eth.gas_price
                        
                        test_results["chains_connected"] += 1
                        test_results["chain_details"][chain] = {
                            "connected": True,
                            "block_number": block_number,
                            "chain_id": chain_id,
                            "gas_price_gwei": gas_price / 1e9
                        }
                        
                        logger.info(f"   ✅ {chain}: Block {block_number:,}, Gas {gas_price/1e9:.1f} gwei")
                    else:
                        test_results["chain_details"][chain] = {"connected": False, "error": "No web3 instance"}
                        logger.error(f"   ❌ {chain}: Connection failed")
                        
                except Exception as e:
                    test_results["chain_details"][chain] = {"connected": False, "error": str(e)}
                    logger.error(f"   ❌ {chain}: {e}")
            
            connectivity_ratio = test_results["chains_connected"] / test_results["chains_tested"]
            test_results["connectivity_ratio"] = connectivity_ratio
            test_results["success"] = connectivity_ratio >= 0.8  # 80% threshold
            
            logger.info(f"   📊 Connectivity: {test_results['chains_connected']}/{test_results['chains_tested']} ({connectivity_ratio:.1%})")
            
        except Exception as e:
            logger.error(f"   ❌ Blockchain connectivity test failed: {e}")
            test_results["error"] = str(e)
        
        return test_results
    
    async def test_price_oracle_accuracy(self) -> Dict[str, Any]:
        """Test price oracle accuracy with known asset prices."""
        logger.info("💰 Testing price oracle accuracy...")
        
        test_results = {
            "test_name": "price_oracle_accuracy",
            "assets_tested": 0,
            "prices_fetched": 0,
            "price_details": {},
            "success": False
        }
        
        try:
            # Test major assets with expected price ranges
            test_assets = [
                ("ETH_MAINNET_WETH", 1000, 10000),   # ETH: $1k-$10k
                ("ETH_MAINNET_USDC", 0.95, 1.05),    # USDC: ~$1
                ("ETH_MAINNET_USDT", 0.95, 1.05),    # USDT: ~$1
                ("ETH_MAINNET_DAI", 0.95, 1.05),     # DAI: ~$1
            ]
            
            for asset_id, min_price, max_price in test_assets:
                test_results["assets_tested"] += 1
                
                try:
                    price = await self.oracle.get_price_usd(asset_id)
                    
                    if price is not None:
                        test_results["prices_fetched"] += 1
                        
                        price_valid = min_price <= price <= max_price
                        test_results["price_details"][asset_id] = {
                            "price": price,
                            "expected_range": f"${min_price}-${max_price}",
                            "valid": price_valid
                        }
                        
                        status = "✅" if price_valid else "⚠️"
                        logger.info(f"   {status} {asset_id}: ${price:.2f}")
                        
                        if not price_valid:
                            logger.warning(f"      Price outside expected range ${min_price}-${max_price}")
                    else:
                        test_results["price_details"][asset_id] = {
                            "price": None,
                            "error": "No price returned",
                            "valid": False
                        }
                        logger.error(f"   ❌ {asset_id}: No price returned")
                        
                except Exception as e:
                    test_results["price_details"][asset_id] = {
                        "price": None,
                        "error": str(e),
                        "valid": False
                    }
                    logger.error(f"   ❌ {asset_id}: {e}")
            
            success_rate = test_results["prices_fetched"] / test_results["assets_tested"]
            valid_prices = sum(1 for details in test_results["price_details"].values() 
                             if details.get("valid", False))
            accuracy_rate = valid_prices / test_results["assets_tested"]
            
            test_results["success_rate"] = success_rate
            test_results["accuracy_rate"] = accuracy_rate
            test_results["success"] = success_rate >= 0.75 and accuracy_rate >= 0.75
            
            logger.info(f"   📊 Price fetching: {success_rate:.1%}, Accuracy: {accuracy_rate:.1%}")
            
        except Exception as e:
            logger.error(f"   ❌ Price oracle test failed: {e}")
            test_results["error"] = str(e)
        
        return test_results
    
    async def test_protocol_registry_integration(self) -> Dict[str, Any]:
        """Test protocol registry integration with real contracts."""
        logger.info("🏗️ Testing protocol registry integration...")
        
        test_results = {
            "test_name": "protocol_registry",
            "protocols_tested": 0,
            "protocols_valid": 0,
            "protocol_details": {},
            "success": False
        }
        
        try:
            critical_protocols = ["uniswap_v3", "aave_v3", "curve", "balancer", "sushiswap"]
            
            for protocol_id in critical_protocols:
                test_results["protocols_tested"] += 1
                
                try:
                    protocol = production_registry.get_protocol(protocol_id)
                    
                    if protocol and protocol.is_enabled:
                        eth_contracts = protocol.contracts.get("ethereum", {})
                        contract_count = len(eth_contracts)
                        
                        test_results["protocols_valid"] += 1
                        test_results["protocol_details"][protocol_id] = {
                            "enabled": True,
                            "contract_count": contract_count,
                            "contracts": list(eth_contracts.keys())
                        }
                        
                        logger.info(f"   ✅ {protocol_id}: {contract_count} contracts")
                    else:
                        test_results["protocol_details"][protocol_id] = {
                            "enabled": False,
                            "error": "Protocol not found or disabled"
                        }
                        logger.error(f"   ❌ {protocol_id}: Not found or disabled")
                        
                except Exception as e:
                    test_results["protocol_details"][protocol_id] = {
                        "enabled": False,
                        "error": str(e)
                    }
                    logger.error(f"   ❌ {protocol_id}: {e}")
            
            # Test specific contract address resolution
            test_contracts = [
                ("uniswap_v3", "ethereum", "factory", "0x1F98431c8aD98523631AE4a59f267346ea31F984"),
                ("aave_v3", "ethereum", "pool", "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"),
            ]
            
            address_tests_passed = 0
            for protocol, chain, contract_type, expected_address in test_contracts:
                try:
                    actual_address = production_registry.get_contract_address(protocol, chain, contract_type)
                    if actual_address == expected_address:
                        address_tests_passed += 1
                        logger.info(f"   ✅ {protocol} {contract_type}: {actual_address}")
                    else:
                        logger.error(f"   ❌ {protocol} {contract_type}: Expected {expected_address}, got {actual_address}")
                except Exception as e:
                    logger.error(f"   ❌ {protocol} {contract_type}: {e}")
            
            protocol_success_rate = test_results["protocols_valid"] / test_results["protocols_tested"]
            address_success_rate = address_tests_passed / len(test_contracts)
            
            test_results["protocol_success_rate"] = protocol_success_rate
            test_results["address_success_rate"] = address_success_rate
            test_results["success"] = protocol_success_rate >= 0.8 and address_success_rate >= 0.8
            
            logger.info(f"   📊 Protocol registry: {protocol_success_rate:.1%} protocols, {address_success_rate:.1%} addresses")
            
        except Exception as e:
            logger.error(f"   ❌ Protocol registry test failed: {e}")
            test_results["error"] = str(e)
        
        return test_results
    
    async def test_edge_discovery_and_updates(self) -> Dict[str, Any]:
        """Test edge discovery and state updates."""
        logger.info("🔀 Testing edge discovery and state updates...")
        
        test_results = {
            "test_name": "edge_discovery",
            "edges_discovered": 0,
            "edges_updated": 0,
            "update_success_rate": 0.0,
            "success": False
        }
        
        try:
            # Discover edges
            edges = await self.edge_pipeline.discover_edges()
            test_results["edges_discovered"] = len(edges)
            
            logger.info(f"   🔍 Discovered {len(edges)} edges")
            
            if len(edges) > 0:
                # Test edge state updates on a subset
                test_edges = edges[:10]  # Test first 10 edges
                updated_count = 0
                
                for edge in test_edges:
                    try:
                        updated_edge = await self.edge_pipeline.update_edge_state(edge)
                        if updated_edge and updated_edge.state.conversion_rate:
                            updated_count += 1
                            logger.info(f"   ✅ Updated {edge.edge_id}: rate={updated_edge.state.conversion_rate:.6f}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Failed to update {edge.edge_id}: {e}")
                
                test_results["edges_updated"] = updated_count
                test_results["update_success_rate"] = updated_count / len(test_edges)
                test_results["success"] = (
                    test_results["edges_discovered"] > 0 and 
                    test_results["update_success_rate"] >= 0.5
                )
                
                logger.info(f"   📊 Edge updates: {updated_count}/{len(test_edges)} ({test_results['update_success_rate']:.1%})")
            else:
                logger.warning("   ⚠️ No edges discovered")
                test_results["success"] = False
            
        except Exception as e:
            logger.error(f"   ❌ Edge discovery test failed: {e}")
            test_results["error"] = str(e)
        
        return test_results
    
    async def test_transaction_building(self) -> Dict[str, Any]:
        """Test transaction building capabilities."""
        logger.info("🏗️ Testing transaction building...")
        
        test_results = {
            "test_name": "transaction_building",
            "transactions_attempted": 0,
            "transactions_built": 0,
            "simulations_attempted": 0,
            "simulations_passed": 0,
            "success": False
        }
        
        try:
            # Create test arbitrage path
            from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeState, EdgeType
            from decimal import Decimal
            
            test_path = [
                YieldGraphEdge(
                    edge_id="test_uniswap_v3_usdc_weth",
                    source_asset_id="ETH_MAINNET_USDC",
                    target_asset_id="ETH_MAINNET_WETH",
                    edge_type=EdgeType.TRADE,
                    protocol_name="uniswap_v3",
                    chain_name="ethereum",
                    state=EdgeState(
                        conversion_rate=0.00035,
                        liquidity_usd=1000000,
                        gas_cost_usd=15.0,
                        confidence_score=0.95
                    )
                ),
                YieldGraphEdge(
                    edge_id="test_uniswap_v3_weth_usdc",
                    source_asset_id="ETH_MAINNET_WETH",
                    target_asset_id="ETH_MAINNET_USDC",
                    edge_type=EdgeType.TRADE,
                    protocol_name="uniswap_v3",
                    chain_name="ethereum",
                    state=EdgeState(
                        conversion_rate=2860.0,
                        liquidity_usd=500000,
                        gas_cost_usd=20.0,
                        confidence_score=0.90
                    )
                )
            ]
            
            # Test transaction building
            test_results["transactions_attempted"] = 1
            
            # Provide a test recipient address since we don't have a private key
            test_recipient = "0x742d35Cc6634C0532925a3b8D39C39c0fa6d5C4d"  # Test address
            
            transaction = await self.transaction_builder.build_simple_arbitrage(
                test_path,
                Decimal("1000"),  # $1000 test amount
                recipient_address=test_recipient
            )
            
            if transaction:
                test_results["transactions_built"] = 1
                logger.info(f"   ✅ Built transaction: {transaction.transaction_id}")
                logger.info(f"   📊 Expected profit: ${transaction.expected_profit:.2f}")
                logger.info(f"   ⛽ Gas estimate: {transaction.max_gas_limit:,}")
                
                # Test simulation
                test_results["simulations_attempted"] = 1
                simulation_passed = await self.transaction_builder.simulate_transaction(transaction)
                
                if simulation_passed:
                    test_results["simulations_passed"] = 1
                    logger.info("   ✅ Transaction simulation passed")
                else:
                    logger.warning("   ⚠️ Transaction simulation failed")
            else:
                logger.warning("   ⚠️ Transaction building failed")
            
            build_success_rate = test_results["transactions_built"] / test_results["transactions_attempted"]
            sim_success_rate = test_results["simulations_passed"] / max(test_results["simulations_attempted"], 1)
            
            test_results["build_success_rate"] = build_success_rate
            test_results["simulation_success_rate"] = sim_success_rate
            test_results["success"] = build_success_rate >= 0.8
            
            logger.info(f"   📊 Transaction building: {build_success_rate:.1%}, Simulation: {sim_success_rate:.1%}")
            
        except Exception as e:
            logger.error(f"   ❌ Transaction building test failed: {e}")
            test_results["error"] = str(e)
        
        return test_results
    
    async def test_monitoring_system(self) -> Dict[str, Any]:
        """Test production monitoring system."""
        logger.info("🏥 Testing production monitoring system...")
        
        test_results = {
            "test_name": "monitoring_system",
            "monitoring_started": False,
            "health_checks_run": 0,
            "system_status": None,
            "success": False
        }
        
        try:
            # Start monitoring
            await self.production_monitor.start_monitoring()
            test_results["monitoring_started"] = True
            logger.info("   ✅ Monitoring started")
            
            # Let monitoring run briefly
            await asyncio.sleep(5)
            
            # Check system health
            health_summary = await self.production_monitor.get_system_health()
            test_results["system_status"] = health_summary["overall_status"]
            test_results["active_alerts"] = health_summary["active_alerts"]
            
            logger.info(f"   📊 System status: {health_summary['overall_status']}")
            logger.info(f"   🚨 Active alerts: {health_summary['active_alerts']}")
            
            # Check metrics
            metrics_summary = await self.production_monitor.get_metrics_summary()
            test_results["metrics_tracked"] = len(metrics_summary)
            
            logger.info(f"   📈 Metrics tracked: {len(metrics_summary)}")
            
            # Stop monitoring
            await self.production_monitor.stop_monitoring()
            logger.info("   ✅ Monitoring stopped")
            
            test_results["success"] = (
                test_results["monitoring_started"] and
                test_results["system_status"] in ["healthy", "degraded"]
            )
            
        except Exception as e:
            logger.error(f"   ❌ Monitoring system test failed: {e}")
            test_results["error"] = str(e)
            
            # Ensure monitoring is stopped
            try:
                await self.production_monitor.stop_monitoring()
            except:
                pass
        
        return test_results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("🚀 Running comprehensive integration test suite...")
        start_time = time.time()
        
        test_results = {
            "overall_success": False,
            "tests_run": 0,
            "tests_passed": 0,
            "execution_time_seconds": 0,
            "test_details": {}
        }
        
        # Initialize components
        if not await self.initialize_components():
            test_results["error"] = "Component initialization failed"
            return test_results
        
        # Define test sequence
        tests = [
            ("blockchain_connectivity", self.test_blockchain_connectivity),
            ("price_oracle_accuracy", self.test_price_oracle_accuracy),
            ("protocol_registry", self.test_protocol_registry_integration),
            ("edge_discovery", self.test_edge_discovery_and_updates),
            ("transaction_building", self.test_transaction_building),
            ("monitoring_system", self.test_monitoring_system),
        ]
        
        # Run tests
        for test_name, test_func in tests:
            test_results["tests_run"] += 1
            
            try:
                logger.info(f"\n{'='*60}")
                result = await test_func()
                test_results["test_details"][test_name] = result
                
                if result.get("success", False):
                    test_results["tests_passed"] += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: FAILED with exception: {e}")
                test_results["test_details"][test_name] = {
                    "test_name": test_name,
                    "success": False,
                    "error": str(e)
                }
        
        # Calculate results
        test_results["execution_time_seconds"] = time.time() - start_time
        test_results["success_rate"] = test_results["tests_passed"] / test_results["tests_run"]
        test_results["overall_success"] = test_results["success_rate"] >= 0.8  # 80% threshold
        
        return test_results
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.production_monitor:
                await self.production_monitor.shutdown()
            
            if self.blockchain_provider:
                await self.blockchain_provider.close()
            
            logger.info("🧹 Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")


async def main():
    """Main test runner."""
    print("🚀 DeFi Arbitrage System - Integration Test Suite")
    print("=" * 60)
    
    runner = IntegrationTestRunner()
    
    try:
        # Run all tests
        results = await runner.run_all_tests()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 Integration Test Results Summary")
        print("=" * 60)
        
        print(f"Tests run: {results['tests_run']}")
        print(f"Tests passed: {results['tests_passed']}")
        
        if results['tests_run'] > 0:
            print(f"Success rate: {results['success_rate']:.1%}")
        else:
            print("Success rate: N/A (no tests completed)")
            
        print(f"Execution time: {results['execution_time_seconds']:.1f} seconds")
        
        print("\n📊 Individual Test Results:")
        for test_name, test_result in results["test_details"].items():
            status = "✅ PASSED" if test_result.get("success", False) else "❌ FAILED"
            print(f"   {status}: {test_name.replace('_', ' ').title()}")
            
            if "error" in test_result:
                print(f"      Error: {test_result['error']}")
        
        if results["overall_success"]:
            print("\n🎉 Integration test suite PASSED!")
            print("✅ System is ready for production deployment")
        else:
            print("\n⚠️ Integration test suite FAILED!")
            print("❌ System needs fixes before production deployment")
        
        return results["overall_success"]
        
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)