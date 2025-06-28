#!/usr/bin/env python3
"""Test script for real transaction builder with production DeFi protocols."""
import asyncio
import sys
import logging
from decimal import Decimal
from typing import List

# Add src to path
sys.path.append('/home/david/projects/yieldarbitrage/src')

from yield_arbitrage.execution.real_transaction_builder import (
    RealTransactionBuilder, TransactionStatus, ArbitrageTransaction
)
from yield_arbitrage.blockchain_connector.provider import BlockchainProvider
from yield_arbitrage.execution.onchain_price_oracle import OnChainPriceOracle
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType, EdgeState
from yield_arbitrage.config.production import get_config
from yield_arbitrage.protocols.production_registry import production_registry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_arbitrage_path() -> List[YieldGraphEdge]:
    """Create a mock arbitrage path for testing."""
    
    # Create a simple USDC -> WETH -> USDC arbitrage path
    edge1 = YieldGraphEdge(
        edge_id="uniswap_v3_0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640_0xa0b86a33e6441b5311ed1be2b26b7bac4f0d5f0b_0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
        source_asset_id="ETH_MAINNET_USDC",
        target_asset_id="ETH_MAINNET_WETH",
        edge_type=EdgeType.TRADE,
        protocol_name="uniswap_v3",
        chain_name="ethereum"
    )
    
    # Set realistic edge state
    edge1.state = EdgeState(
        conversion_rate=0.0003,  # 1 USDC = 0.0003 WETH (WETH ~$3333)
        liquidity_usd=50_000_000.0,  # $50M liquidity
        gas_cost_usd=15.0,
        confidence_score=0.95
    )
    
    edge2 = YieldGraphEdge(
        edge_id="uniswap_v3_0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8_0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2_0xa0b86a33e6441b5311ed1be2b26b7bac4f0d5f0b",
        source_asset_id="ETH_MAINNET_WETH",
        target_asset_id="ETH_MAINNET_USDC",
        edge_type=EdgeType.TRADE,
        protocol_name="uniswap_v3",
        chain_name="ethereum"
    )
    
    # Set slightly better rate for arbitrage opportunity
    edge2.state = EdgeState(
        conversion_rate=3340.0,  # 1 WETH = 3340 USDC (better rate)
        liquidity_usd=30_000_000.0,  # $30M liquidity
        gas_cost_usd=15.0,
        confidence_score=0.95
    )
    
    return [edge1, edge2]


async def initialize_dependencies():
    """Initialize all required dependencies."""
    print("🚀 Initializing transaction builder dependencies...")
    
    # Load configuration
    config = get_config()
    
    # Initialize blockchain provider
    blockchain_provider = BlockchainProvider()
    await blockchain_provider.initialize()
    
    # Initialize on-chain price oracle
    try:
        from unittest.mock import AsyncMock
        redis_client = AsyncMock()  # Mock Redis for testing
        oracle = OnChainPriceOracle(blockchain_provider, redis_client)
        print("   ✅ Oracle initialized")
    except Exception as e:
        print(f"   ⚠️  Oracle initialization failed: {e}")
        oracle = None
    
    print("   ✅ All dependencies initialized")
    return blockchain_provider, oracle


async def test_builder_initialization():
    """Test transaction builder initialization."""
    print("\n📋 Testing Transaction Builder Initialization\n")
    
    blockchain_provider, oracle = await initialize_dependencies()
    
    try:
        # Initialize transaction builder (without private key for safety)
        builder = RealTransactionBuilder(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            private_key=None  # No real private key for testing
        )
        
        print("   🔄 Initializing transaction builder...")
        success = await builder.initialize()
        
        if success:
            print("   ✅ Transaction builder initialized successfully")
            
            # Check contract interfaces
            interfaces = builder.contract_interfaces
            print(f"   📊 Loaded {len(interfaces)} contract interfaces:")
            for contract_name in interfaces:
                functions = list(interfaces[contract_name].keys())
                print(f"      • {contract_name}: {len(functions)} functions")
            
            # Test registry integration
            router_address = production_registry.get_contract_address("uniswap_v3", "ethereum", "swap_router")
            if router_address:
                print(f"   ✅ Uniswap V3 router address: {router_address}")
            else:
                print("   ❌ Failed to get Uniswap V3 router address")
            
            return builder
        else:
            print("   ❌ Transaction builder initialization failed")
            return None
            
    except Exception as e:
        print(f"   ❌ Initialization test failed: {e}")
        return None
    finally:
        await blockchain_provider.close()


async def test_simple_arbitrage_building():
    """Test building simple arbitrage transactions."""
    print("\n🔄 Testing Simple Arbitrage Building\n")
    
    blockchain_provider, oracle = await initialize_dependencies()
    
    try:
        builder = RealTransactionBuilder(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            private_key="0x" + "1" * 64  # Dummy private key for testing
        )
        
        await builder.initialize()
        
        # Create mock arbitrage path
        arbitrage_path = create_mock_arbitrage_path()
        input_amount = Decimal("1000")  # $1000 USDC
        
        print(f"   🔄 Building arbitrage transaction for ${input_amount}")
        print(f"   📊 Path: {len(arbitrage_path)} edges")
        for i, edge in enumerate(arbitrage_path):
            print(f"      Step {i+1}: {edge.source_asset_id} -> {edge.target_asset_id}")
            print(f"                Rate: {edge.state.conversion_rate}")
        
        # Build transaction
        transaction = await builder.build_simple_arbitrage(
            arbitrage_path=arbitrage_path,
            input_amount=input_amount,
            recipient_address="0x742d35Cc6634C0532925a3b8D82BA46fC8a61e04"  # Test address
        )
        
        if transaction:
            print(f"   ✅ Transaction built successfully: {transaction.transaction_id}")
            print(f"      Strategy: {transaction.strategy_type}")
            print(f"      Steps: {len(transaction.steps)}")
            print(f"      Expected profit: ${transaction.expected_profit:.2f}")
            print(f"      Max gas limit: {transaction.max_gas_limit:,}")
            print(f"      Status: {transaction.status.value}")
            
            # Validate transaction structure
            if transaction.built_tx:
                print("   ✅ Transaction calldata built")
                print(f"      To: {transaction.built_tx.get('to', 'N/A')}")
                print(f"      Gas: {transaction.built_tx.get('gas', 0):,}")
                print(f"      Gas Price: {transaction.built_tx.get('gasPrice', 0):,} wei")
            else:
                print("   ❌ No transaction calldata")
            
            # Test transaction steps
            print("\n   📊 Transaction Steps:")
            for i, step in enumerate(transaction.steps):
                print(f"      Step {i+1}: {step.step_id}")
                print(f"         Contract: {step.contract_address}")
                print(f"         Function: {step.function_name}")
                print(f"         Input: {step.input_amount}")
                print(f"         Expected Output: {step.expected_output}")
                print(f"         Gas Estimate: {step.gas_estimate:,}")
            
            return transaction
        else:
            print("   ❌ Failed to build arbitrage transaction")
            return None
            
    except Exception as e:
        print(f"   ❌ Simple arbitrage building test failed: {e}")
        return None
    finally:
        await blockchain_provider.close()


async def test_flash_loan_arbitrage_building():
    """Test building flash loan arbitrage transactions."""
    print("\n⚡ Testing Flash Loan Arbitrage Building\n")
    
    blockchain_provider, oracle = await initialize_dependencies()
    
    try:
        builder = RealTransactionBuilder(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            private_key="0x" + "1" * 64  # Dummy private key for testing
        )
        
        await builder.initialize()
        
        # Create arbitrage path (excluding flash loan)
        arbitrage_path = create_mock_arbitrage_path()
        flash_loan_amount = Decimal("10000")  # $10k flash loan
        flash_loan_asset = "USDC"
        
        print(f"   🔄 Building flash loan arbitrage for ${flash_loan_amount} {flash_loan_asset}")
        print(f"   📊 Arbitrage path: {len(arbitrage_path)} edges")
        
        # Build flash loan transaction
        transaction = await builder.build_flash_loan_arbitrage(
            arbitrage_path=arbitrage_path,
            flash_loan_amount=flash_loan_amount,
            flash_loan_asset=flash_loan_asset,
            recipient_address="0x742d35Cc6634C0532925a3b8D82BA46fC8a61e04"
        )
        
        if transaction:
            print(f"   ✅ Flash loan transaction built: {transaction.transaction_id}")
            print(f"      Strategy: {transaction.strategy_type}")
            print(f"      Flash loan amount: ${transaction.total_input_amount}")
            print(f"      Expected profit: ${transaction.expected_profit:.2f}")
            print(f"      Max gas limit: {transaction.max_gas_limit:,}")
            print(f"      Status: {transaction.status.value}")
            
            # Check Aave integration
            aave_pool = production_registry.get_contract_address("aave_v3", "ethereum", "pool")
            if aave_pool and transaction.built_tx:
                tx_to = transaction.built_tx.get('to')
                if tx_to == aave_pool:
                    print("   ✅ Transaction targets Aave V3 pool correctly")
                else:
                    print(f"   ⚠️  Transaction target mismatch: {tx_to} vs {aave_pool}")
            
            return transaction
        else:
            print("   ❌ Failed to build flash loan arbitrage transaction")
            return None
            
    except Exception as e:
        print(f"   ❌ Flash loan arbitrage building test failed: {e}")
        return None
    finally:
        await blockchain_provider.close()


async def test_transaction_simulation():
    """Test transaction simulation capabilities."""
    print("\n🎯 Testing Transaction Simulation\n")
    
    blockchain_provider, oracle = await initialize_dependencies()
    
    try:
        builder = RealTransactionBuilder(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            private_key="0x" + "1" * 64
        )
        
        await builder.initialize()
        
        # Build a test transaction
        arbitrage_path = create_mock_arbitrage_path()
        transaction = await builder.build_simple_arbitrage(
            arbitrage_path=arbitrage_path,
            input_amount=Decimal("1000")
        )
        
        if not transaction:
            print("   ❌ Failed to build transaction for simulation test")
            return False
        
        print(f"   🔄 Simulating transaction {transaction.transaction_id}...")
        
        # Simulate the transaction
        simulation_success = await builder.simulate_transaction(transaction)
        
        if simulation_success:
            print("   ✅ Transaction simulation passed")
            
            if transaction.simulation_result:
                result = transaction.simulation_result
                print(f"      Success: {result.get('success', False)}")
                print(f"      Gas used: {result.get('gas_used', 0):,.0f}")
                print(f"      Simulated at: {result.get('simulated_at', 0):.0f}")
                
                if result.get('error'):
                    print(f"      Error: {result['error']}")
            
            print(f"      Status: {transaction.status.value}")
            return True
        else:
            print("   ❌ Transaction simulation failed")
            if transaction.simulation_result and transaction.simulation_result.get('error'):
                print(f"      Error: {transaction.simulation_result['error']}")
            return False
            
    except Exception as e:
        print(f"   ❌ Transaction simulation test failed: {e}")
        return False
    finally:
        await blockchain_provider.close()


async def test_transaction_signing():
    """Test transaction signing functionality."""
    print("\n🔐 Testing Transaction Signing\n")
    
    blockchain_provider, oracle = await initialize_dependencies()
    
    try:
        builder = RealTransactionBuilder(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            private_key="0x" + "1" * 64  # Dummy key
        )
        
        await builder.initialize()
        
        # Build and simulate a test transaction
        arbitrage_path = create_mock_arbitrage_path()
        transaction = await builder.build_simple_arbitrage(
            arbitrage_path=arbitrage_path,
            input_amount=Decimal("1000")
        )
        
        if not transaction:
            print("   ❌ Failed to build transaction for signing test")
            return False
        
        # Simulate first
        await builder.simulate_transaction(transaction)
        
        print(f"   🔄 Signing transaction {transaction.transaction_id}...")
        
        # Sign the transaction
        signing_success = await builder.sign_transaction(transaction)
        
        if signing_success:
            print("   ✅ Transaction signed successfully")
            print(f"      Status: {transaction.status.value}")
            
            if transaction.signed_tx:
                print(f"      Signed tx length: {len(transaction.signed_tx)} bytes")
                print(f"      Signed tx preview: {transaction.signed_tx.hex()[:64]}...")
            
            return True
        else:
            print("   ❌ Transaction signing failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Transaction signing test failed: {e}")
        return False
    finally:
        await blockchain_provider.close()


async def test_builder_statistics():
    """Test transaction builder statistics tracking."""
    print("\n📊 Testing Builder Statistics\n")
    
    blockchain_provider, oracle = await initialize_dependencies()
    
    try:
        builder = RealTransactionBuilder(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            private_key="0x" + "1" * 64
        )
        
        await builder.initialize()
        
        # Build multiple test transactions
        arbitrage_path = create_mock_arbitrage_path()
        
        transactions = []
        for i in range(3):
            tx = await builder.build_simple_arbitrage(
                arbitrage_path=arbitrage_path,
                input_amount=Decimal("1000") * (i + 1)
            )
            if tx:
                transactions.append(tx)
                await builder.simulate_transaction(tx)
        
        print(f"   📊 Built {len(transactions)} test transactions")
        
        # Get statistics
        stats = builder.get_transaction_stats()
        
        print("   📊 Builder Statistics:")
        print(f"      Transactions built: {stats['transactions_built']}")
        print(f"      Transactions simulated: {stats['transactions_simulated']}")
        print(f"      Pending transactions: {stats['pending_transactions']}")
        print(f"      Average build time: {stats['average_build_time_ms']:.1f}ms")
        print(f"      Supported strategies: {stats['supported_strategies']}")
        print(f"      Supported protocols: {stats['supported_protocols']}")
        
        if stats['transactions_built'] > 0:
            print("   ✅ Statistics tracking working correctly")
            return True
        else:
            print("   ❌ No transactions built for statistics")
            return False
            
    except Exception as e:
        print(f"   ❌ Statistics test failed: {e}")
        return False
    finally:
        await blockchain_provider.close()


async def test_production_readiness():
    """Test production readiness of the transaction builder."""
    print("\n🚀 Testing Production Readiness\n")
    
    blockchain_provider, oracle = await initialize_dependencies()
    
    try:
        builder = RealTransactionBuilder(
            blockchain_provider=blockchain_provider,
            oracle=oracle,
            private_key=None  # No private key for production readiness test
        )
        
        await builder.initialize()
        
        print("   📊 Production Readiness Assessment:")
        
        # Check contract interface coverage
        interfaces = builder.contract_interfaces
        required_contracts = ["uniswap_v3_router", "aave_v3_pool"]
        
        interface_coverage = 0
        for contract in required_contracts:
            if contract in interfaces:
                print(f"      ✅ {contract} interface loaded")
                interface_coverage += 1
            else:
                print(f"      ❌ {contract} interface missing")
        
        # Check protocol registry integration
        uniswap_router = production_registry.get_contract_address("uniswap_v3", "ethereum", "swap_router")
        aave_pool = production_registry.get_contract_address("aave_v3", "ethereum", "pool")
        
        registry_integration = 0
        if uniswap_router:
            print(f"      ✅ Uniswap V3 router: {uniswap_router}")
            registry_integration += 1
        else:
            print("      ❌ Uniswap V3 router address not found")
        
        if aave_pool:
            print(f"      ✅ Aave V3 pool: {aave_pool}")
            registry_integration += 1
        else:
            print("      ❌ Aave V3 pool address not found")
        
        # Check blockchain connectivity
        web3 = await blockchain_provider.get_web3("ethereum")
        blockchain_ready = 0
        if web3:
            block_number = await web3.eth.block_number
            print(f"      ✅ Ethereum connectivity: Block {block_number:,}")
            blockchain_ready = 1
        else:
            print("      ❌ Ethereum connectivity failed")
        
        # Check supported strategies
        stats = builder.get_transaction_stats()
        strategy_support = len(stats['supported_strategies'])
        print(f"      ✅ Supported strategies: {strategy_support}")
        
        # Calculate readiness score
        total_checks = 4
        passed_checks = (
            (interface_coverage >= 2) +
            (registry_integration >= 2) +
            blockchain_ready +
            (strategy_support >= 2)
        )
        
        readiness_score = (passed_checks / total_checks) * 100
        
        print(f"\n   📊 Production Readiness Score: {readiness_score:.0f}/100")
        
        if readiness_score >= 75:
            print("   🚀 Transaction builder ready for production")
            return True
        elif readiness_score >= 50:
            print("   ⚠️  Transaction builder needs improvements")
            return False
        else:
            print("   ❌ Transaction builder not ready for production")
            return False
            
    except Exception as e:
        print(f"   ❌ Production readiness test failed: {e}")
        return False
    finally:
        await blockchain_provider.close()


async def main():
    """Run all transaction builder tests."""
    print("🚀 Real Transaction Builder Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    try:
        # Test 1: Builder initialization
        builder = await test_builder_initialization()
        test_results['initialization'] = builder is not None
        
        # Test 2: Simple arbitrage building
        simple_tx = await test_simple_arbitrage_building()
        test_results['simple_arbitrage'] = simple_tx is not None
        
        # Test 3: Flash loan arbitrage building
        flash_tx = await test_flash_loan_arbitrage_building()
        test_results['flash_loan_arbitrage'] = flash_tx is not None
        
        # Test 4: Transaction simulation
        simulation_success = await test_transaction_simulation()
        test_results['simulation'] = simulation_success
        
        # Test 5: Transaction signing
        signing_success = await test_transaction_signing()
        test_results['signing'] = signing_success
        
        # Test 6: Statistics tracking
        stats_success = await test_builder_statistics()
        test_results['statistics'] = stats_success
        
        # Test 7: Production readiness
        production_ready = await test_production_readiness()
        test_results['production_readiness'] = production_ready
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        test_results['overall'] = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 Test Suite Summary")
    print("=" * 60)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        test_display = test_name.replace('_', ' ').title()
        print(f"   {status}: {test_display}")
    
    print(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Transaction builder ready for production.")
        print("\n✅ Task 14.5: Real Transaction Building & Testing - COMPLETED")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed. Minor issues need attention.")
    else:
        print("❌ Multiple test failures. Transaction builder needs significant work.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    asyncio.run(main())