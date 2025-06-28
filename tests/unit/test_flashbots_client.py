"""
Unit tests for Flashbots Client.

Tests the Flashbots integration functionality including bundle creation,
submission, simulation, and monitoring.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal

from yield_arbitrage.mev_protection.flashbots_client import (
    FlashbotsClient, FlashbotsBundle, FlashbotsBundleResponse,
    FlashbotsSimulationResult, FlashbotsNetwork,
    create_flashbots_client, submit_execution_plan_to_flashbots
)
from yield_arbitrage.execution.enhanced_transaction_builder import (
    BatchExecutionPlan, RouterTransaction
)
from yield_arbitrage.mev_protection.mev_risk_assessor import PathMEVAnalysis, MEVRiskLevel


class TestFlashbotsClient:
    """Test Flashbots client functionality."""
    
    @pytest.fixture
    def private_key(self):
        """Test private key."""
        return "0x" + "1" * 64
    
    @pytest.fixture
    def flashbots_client(self, private_key):
        """Create Flashbots client instance."""
        return FlashbotsClient(
            private_key=private_key,
            network=FlashbotsNetwork.MAINNET
        )
    
    @pytest.fixture
    def sample_router_transaction(self):
        """Create sample router transaction."""
        return RouterTransaction(
            segment_id="test_segment",
            to_address="0x1234567890123456789012345678901234567890",
            from_address="0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
            data=b"\x12\x34\x56\x78",
            gas_limit=500_000,
            estimated_gas=450_000
        )
    
    @pytest.fixture
    def sample_execution_plan(self, sample_router_transaction):
        """Create sample execution plan."""
        return BatchExecutionPlan(
            plan_id="test_plan",
            router_address="0x1234567890123456789012345678901234567890",
            executor_address="0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
            segments=[sample_router_transaction],
            total_gas_estimate=500_000
        )
    
    @pytest.fixture
    def sample_mev_analysis(self):
        """Create sample MEV analysis."""
        return PathMEVAnalysis(
            path_id="test_path",
            total_edges=2,
            overall_risk_level=MEVRiskLevel.HIGH,
            compounded_risk=0.8,
            estimated_total_mev_loss_bps=50.0
        )


def test_flashbots_client_initialization(private_key):
    """Test Flashbots client initialization."""
    client = FlashbotsClient(private_key, FlashbotsNetwork.MAINNET)
    
    assert client.private_key == private_key
    assert client.network == FlashbotsNetwork.MAINNET
    assert client.relay_url == "https://relay.flashbots.net"
    assert client.account.address is not None
    assert len(client.submitted_bundles) == 0


def test_network_relay_urls(private_key):
    """Test different network relay URLs."""
    networks_urls = [
        (FlashbotsNetwork.MAINNET, "https://relay.flashbots.net"),
        (FlashbotsNetwork.GOERLI, "https://relay-goerli.flashbots.net"),
        (FlashbotsNetwork.SEPOLIA, "https://relay-sepolia.flashbots.net")
    ]
    
    for network, expected_url in networks_urls:
        client = FlashbotsClient(private_key, network)
        assert client.relay_url == expected_url


@pytest.mark.asyncio
async def test_create_bundle_from_execution_plan(
    flashbots_client,
    sample_execution_plan,
    sample_mev_analysis
):
    """Test creating Flashbots bundle from execution plan."""
    
    # Mock next block number
    flashbots_client._get_next_block_number = AsyncMock(return_value=18_500_000)
    
    bundle = await flashbots_client.create_bundle_from_execution_plan(
        sample_execution_plan,
        sample_mev_analysis,
        priority_fee_gwei=10.0
    )
    
    # Verify bundle structure
    assert isinstance(bundle, FlashbotsBundle)
    assert len(bundle.transactions) == 1
    assert bundle.target_block == 18_500_000
    assert bundle.max_block_number == 18_500_003  # target + 3
    assert bundle.bundle_id == f"bundle_{sample_execution_plan.plan_id}_18500000"
    assert bundle.estimated_gas_used == 500_000
    assert bundle.priority_fee_wei == int(10.0 * 1e9)


def test_convert_router_transaction_to_flashbots(
    flashbots_client,
    sample_router_transaction
):
    """Test converting router transaction to Flashbots format."""
    
    fb_tx = flashbots_client._convert_router_transaction_to_flashbots(
        sample_router_transaction,
        priority_fee_gwei=5.0,
        is_last=True
    )
    
    # Verify Flashbots transaction structure
    assert "signedTransaction" in fb_tx
    assert "hash" in fb_tx
    assert "account" in fb_tx
    assert "decodedTxn" in fb_tx
    
    decoded = fb_tx["decodedTxn"]
    assert decoded["to"] == sample_router_transaction.to_address
    assert decoded["gas"] == hex(sample_router_transaction.gas_limit)
    assert decoded["type"] == "0x2"  # EIP-1559
    
    # Last transaction should have coinbase payment
    assert decoded["value"] != "0"  # Should have MEV payment


@pytest.mark.asyncio
async def test_bundle_simulation(flashbots_client):
    """Test bundle simulation functionality."""
    
    # Create test bundle
    bundle = FlashbotsBundle(
        transactions=[{"signedTransaction": "0x" + "0" * 100}],
        target_block=18_500_000,
        bundle_id="test_bundle"
    )
    
    # Mock HTTP session
    mock_response = Mock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "result": {
            "totalGasUsed": 500_000,
            "coinbaseDiff": "0x1bc16d674ec80000",  # 2 ETH in hex
            "results": [{"value": "0x0"}],
            "stateBlockNumber": 18_499_999,
            "bundleBlockNumber": 18_500_000
        }
    })
    
    flashbots_client.session = Mock()
    flashbots_client.session.post = AsyncMock(return_value=mock_response)
    flashbots_client._get_latest_block_number = AsyncMock(return_value=18_499_999)
    
    # Test simulation
    result = await flashbots_client.simulate_bundle(bundle)
    
    # Verify simulation result
    assert isinstance(result, FlashbotsSimulationResult)
    assert result.success is True
    assert result.total_gas_used == 500_000
    assert result.coinbase_diff > 0
    assert result.state_block == 18_499_999


@pytest.mark.asyncio
async def test_bundle_simulation_error(flashbots_client):
    """Test bundle simulation error handling."""
    
    bundle = FlashbotsBundle(
        transactions=[{"signedTransaction": "0x" + "0" * 100}],
        target_block=18_500_000,
        bundle_id="test_bundle"
    )
    
    # Mock error response
    mock_response = Mock()
    mock_response.status = 400
    mock_response.text = AsyncMock(return_value="Bad Request")
    
    flashbots_client.session = Mock()
    flashbots_client.session.post = AsyncMock(return_value=mock_response)
    flashbots_client._get_latest_block_number = AsyncMock(return_value=18_499_999)
    
    # Test simulation error
    result = await flashbots_client.simulate_bundle(bundle)
    
    assert result.success is False
    assert "HTTP 400" in result.error


@pytest.mark.asyncio
async def test_bundle_submission(flashbots_client):
    """Test bundle submission functionality."""
    
    bundle = FlashbotsBundle(
        transactions=[{"signedTransaction": "0x" + "0" * 100}],
        target_block=18_500_000,
        bundle_id="test_bundle"
    )
    
    # Mock successful submission
    mock_response = Mock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "result": {"bundleHash": "0xabcdef1234567890"}
    })
    
    flashbots_client.session = Mock()
    flashbots_client.session.post = AsyncMock(return_value=mock_response)
    
    # Mock simulation (will be called first)
    flashbots_client.simulate_bundle = AsyncMock(return_value=FlashbotsSimulationResult(
        success=True,
        bundle_hash="test_bundle",
        total_gas_used=500_000
    ))
    
    # Test submission
    response = await flashbots_client.submit_bundle(bundle, simulate_first=True)
    
    # Verify submission response
    assert isinstance(response, FlashbotsBundleResponse)
    assert response.success is True
    assert response.bundle_hash == "0xabcdef1234567890"
    
    # Verify tracking
    assert "0xabcdef1234567890" in flashbots_client.submitted_bundles
    assert flashbots_client.stats["bundles_submitted"] == 1


@pytest.mark.asyncio
async def test_bundle_inclusion_monitoring(flashbots_client):
    """Test bundle inclusion monitoring."""
    
    bundle_hash = "0xabcdef1234567890"
    target_block = 18_500_000
    
    # Create response for tracking
    response = FlashbotsBundleResponse(
        bundle_hash=bundle_hash,
        success=True
    )
    flashbots_client.bundle_responses[bundle_hash] = response
    
    # Mock inclusion check
    flashbots_client.check_bundle_inclusion = AsyncMock(return_value={
        "included": True,
        "block_number": target_block,
        "is_simulated": True,
        "sent_to_miners": True
    })
    
    flashbots_client._wait_for_block = AsyncMock()
    
    # Test monitoring
    result = await flashbots_client.monitor_bundle_inclusion(
        bundle_hash, target_block, max_blocks_to_wait=1
    )
    
    # Verify inclusion detected
    assert result.included_in_block == target_block
    assert result.included_at is not None
    assert flashbots_client.stats["bundles_included"] == 1


@pytest.mark.asyncio
async def test_bundle_inclusion_timeout(flashbots_client):
    """Test bundle inclusion monitoring timeout."""
    
    bundle_hash = "0xabcdef1234567890"
    target_block = 18_500_000
    
    # Create response for tracking
    response = FlashbotsBundleResponse(
        bundle_hash=bundle_hash,
        success=True
    )
    flashbots_client.bundle_responses[bundle_hash] = response
    
    # Mock inclusion check always returns false
    flashbots_client.check_bundle_inclusion = AsyncMock(return_value={
        "included": False
    })
    
    flashbots_client._wait_for_block = AsyncMock()
    
    # Test monitoring timeout
    result = await flashbots_client.monitor_bundle_inclusion(
        bundle_hash, target_block, max_blocks_to_wait=2
    )
    
    # Verify no inclusion
    assert result.included_in_block is None
    assert flashbots_client.stats["bundles_included"] == 0


def test_bundle_statistics(flashbots_client):
    """Test statistics tracking."""
    
    # Initial stats
    stats = flashbots_client.get_stats()
    assert stats["bundles_submitted"] == 0
    assert stats["bundles_included"] == 0
    assert stats["inclusion_rate"] == 0.0
    
    # Simulate some activity
    flashbots_client.stats["bundles_submitted"] = 10
    flashbots_client.stats["bundles_included"] = 7
    
    updated_stats = flashbots_client.get_stats()
    assert updated_stats["inclusion_rate"] == 70.0


@pytest.mark.asyncio
async def test_convenience_functions(private_key):
    """Test convenience functions."""
    
    # Test client creation
    client = await create_flashbots_client(private_key, FlashbotsNetwork.MAINNET)
    assert isinstance(client, FlashbotsClient)
    await client.close()
    
    # Test execution plan submission (with mocks)
    client = FlashbotsClient(private_key, FlashbotsNetwork.MAINNET)
    
    # Mock methods
    client.create_bundle_from_execution_plan = AsyncMock(return_value=FlashbotsBundle(
        transactions=[],
        target_block=18_500_000
    ))
    client.submit_bundle = AsyncMock(return_value=FlashbotsBundleResponse(
        bundle_hash="0xtest",
        success=True
    ))
    client.monitor_bundle_inclusion = AsyncMock(return_value=FlashbotsBundleResponse(
        bundle_hash="0xtest",
        success=True,
        included_in_block=18_500_000
    ))
    
    # Create mock execution plan and MEV analysis
    execution_plan = BatchExecutionPlan(
        plan_id="test",
        router_address="0x1234567890123456789012345678901234567890",
        executor_address="0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
        segments=[]
    )
    
    mev_analysis = PathMEVAnalysis(
        path_id="test",
        total_edges=1,
        overall_risk_level=MEVRiskLevel.HIGH,
        compounded_risk=0.8
    )
    
    # Test submission
    response = await submit_execution_plan_to_flashbots(
        execution_plan,
        mev_analysis,
        client
    )
    
    assert response.success is True


@pytest.mark.asyncio
async def test_request_signing(flashbots_client):
    """Test Flashbots request signing."""
    
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_callBundle",
        "params": [{"test": "data"}]
    }
    
    signed_request = flashbots_client._sign_request(request)
    
    # Verify signed request structure
    assert "signature" in signed_request
    assert signed_request["jsonrpc"] == request["jsonrpc"]
    assert signed_request["method"] == request["method"]
    assert signed_request["params"] == request["params"]


def test_flashbots_headers(flashbots_client):
    """Test Flashbots request headers."""
    
    headers = flashbots_client._get_flashbots_headers()
    
    assert "Content-Type" in headers
    assert headers["Content-Type"] == "application/json"
    assert "X-Flashbots-Signature" in headers
    assert flashbots_client.account.address in headers["X-Flashbots-Signature"]


if __name__ == "__main__":
    # Run basic test
    print("🧪 Testing Flashbots Client")
    print("=" * 40)
    
    # Test basic functionality
    private_key = "0x" + "1" * 64
    client = FlashbotsClient(private_key, FlashbotsNetwork.MAINNET)
    
    print(f"✅ Flashbots client created:")
    print(f"   - Network: {client.network.value}")
    print(f"   - Relay URL: {client.relay_url}")
    print(f"   - Account: {client.account.address}")
    
    # Test bundle creation
    router_tx = RouterTransaction(
        segment_id="test",
        to_address="0x1234567890123456789012345678901234567890",
        from_address="0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
        data=b"\x12\x34",
        gas_limit=300_000
    )
    
    fb_tx = client._convert_router_transaction_to_flashbots(
        router_tx, 5.0, True
    )
    
    print(f"✅ Router transaction converted:")
    print(f"   - Has signed transaction: {'signedTransaction' in fb_tx}")
    print(f"   - Has hash: {'hash' in fb_tx}")
    print(f"   - Account: {fb_tx['account']}")
    
    print("\n✅ Flashbots client test passed!")