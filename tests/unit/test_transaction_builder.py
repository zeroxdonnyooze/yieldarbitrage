"""Unit tests for transaction builder."""
import pytest
from decimal import Decimal

from yield_arbitrage.execution.transaction_builder import (
    TransactionBuilder,
    TokenInfo,
    TokenStandard,
    SwapParams
)
from yield_arbitrage.graph_engine.models import YieldGraphEdge, EdgeType


class TestTokenInfo:
    """Test TokenInfo dataclass."""
    
    def test_token_info_creation(self):
        """Test TokenInfo creation."""
        token = TokenInfo(
            address="0xA0b86a33E6417c5c6eF57e11d8CAf7d8C0f7C8F8",
            symbol="USDC",
            decimals=6,
            standard=TokenStandard.ERC20
        )
        
        assert token.address == "0xA0b86a33E6417c5c6eF57e11d8CAf7d8C0f7C8F8"
        assert token.symbol == "USDC"
        assert token.decimals == 6
        assert token.standard == TokenStandard.ERC20
    
    def test_amount_formatting(self):
        """Test amount formatting for different token decimals."""
        # USDC (6 decimals)
        usdc = TokenInfo("0xA0b86a33E6417c5c6eF57e11d8CAf7d8C0f7C8F8", "USDC", 6)
        assert usdc.format_amount(100.0) == 100_000_000  # 100 USDC
        assert usdc.format_amount(1.5) == 1_500_000      # 1.5 USDC
        assert usdc.format_amount("0.123456") == 123_456  # 0.123456 USDC
        
        # WETH (18 decimals)
        weth = TokenInfo("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "WETH", 18)
        assert weth.format_amount(1.0) == 10**18          # 1 ETH
        assert weth.format_amount(0.1) == 10**17          # 0.1 ETH
        
        # WBTC (8 decimals)
        wbtc = TokenInfo("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "WBTC", 8)
        assert wbtc.format_amount(1.0) == 100_000_000     # 1 BTC
        assert wbtc.format_amount(0.01) == 1_000_000      # 0.01 BTC
    
    def test_decimal_precision(self):
        """Test decimal precision handling."""
        token = TokenInfo("0xtest", "TEST", 18)
        
        # Test with Decimal
        amount_decimal = Decimal("1.123456789012345678")
        formatted = token.format_amount(amount_decimal)
        assert formatted == 1123456789012345678
        
        # Test with string to avoid float precision issues
        amount_string = "1.123456789012345678"
        formatted_string = token.format_amount(amount_string)
        assert formatted_string == 1123456789012345678


class TestTransactionBuilder:
    """Test TransactionBuilder functionality."""
    
    @pytest.fixture
    def builder(self):
        """Create TransactionBuilder instance."""
        return TransactionBuilder()
    
    @pytest.fixture
    def sample_trade_edge(self):
        """Create a sample trade edge."""
        return YieldGraphEdge(
            edge_id="eth_usdc_uniswap_v3",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswapv3",
            chain_name="ethereum"
        )
    
    @pytest.fixture
    def sample_lend_edge(self):
        """Create a sample lending edge."""
        return YieldGraphEdge(
            edge_id="usdc_aave_deposit",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_AUSDC",
            edge_type=EdgeType.LEND,
            protocol_name="aave",
            chain_name="ethereum"
        )
    
    def test_builder_initialization(self, builder):
        """Test builder initialization."""
        assert "uniswap_v2_router" in builder.contracts
        assert "uniswap_v3_router" in builder.contracts
        assert "aave_pool" in builder.contracts
        assert "WETH" in builder.tokens
        assert "USDC" in builder.tokens
        assert "transfer" in builder.function_sigs
        assert "swapExactTokensForTokens" in builder.function_sigs
    
    def test_get_token_info(self, builder):
        """Test token info extraction from asset IDs."""
        # Test known tokens
        weth_info = builder._get_token_info("ETH_MAINNET_WETH")
        assert weth_info is not None
        assert weth_info.symbol == "WETH"
        assert weth_info.decimals == 18
        
        usdc_info = builder._get_token_info("ETH_MAINNET_USDC")
        assert usdc_info is not None
        assert usdc_info.symbol == "USDC"
        assert usdc_info.decimals == 6
        
        # Test unknown token
        unknown_info = builder._get_token_info("ETH_MAINNET_UNKNOWN")
        assert unknown_info is None
    
    def test_build_uniswap_v2_transaction(self, builder, sample_trade_edge):
        """Test Uniswap V2 transaction building."""
        # Modify edge to be V2
        sample_trade_edge.protocol_name = "uniswapv2"
        
        from_address = "0x1234567890123456789012345678901234567890"
        
        tx = builder.build_edge_transaction(
            sample_trade_edge,
            input_amount=1.0,  # 1 WETH
            from_address=from_address,
            current_block=18000000
        )
        
        assert tx.from_address == from_address
        assert tx.to_address == builder.contracts["uniswap_v2_router"]
        # The transaction should be either swapExactETHForTokens (0x7ff36ab5) or swapExactTokensForTokens (0x38ed1739)
        assert tx.data.startswith("0x7ff36ab5") or tx.data.startswith("0x38ed1739")
        assert len(tx.data) > 10  # Should have encoded parameters
    
    def test_build_uniswap_v3_transaction(self, builder, sample_trade_edge):
        """Test Uniswap V3 transaction building."""
        from_address = "0x1234567890123456789012345678901234567890"
        
        tx = builder.build_edge_transaction(
            sample_trade_edge,
            input_amount=1.0,  # 1 WETH
            from_address=from_address,
            current_block=18000000
        )
        
        assert tx.from_address == from_address
        assert tx.to_address == builder.contracts["uniswap_v3_router"]
        assert tx.data.startswith("0x414bf389")  # exactInputSingle selector
        assert len(tx.data) > 10  # Should have encoded parameters
    
    def test_build_sushiswap_transaction(self, builder, sample_trade_edge):
        """Test SushiSwap transaction building."""
        # Modify edge to be SushiSwap
        sample_trade_edge.protocol_name = "sushiswap"
        
        from_address = "0x1234567890123456789012345678901234567890"
        
        tx = builder.build_edge_transaction(
            sample_trade_edge,
            input_amount=1.0,
            from_address=from_address,
            current_block=18000000
        )
        
        assert tx.from_address == from_address
        assert tx.to_address == builder.contracts["sushiswap_router"]
        assert tx.data.startswith("0x38ed1739")  # swapExactTokensForTokens selector
    
    def test_build_aave_deposit_transaction(self, builder, sample_lend_edge):
        """Test Aave deposit transaction building."""
        from_address = "0x1234567890123456789012345678901234567890"
        
        tx = builder.build_edge_transaction(
            sample_lend_edge,
            input_amount=1000.0,  # 1000 USDC
            from_address=from_address
        )
        
        assert tx.from_address == from_address
        assert tx.to_address == builder.contracts["aave_pool"]
        assert tx.data.startswith("0xe8eda9df")  # deposit selector
        assert len(tx.data) > 10  # Should have encoded parameters
    
    def test_build_path_transactions(self, builder):
        """Test building transactions for a complete path."""
        # Create a simple ETH -> USDC -> ETH arbitrage path
        edge1 = YieldGraphEdge(
            edge_id="eth_usdc_trade",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswapv3",
            chain_name="ethereum"
        )
        
        edge2 = YieldGraphEdge(
            edge_id="usdc_eth_trade",
            source_asset_id="ETH_MAINNET_USDC",
            target_asset_id="ETH_MAINNET_WETH",
            edge_type=EdgeType.TRADE,
            protocol_name="sushiswap",
            chain_name="ethereum"
        )
        
        path = [edge1, edge2]
        from_address = "0x1234567890123456789012345678901234567890"
        
        transactions = builder.build_path_transactions(
            path=path,
            initial_amount=1.0,  # 1 ETH
            from_address=from_address,
            current_block=18000000
        )
        
        assert len(transactions) == 2
        
        # First transaction should be Uniswap V3
        assert transactions[0].to_address == builder.contracts["uniswap_v3_router"]
        assert transactions[0].data.startswith("0x414bf389")  # exactInputSingle
        
        # Second transaction should be SushiSwap
        assert transactions[1].to_address == builder.contracts["sushiswap_router"]
        assert transactions[1].data.startswith("0x38ed1739")  # swapExactTokensForTokens
    
    def test_build_approval_transaction(self, builder):
        """Test ERC20 approval transaction building."""
        token_address = "0xA0b86a33E6417c5c6eF57e11d8CAf7d8C0f7C8F8"  # USDC
        spender_address = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"  # Uniswap V2 Router
        from_address = "0x1234567890123456789012345678901234567890"
        
        # Test specific amount approval
        tx = builder.build_approval_transaction(
            token_address=token_address,
            spender_address=spender_address,
            amount=1000000000,  # 1000 USDC (6 decimals)
            from_address=from_address
        )
        
        assert tx.from_address == from_address
        assert tx.to_address == token_address
        assert tx.data.startswith("0x095ea7b3")  # approve selector
        assert len(tx.data) > 10  # Should have encoded parameters
    
    def test_build_unlimited_approval_transaction(self, builder):
        """Test unlimited approval transaction building."""
        token_address = "0xA0b86a33E6417c5c6eF57e11d8CAf7d8C0f7C8F8"
        spender_address = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
        from_address = "0x1234567890123456789012345678901234567890"
        
        tx = builder.build_approval_transaction(
            token_address=token_address,
            spender_address=spender_address,
            amount="unlimited",
            from_address=from_address
        )
        
        assert tx.from_address == from_address
        assert tx.to_address == token_address
        assert tx.data.startswith("0x095ea7b3")  # approve selector
    
    def test_deadline_calculation(self, builder):
        """Test deadline calculation."""
        import time
        
        deadline = builder._calculate_deadline(18000000)
        current_time = int(time.time())
        
        # Should be approximately 20 minutes from now (1200 seconds)
        assert deadline > current_time
        assert deadline <= current_time + 1300  # Some buffer
    
    def test_unsupported_edge_type(self, builder):
        """Test handling of unsupported edge types."""
        # Create edge with unsupported type (using BACK_RUN as example)
        unsupported_edge = YieldGraphEdge(
            edge_id="unsupported_edge",
            source_asset_id="ETH_MAINNET_WETH",
            target_asset_id="ETH_MAINNET_USDC",
            edge_type=EdgeType.BACK_RUN,  # Not implemented in builder
            protocol_name="test",
            chain_name="ethereum"
        )
        
        from_address = "0x1234567890123456789012345678901234567890"
        
        with pytest.raises(ValueError, match="Unsupported edge type"):
            builder.build_edge_transaction(
                unsupported_edge,
                input_amount=1.0,
                from_address=from_address
            )
    
    def test_unknown_token_handling(self, builder):
        """Test handling of unknown tokens."""
        unknown_edge = YieldGraphEdge(
            edge_id="unknown_token_trade",
            source_asset_id="ETH_MAINNET_UNKNOWN1",
            target_asset_id="ETH_MAINNET_UNKNOWN2",
            edge_type=EdgeType.TRADE,
            protocol_name="uniswapv3",
            chain_name="ethereum"
        )
        
        from_address = "0x1234567890123456789012345678901234567890"
        
        with pytest.raises(ValueError, match="Unknown tokens"):
            builder.build_edge_transaction(
                unknown_edge,
                input_amount=1.0,
                from_address=from_address
            )


class TestSwapParams:
    """Test SwapParams dataclass."""
    
    def test_swap_params_creation(self):
        """Test SwapParams creation."""
        token_in = TokenInfo("0xWETH", "WETH", 18)
        token_out = TokenInfo("0xUSDC", "USDC", 6)
        
        params = SwapParams(
            token_in=token_in,
            token_out=token_out,
            amount_in=1.0,
            amount_out_min=1800.0,
            recipient="0x1234567890123456789012345678901234567890",
            deadline=1234567890,
            fee_tier=3000,
            slippage_tolerance=0.005
        )
        
        assert params.token_in.symbol == "WETH"
        assert params.token_out.symbol == "USDC"
        assert params.amount_in == 1.0
        assert params.amount_out_min == 1800.0
        assert params.fee_tier == 3000
        assert params.slippage_tolerance == 0.005