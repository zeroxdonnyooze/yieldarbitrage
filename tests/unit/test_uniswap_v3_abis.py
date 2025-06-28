"""Unit tests for Uniswap V3 ABI management."""
import pytest

from yield_arbitrage.protocols.abi_manager import ABIManager, abi_manager
from yield_arbitrage.protocols.contracts import (
    get_uniswap_v3_contract,
    get_well_known_tokens,
    is_supported_chain,
    get_supported_chains,
    UNISWAP_V3_FEE_TIERS
)


class TestABIManager:
    """Test ABI manager functionality."""
    
    def test_abi_manager_initialization(self):
        """Test that ABI manager initializes correctly."""
        manager = ABIManager()
        
        protocols = manager.list_protocols()
        assert "uniswap_v3" in protocols
        assert "erc20" in protocols
    
    def test_get_uniswap_v3_factory_abi(self):
        """Test getting Uniswap V3 Factory ABI."""
        abi = abi_manager.get_uniswap_v3_factory_abi()
        
        assert isinstance(abi, list)
        assert len(abi) > 0
        
        # Check for essential functions
        function_names = [item["name"] for item in abi if item.get("type") == "function"]
        assert "getPool" in function_names
        
        # Check for essential events
        event_names = [item["name"] for item in abi if item.get("type") == "event"]
        assert "PoolCreated" in event_names
    
    def test_get_uniswap_v3_quoter_abi(self):
        """Test getting Uniswap V3 Quoter ABI."""
        abi = abi_manager.get_uniswap_v3_quoter_abi()
        
        assert isinstance(abi, list)
        assert len(abi) > 0
        
        # Check for essential functions
        function_names = [item["name"] for item in abi if item.get("type") == "function"]
        assert "quoteExactInputSingle" in function_names
        assert "quoteExactOutputSingle" in function_names
    
    def test_get_uniswap_v3_pool_abi(self):
        """Test getting Uniswap V3 Pool ABI."""
        abi = abi_manager.get_uniswap_v3_pool_abi()
        
        assert isinstance(abi, list)
        assert len(abi) > 0
        
        # Check for essential functions
        function_names = [item["name"] for item in abi if item.get("type") == "function"]
        assert "token0" in function_names
        assert "token1" in function_names
        assert "fee" in function_names
        assert "slot0" in function_names
        assert "liquidity" in function_names
        
        # Check for essential events
        event_names = [item["name"] for item in abi if item.get("type") == "event"]
        assert "Swap" in event_names
        assert "Mint" in event_names
    
    def test_get_erc20_abi(self):
        """Test getting ERC20 ABI."""
        abi = abi_manager.get_erc20_abi()
        
        assert isinstance(abi, list)
        assert len(abi) > 0
        
        # Check for essential functions
        function_names = [item["name"] for item in abi if item.get("type") == "function"]
        assert "name" in function_names
        assert "symbol" in function_names
        assert "decimals" in function_names
        assert "totalSupply" in function_names
        assert "balanceOf" in function_names
    
    def test_get_abi_with_protocol_and_contract(self):
        """Test getting ABI with protocol and contract type."""
        # Test valid combinations
        factory_abi = abi_manager.get_abi("uniswap_v3", "factory")
        assert factory_abi is not None
        
        quoter_abi = abi_manager.get_abi("uniswap_v3", "quoter")
        assert quoter_abi is not None
        
        pool_abi = abi_manager.get_abi("uniswap_v3", "pool")
        assert pool_abi is not None
        
        erc20_abi = abi_manager.get_abi("erc20")
        assert erc20_abi is not None
    
    def test_get_abi_invalid_cases(self):
        """Test getting ABI with invalid inputs."""
        # Invalid protocol
        assert abi_manager.get_abi("invalid_protocol") is None
        
        # Invalid contract type
        assert abi_manager.get_abi("uniswap_v3", "invalid_contract") is None
        
        # Missing contract type for complex protocol
        assert abi_manager.get_abi("uniswap_v3") is None
    
    def test_list_contract_types(self):
        """Test listing contract types for protocols."""
        # Uniswap V3 has multiple contract types
        uniswap_types = abi_manager.list_contract_types("uniswap_v3")
        assert "factory" in uniswap_types
        assert "quoter" in uniswap_types
        assert "pool" in uniswap_types
        
        # ERC20 is a simple protocol
        erc20_types = abi_manager.list_contract_types("erc20")
        assert erc20_types == ["main"]
        
        # Invalid protocol
        invalid_types = abi_manager.list_contract_types("invalid")
        assert invalid_types == []
    
    def test_validate_abi(self):
        """Test ABI validation."""
        # Valid ABI
        valid_abi = [
            {
                "type": "function",
                "name": "test",
                "inputs": [],
                "outputs": []
            }
        ]
        assert abi_manager.validate_abi(valid_abi) is True
        
        # Invalid ABI - not a list
        assert abi_manager.validate_abi({}) is False
        
        # Invalid ABI - missing type
        invalid_abi = [{"name": "test"}]
        assert abi_manager.validate_abi(invalid_abi) is False
        
        # Invalid ABI - missing name for function
        invalid_abi = [{"type": "function"}]
        assert abi_manager.validate_abi(invalid_abi) is False


class TestContractAddresses:
    """Test contract address management."""
    
    def test_get_uniswap_v3_contract_ethereum(self):
        """Test getting Uniswap V3 contract addresses for Ethereum."""
        factory = get_uniswap_v3_contract("ethereum", "factory")
        assert factory == "0x1F98431c8aD98523631AE4a59f267346ea31F984"
        
        quoter = get_uniswap_v3_contract("ethereum", "quoter")
        assert quoter == "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6"
        
        # Test case insensitive
        factory_upper = get_uniswap_v3_contract("ETHEREUM", "FACTORY")
        assert factory_upper == factory
    
    def test_get_uniswap_v3_contract_arbitrum(self):
        """Test getting Uniswap V3 contract addresses for Arbitrum."""
        factory = get_uniswap_v3_contract("arbitrum", "factory")
        assert factory == "0x1F98431c8aD98523631AE4a59f267346ea31F984"
        
        quoter = get_uniswap_v3_contract("arbitrum", "quoter")
        assert quoter == "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6"
    
    def test_get_uniswap_v3_contract_base(self):
        """Test getting Uniswap V3 contract addresses for Base."""
        factory = get_uniswap_v3_contract("base", "factory")
        assert factory == "0x33128a8fC17869897dcE68Ed026d694621f6FDfD"
        
        quoter = get_uniswap_v3_contract("base", "quoter")
        assert quoter == "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a"
    
    def test_get_uniswap_v3_contract_invalid(self):
        """Test getting contract addresses for invalid inputs."""
        # Invalid chain
        assert get_uniswap_v3_contract("invalid_chain", "factory") is None
        
        # Invalid contract type
        assert get_uniswap_v3_contract("ethereum", "invalid_contract") is None
    
    def test_get_well_known_tokens(self):
        """Test getting well-known token addresses."""
        eth_tokens = get_well_known_tokens("ethereum")
        assert "WETH" in eth_tokens
        assert "USDC" in eth_tokens
        assert "USDT" in eth_tokens
        assert "DAI" in eth_tokens
        assert "WBTC" in eth_tokens
        
        arb_tokens = get_well_known_tokens("arbitrum")
        assert "WETH" in arb_tokens
        assert "USDC" in arb_tokens
        assert "USDT" in arb_tokens
        
        # Invalid chain
        invalid_tokens = get_well_known_tokens("invalid_chain")
        assert invalid_tokens == {}
    
    def test_is_supported_chain(self):
        """Test chain support checking."""
        assert is_supported_chain("ethereum") is True
        assert is_supported_chain("arbitrum") is True
        assert is_supported_chain("base") is True
        assert is_supported_chain("ETHEREUM") is True  # Case insensitive
        
        assert is_supported_chain("invalid_chain") is False
        assert is_supported_chain("polygon") is False  # Not yet supported
    
    def test_get_supported_chains(self):
        """Test getting list of supported chains."""
        chains = get_supported_chains()
        
        assert isinstance(chains, list)
        assert "ethereum" in chains
        assert "arbitrum" in chains
        assert "base" in chains
        assert len(chains) >= 3
    
    def test_uniswap_v3_fee_tiers(self):
        """Test Uniswap V3 fee tiers."""
        assert isinstance(UNISWAP_V3_FEE_TIERS, list)
        assert 100 in UNISWAP_V3_FEE_TIERS    # 0.01%
        assert 500 in UNISWAP_V3_FEE_TIERS    # 0.05%
        assert 3000 in UNISWAP_V3_FEE_TIERS   # 0.3%
        assert 10000 in UNISWAP_V3_FEE_TIERS  # 1%
        
        # Ensure all are integers
        assert all(isinstance(fee, int) for fee in UNISWAP_V3_FEE_TIERS)