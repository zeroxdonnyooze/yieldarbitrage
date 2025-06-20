"""Blockchain provider for multi-chain EVM interactions."""
import asyncio
from typing import Dict, Optional, List, Any, Union
import logging

from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider
from web3.exceptions import Web3Exception

from ..config.settings import settings
from .async_multicall import AsyncMulticallProvider, MulticallRequest, MulticallResult


logger = logging.getLogger(__name__)


class ChainConfig:
    """Configuration for a blockchain network."""
    
    def __init__(
        self,
        name: str,
        chain_id: int,
        rpc_url: str,
        multicall_address: Optional[str] = None,
        block_explorer_url: Optional[str] = None,
        native_currency: str = "ETH"
    ):
        self.name = name
        self.chain_id = chain_id
        self.rpc_url = rpc_url
        self.multicall_address = multicall_address
        self.block_explorer_url = block_explorer_url
        self.native_currency = native_currency


class BlockchainProvider:
    """Async blockchain provider for multi-chain EVM interactions."""
    
    def __init__(self):
        """Initialize the blockchain provider."""
        self.web3_instances: Dict[str, AsyncWeb3] = {}
        self.chain_configs: Dict[str, ChainConfig] = {}
        self.multicall_providers: Dict[str, AsyncMulticallProvider] = {}
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize all blockchain connections."""
        if self._initialized:
            return
            
        logger.info("🔗 Initializing blockchain connections...")
        
        # Define chain configurations
        await self._setup_chain_configs()
        
        # Initialize Web3 instances for available chains
        await self._initialize_web3_instances()
        
        # Initialize multicall providers for each chain
        await self._initialize_multicall_providers()
        
        self._initialized = True
        logger.info(f"✅ Initialized {len(self.web3_instances)} blockchain connections with multicall support")
    
    async def _setup_chain_configs(self) -> None:
        """Set up configuration for supported chains."""
        configs = {}
        
        # Ethereum mainnet
        if settings.ethereum_rpc_url:
            configs["ethereum"] = ChainConfig(
                name="Ethereum",
                chain_id=1,
                rpc_url=settings.ethereum_rpc_url,
                multicall_address="0x5BA1e12693Dc8F9c48aAD8770482f4739bEeD696",  # Multicall3
                block_explorer_url="https://etherscan.io",
                native_currency="ETH"
            )
        
        # Arbitrum mainnet
        if settings.arbitrum_rpc_url:
            configs["arbitrum"] = ChainConfig(
                name="Arbitrum",
                chain_id=42161,
                rpc_url=settings.arbitrum_rpc_url,
                multicall_address="0xcA11bde05977b3631167028862bE2a173976CA11",  # Multicall3
                block_explorer_url="https://arbiscan.io",
                native_currency="ETH"
            )
        
        # Base mainnet
        if settings.base_rpc_url:
            configs["base"] = ChainConfig(
                name="Base",
                chain_id=8453,
                rpc_url=settings.base_rpc_url,
                multicall_address="0xcA11bde05977b3631167028862bE2a173976CA11",  # Multicall3
                block_explorer_url="https://basescan.org",
                native_currency="ETH"
            )
        
        # Sonic Labs (S) mainnet
        if settings.sonic_rpc_url:
            configs["sonic"] = ChainConfig(
                name="Sonic",
                chain_id=146,  # Sonic mainnet chain ID
                rpc_url=settings.sonic_rpc_url,
                multicall_address="0xcA11bde05977b3631167028862bE2a173976CA11",  # Multicall3
                block_explorer_url="https://sonicscan.org",
                native_currency="S"
            )
        
        # Berachain mainnet
        if settings.berachain_rpc_url:
            configs["berachain"] = ChainConfig(
                name="Berachain", 
                chain_id=80094,  # Berachain mainnet chain ID
                rpc_url=settings.berachain_rpc_url,
                multicall_address="0xcA11bde05977b3631167028862bE2a173976CA11",  # Multicall3
                block_explorer_url="https://beratrail.io",
                native_currency="BERA"
            )
        
        self.chain_configs = configs
        logger.info(f"📋 Configured {len(configs)} chains: {list(configs.keys())}")
    
    async def _initialize_web3_instances(self) -> None:
        """Initialize Web3 instances for all configured chains."""
        for chain_name, config in self.chain_configs.items():
            try:
                # Create async HTTP provider
                provider = AsyncHTTPProvider(
                    config.rpc_url,
                    request_kwargs={"timeout": 30}
                )
                
                # Create AsyncWeb3 instance
                w3 = AsyncWeb3(provider)
                
                # Test connection
                await w3.is_connected()
                chain_id = await w3.eth.chain_id
                
                if chain_id != config.chain_id:
                    logger.warning(
                        f"⚠️ Chain ID mismatch for {chain_name}: "
                        f"expected {config.chain_id}, got {chain_id}"
                    )
                
                self.web3_instances[chain_name] = w3
                logger.info(f"✅ Connected to {config.name} (chain ID: {chain_id})")
                
            except Exception as e:
                logger.error(f"❌ Failed to connect to {chain_name}: {e}")
                continue
    
    async def _initialize_multicall_providers(self) -> None:
        """Initialize multicall providers for all available chains."""
        for chain_name, w3 in self.web3_instances.items():
            config = self.chain_configs[chain_name]
            
            try:
                multicall_provider = AsyncMulticallProvider(
                    w3=w3,
                    multicall_address=config.multicall_address,
                    max_batch_size=100,  # Configurable batch size
                    use_multicall_py=True  # Try to use multicall.py if available
                )
                
                self.multicall_providers[chain_name] = multicall_provider
                logger.info(f"✅ Initialized multicall provider for {config.name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize multicall for {chain_name}: {e}")
                continue
    
    async def get_web3(self, chain_name: str) -> Optional[AsyncWeb3]:
        """Get Web3 instance for a specific chain."""
        if not self._initialized:
            await self.initialize()
        
        return self.web3_instances.get(chain_name.lower())
    
    async def get_chain_config(self, chain_name: str) -> Optional[ChainConfig]:
        """Get chain configuration for a specific chain."""
        if not self._initialized:
            await self.initialize()
        
        return self.chain_configs.get(chain_name.lower())
    
    async def get_supported_chains(self) -> List[str]:
        """Get list of supported chain names."""
        if not self._initialized:
            await self.initialize()
        
        return list(self.web3_instances.keys())
    
    async def get_multicall_provider(self, chain_name: str) -> Optional[AsyncMulticallProvider]:
        """Get multicall provider for a specific chain."""
        if not self._initialized:
            await self.initialize()
        
        return self.multicall_providers.get(chain_name.lower())
    
    async def is_connected(self, chain_name: str) -> bool:
        """Check if connected to a specific chain."""
        w3 = await self.get_web3(chain_name)
        if not w3:
            return False
        
        try:
            return await w3.is_connected()
        except Exception:
            return False
    
    async def get_block_number(self, chain_name: str) -> Optional[int]:
        """Get current block number for a chain."""
        w3 = await self.get_web3(chain_name)
        if not w3:
            return None
        
        try:
            return await w3.eth.block_number
        except Web3Exception as e:
            logger.error(f"Failed to get block number for {chain_name}: {e}")
            return None
    
    async def get_gas_price(self, chain_name: str) -> Optional[int]:
        """Get current gas price for a chain (in wei)."""
        w3 = await self.get_web3(chain_name)
        if not w3:
            return None
        
        try:
            return await w3.eth.gas_price
        except Web3Exception as e:
            logger.error(f"Failed to get gas price for {chain_name}: {e}")
            return None
    
    async def get_balance(self, chain_name: str, address: str) -> Optional[int]:
        """Get native token balance for an address (in wei)."""
        w3 = await self.get_web3(chain_name)
        if not w3:
            return None
        
        try:
            return await w3.eth.get_balance(address)
        except Web3Exception as e:
            logger.error(f"Failed to get balance for {address} on {chain_name}: {e}")
            return None
    
    async def get_chain_health(self, chain_name: str) -> Dict[str, Any]:
        """Get health information for a specific chain."""
        w3 = await self.get_web3(chain_name)
        config = await self.get_chain_config(chain_name)
        
        if not w3 or not config:
            return {
                "chain": chain_name,
                "status": "not_configured",
                "connected": False
            }
        
        try:
            is_connected = await w3.is_connected()
            block_number = None
            gas_price = None
            
            if is_connected:
                block_number = await self.get_block_number(chain_name)
                gas_price = await self.get_gas_price(chain_name)
            
            return {
                "chain": chain_name,
                "name": config.name,
                "chain_id": config.chain_id,
                "status": "healthy" if is_connected else "unhealthy",
                "connected": is_connected,
                "block_number": block_number,
                "gas_price": gas_price,
                "native_currency": config.native_currency
            }
        except Exception as e:
            return {
                "chain": chain_name,
                "name": config.name if config else "Unknown",
                "status": "error",
                "connected": False,
                "error": str(e)
            }
    
    async def get_all_chain_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health information for all configured chains."""
        if not self._initialized:
            await self.initialize()
        
        health_checks = {}
        
        # Check configured chains
        for chain_name in self.chain_configs.keys():
            health_checks[chain_name] = await self.get_chain_health(chain_name)
        
        return health_checks
    
    async def multicall_token_balances(
        self,
        chain_name: str,
        token_contracts: List[str],
        holder_address: str
    ) -> Dict[str, MulticallResult]:
        """
        Get multiple token balances efficiently using multicall.
        
        Args:
            chain_name: Name of the blockchain network
            token_contracts: List of ERC20 token contract addresses
            holder_address: Address to check balances for
            
        Returns:
            Dictionary mapping token address to balance result
        """
        multicall_provider = await self.get_multicall_provider(chain_name)
        if not multicall_provider:
            raise ValueError(f"Multicall not available for {chain_name}")
        
        try:
            balance_results = await multicall_provider.get_token_balances(
                token_contracts, holder_address
            )
            
            logger.info(f"Retrieved balances for {len(token_contracts)} tokens on {chain_name}")
            return balance_results
            
        except Exception as e:
            logger.error(f"Failed to get token balances on {chain_name}: {e}")
            raise
    
    async def multicall_contract_data(
        self,
        chain_name: str,
        contract_calls: List[MulticallRequest]
    ) -> List[MulticallResult]:
        """
        Execute multiple contract calls efficiently using multicall.
        
        Args:
            chain_name: Name of the blockchain network
            contract_calls: List of contract calls to execute
            
        Returns:
            List of multicall results
        """
        multicall_provider = await self.get_multicall_provider(chain_name)
        if not multicall_provider:
            raise ValueError(f"Multicall not available for {chain_name}")
        
        try:
            results = await multicall_provider.execute_calls(contract_calls)
            
            successful_calls = [r for r in results if r.success]
            failed_calls = [r for r in results if not r.success]
            
            logger.info(
                f"Executed {len(contract_calls)} calls on {chain_name}: "
                f"{len(successful_calls)} successful, {len(failed_calls)} failed"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute multicall on {chain_name}: {e}")
            raise
    
    async def get_defi_protocol_data(
        self,
        chain_name: str,
        protocol_contracts: Dict[str, str]  # protocol_name -> contract_address
    ) -> Dict[str, MulticallResult]:
        """
        Get common DeFi protocol data efficiently.
        
        Args:
            chain_name: Name of the blockchain network
            protocol_contracts: Dictionary mapping protocol names to contract addresses
            
        Returns:
            Dictionary mapping protocol name to result
        """
        multicall_provider = await self.get_multicall_provider(chain_name)
        if not multicall_provider:
            raise ValueError(f"Multicall not available for {chain_name}")
        
        # Common DeFi protocol calls
        contract_calls = []
        
        for protocol_name, contract_address in protocol_contracts.items():
            # Add common calls for each protocol
            contract_calls.extend([
                # Total supply
                (contract_address, "0x18160ddd", f"{protocol_name}_totalSupply"),
                # Symbol  
                (contract_address, "0x95d89b41", f"{protocol_name}_symbol"),
                # Decimals
                (contract_address, "0x313ce567", f"{protocol_name}_decimals")
            ])
        
        try:
            data_results = await multicall_provider.get_multiple_contract_data(contract_calls)
            
            logger.info(f"Retrieved data for {len(protocol_contracts)} DeFi protocols on {chain_name}")
            return data_results
            
        except Exception as e:
            logger.error(f"Failed to get DeFi protocol data on {chain_name}: {e}")
            raise
    
    async def close(self) -> None:
        """Close all blockchain connections."""
        logger.info("🔒 Closing blockchain connections...")
        
        # Close multicall providers
        for chain_name, multicall_provider in self.multicall_providers.items():
            try:
                await multicall_provider.close()
            except Exception as e:
                logger.error(f"Error closing multicall provider for {chain_name}: {e}")
        
        # Close Web3 connections
        for chain_name, w3 in self.web3_instances.items():
            try:
                # Close the provider if it has a close method
                if hasattr(w3.provider, 'close'):
                    await w3.provider.close()
            except Exception as e:
                logger.error(f"Error closing {chain_name} connection: {e}")
        
        self.web3_instances.clear()
        self.multicall_providers.clear()
        self._initialized = False
        logger.info("✅ All blockchain connections closed")


# Global provider instance
blockchain_provider = BlockchainProvider()