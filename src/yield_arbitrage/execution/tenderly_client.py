"""Tenderly API client for blockchain simulation and fork management."""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum

import aiohttp

logger = logging.getLogger(__name__)


class TenderlyNetworkId(str, Enum):
    """Supported Tenderly network identifiers."""
    ETHEREUM = "1"
    POLYGON = "137"
    BSC = "56"
    ARBITRUM = "42161"
    OPTIMISM = "10"
    AVALANCHE = "43114"
    FANTOM = "250"


@dataclass
class TenderlyTransaction:
    """Represents a transaction for Tenderly simulation."""
    from_address: str
    to_address: str
    value: str = "0"  # In wei
    gas: Optional[int] = None
    gas_price: Optional[str] = None  # In wei
    data: str = "0x"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        tx_dict = {
            "from": self.from_address,
            "to": self.to_address,
            "value": self.value,
            "input": self.data,
        }
        
        if self.gas:
            tx_dict["gas"] = hex(self.gas)
        if self.gas_price:
            tx_dict["gas_price"] = self.gas_price
            
        return tx_dict


@dataclass
class TenderlySimulationResult:
    """Result from Tenderly simulation."""
    success: bool
    gas_used: int
    gas_cost_usd: Optional[float] = None
    
    # Transaction details
    transaction_hash: Optional[str] = None
    block_number: Optional[int] = None
    
    # Execution details
    logs: List[Dict[str, Any]] = None
    trace: Optional[Dict[str, Any]] = None
    state_changes: Optional[Dict[str, Any]] = None
    
    # Error details
    error_message: Optional[str] = None
    revert_reason: Optional[str] = None
    
    # Timing
    simulation_time_ms: Optional[float] = None
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.logs is None:
            self.logs = []


@dataclass
class TenderlyVirtualTestnet:
    """Represents a Tenderly Virtual TestNet."""
    testnet_id: str
    slug: str
    display_name: str
    network_id: str
    block_number: int
    created_at: datetime
    
    # Virtual TestNet details
    chain_id: int
    admin_rpc_url: Optional[str] = None
    public_rpc_url: Optional[str] = None
    
    # Fork configuration
    fork_config: Optional[Dict[str, Any]] = None
    
    # State sync configuration
    sync_state_enabled: bool = False
    
    # Explorer configuration
    explorer_enabled: bool = False
    explorer_url: Optional[str] = None
    
    # Legacy compatibility
    is_active: bool = True
    transactions_count: int = 0
    
    @property
    def fork_id(self) -> str:
        """Legacy compatibility property."""
        return self.testnet_id


# Legacy compatibility alias
TenderlyFork = TenderlyVirtualTestnet


class TenderlyClient:
    """
    Async client for Tenderly API operations.
    
    Handles simulation, fork management, and transaction execution
    on Tenderly's infrastructure.
    """
    
    def __init__(
        self,
        api_key: str,
        username: str,
        project_slug: str,
        base_url: str = "https://api.tenderly.co/api/v1",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        rate_limit_per_minute: int = 50
    ):
        """
        Initialize Tenderly client.
        
        Args:
            api_key: Tenderly API access key
            username: Tenderly username
            project_slug: Project slug in Tenderly
            base_url: API base URL
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts
            rate_limit_per_minute: API rate limit
        """
        self.api_key = api_key
        self.username = username
        self.project_slug = project_slug
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(rate_limit_per_minute)
        self._last_requests: List[datetime] = []
        
        # Virtual TestNet management  
        self._active_testnets: Dict[str, TenderlyVirtualTestnet] = {}
        # Legacy compatibility
        self._active_forks = self._active_testnets
        
        # Statistics
        self._stats = {
            "simulations_run": 0,
            "testnets_created": 0,
            "testnets_deleted": 0,
            "forks_created": 0,  # Legacy compatibility
            "forks_deleted": 0,  # Legacy compatibility
            "api_errors": 0,
            "rate_limit_hits": 0,
            "total_gas_simulated": 0,
        }
        
        logger.info(f"Initialized TenderlyClient for project {project_slug}")
    
    async def initialize(self) -> None:
        """Initialize HTTP session and validate API access."""
        if self.session:
            return
            
        headers = {
            "X-Access-Key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "YieldArbitrage/1.0"
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        
        # Validate API access
        await self._validate_api_access()
        logger.info("Tenderly API session initialized and validated")
    
    async def close(self) -> None:
        """Close HTTP session and cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
            
        # Cleanup active virtual testnets if needed
        for testnet_id in list(self._active_testnets.keys()):
            try:
                await self.delete_virtual_testnet(testnet_id)
            except Exception as e:
                logger.warning(f"Failed to cleanup virtual testnet {testnet_id}: {e}")
        
        logger.info("Tenderly client closed")
    
    async def simulate_transaction(
        self,
        transaction: TenderlyTransaction,
        network_id: TenderlyNetworkId = TenderlyNetworkId.ETHEREUM,
        block_number: Optional[int] = None,
        fork_id: Optional[str] = None,
        testnet_id: Optional[str] = None,
        save_if_fails: bool = True
    ) -> TenderlySimulationResult:
        """
        Simulate a single transaction.
        
        Args:
            transaction: Transaction to simulate
            network_id: Network to simulate on
            block_number: Block number for simulation
            fork_id: Optional fork ID to simulate on (legacy compatibility)
            testnet_id: Optional Virtual TestNet ID to simulate on
            save_if_fails: Whether to save failed simulations
            
        Returns:
            TenderlySimulationResult with simulation details
        """
        await self._ensure_rate_limit()
        
        if not self.session:
            await self.initialize()
        
        import time
        start_time = time.time()
        
        try:
            # Prepare simulation payload
            payload = {
                "network_id": network_id.value,
                "save": True,
                "save_if_fails": save_if_fails,
                "simulation_type": "full",
                **transaction.to_dict()
            }
            
            if block_number:
                payload["block_number"] = block_number
            
            # Handle both legacy fork_id and new testnet_id
            target_id = testnet_id or fork_id
            if target_id:
                payload["root"] = target_id
            
            # Make API request
            url = f"{self.base_url}/account/{self.username}/project/{self.project_slug}/simulate"
            
            async with self.session.post(url, json=payload) as response:
                simulation_time_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    result = self._parse_simulation_response(data, simulation_time_ms)
                    
                    self._stats["simulations_run"] += 1
                    self._stats["total_gas_simulated"] += result.gas_used
                    
                    return result
                    
                elif response.status == 429:
                    # Rate limit hit
                    self._stats["rate_limit_hits"] += 1
                    raise TenderlyRateLimitError("API rate limit exceeded")
                    
                else:
                    error_text = await response.text()
                    self._stats["api_errors"] += 1
                    raise TenderlyAPIError(f"Simulation failed: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            self._stats["api_errors"] += 1
            raise TenderlyNetworkError(f"Network error during simulation: {e}")
    
    async def simulate_transaction_bundle(
        self,
        transactions: List[TenderlyTransaction],
        network_id: TenderlyNetworkId = TenderlyNetworkId.ETHEREUM,
        block_number: Optional[int] = None,
        fork_id: Optional[str] = None,
        testnet_id: Optional[str] = None
    ) -> List[TenderlySimulationResult]:
        """
        Simulate a bundle of transactions in sequence.
        
        Args:
            transactions: List of transactions to simulate
            network_id: Network to simulate on
            block_number: Block number for simulation
            fork_id: Optional fork ID to simulate on (legacy compatibility)
            testnet_id: Optional Virtual TestNet ID to simulate on
            
        Returns:
            List of TenderlySimulationResult for each transaction
        """
        results = []
        current_target_id = testnet_id or fork_id
        
        # Create a temporary virtual testnet if needed
        if not current_target_id:
            testnet = await self.create_virtual_testnet(network_id, block_number)
            current_target_id = testnet.testnet_id
            
        try:
            for i, transaction in enumerate(transactions):
                logger.debug(f"Simulating transaction {i+1}/{len(transactions)}")
                
                result = await self.simulate_transaction(
                    transaction,
                    network_id,
                    block_number,
                    testnet_id=current_target_id,
                    save_if_fails=False
                )
                
                results.append(result)
                
                # If simulation failed, stop the bundle
                if not result.success:
                    logger.warning(f"Transaction {i+1} failed, stopping bundle simulation")
                    break
                    
        finally:
            # Cleanup temporary testnet
            if not (testnet_id or fork_id) and current_target_id:
                await self.delete_virtual_testnet(current_target_id)
        
        return results
    
    async def create_virtual_testnet(
        self,
        network_id: TenderlyNetworkId = TenderlyNetworkId.ETHEREUM,
        block_number: Optional[int] = None,
        slug: Optional[str] = None,
        display_name: Optional[str] = None,
        sync_state_enabled: bool = False,
        explorer_enabled: bool = False
    ) -> TenderlyVirtualTestnet:
        """
        Create a new Tenderly Virtual TestNet.
        
        Args:
            network_id: Network to fork from
            block_number: Block number to fork from
            slug: Optional unique identifier for the testnet
            display_name: Optional display name for the testnet
            sync_state_enabled: Whether to keep testnet synced with parent network
            explorer_enabled: Whether to enable public explorer
            
        Returns:
            TenderlyVirtualTestnet object
        """
        await self._ensure_rate_limit()
        
        if not self.session:
            await self.initialize()
        
        # Generate unique slug if not provided
        if not slug:
            import time
            slug = f"testnet-{int(time.time())}"
        
        # Generate display name if not provided
        if not display_name:
            display_name = f"TestNet {slug}"
        
        # Generate custom chain ID (recommended practice: 7357 + original network ID)
        original_chain_id = int(network_id.value)
        custom_chain_id = 73570000 + original_chain_id
        
        payload = {
            "slug": slug,
            "display_name": display_name,
            "fork_config": {
                "network_id": original_chain_id,
                "block_number": "latest" if block_number is None else str(block_number)
            },
            "virtual_network_config": {
                "chain_config": {
                    "chain_id": custom_chain_id
                }
            },
            "sync_state_config": {
                "enabled": sync_state_enabled,
                "commitment_level": "latest"
            },
            "explorer_page_config": {
                "enabled": explorer_enabled,
                "verification_visibility": "bytecode"
            }
        }
        
        url = f"{self.base_url}/account/{self.username}/project/{self.project_slug}/vnets"
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status in [200, 201]:
                    data = await response.json()
                    
                    # Extract RPC URLs from response
                    rpcs = data.get("rpcs", [])
                    admin_rpc_url = None
                    public_rpc_url = None
                    
                    for rpc in rpcs:
                        if rpc.get("name") == "Admin RPC":
                            admin_rpc_url = rpc.get("url")
                        elif rpc.get("name") == "Public RPC":
                            public_rpc_url = rpc.get("url")
                    
                    # Create TenderlyVirtualTestnet object
                    testnet = TenderlyVirtualTestnet(
                        testnet_id=data["id"],
                        slug=slug,
                        display_name=display_name,
                        network_id=network_id.value,
                        block_number=block_number or data.get("latest_block_number", 0),
                        created_at=datetime.utcnow(),
                        chain_id=custom_chain_id,
                        admin_rpc_url=admin_rpc_url,
                        public_rpc_url=public_rpc_url,
                        fork_config=payload["fork_config"],
                        sync_state_enabled=sync_state_enabled,
                        explorer_enabled=explorer_enabled,
                        explorer_url=data.get("explorer_url")
                    )
                    
                    self._active_testnets[testnet.testnet_id] = testnet
                    self._stats["testnets_created"] += 1
                    self._stats["forks_created"] += 1  # Legacy compatibility
                    
                    logger.info(f"Created Tenderly Virtual TestNet: {testnet.testnet_id} (slug: {slug})")
                    return testnet
                    
                else:
                    error_text = await response.text()
                    raise TenderlyAPIError(f"Virtual TestNet creation failed: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            raise TenderlyNetworkError(f"Network error creating Virtual TestNet: {e}")
    
    async def delete_virtual_testnet(self, testnet_id: str) -> bool:
        """
        Delete a Tenderly Virtual TestNet.
        
        Args:
            testnet_id: Virtual TestNet ID to delete
            
        Returns:
            True if successful
        """
        await self._ensure_rate_limit()
        
        if not self.session:
            await self.initialize()
        
        url = f"{self.base_url}/account/{self.username}/project/{self.project_slug}/vnets/{testnet_id}"
        
        try:
            async with self.session.delete(url) as response:
                if response.status in [200, 204]:
                    if testnet_id in self._active_testnets:
                        del self._active_testnets[testnet_id]
                    
                    self._stats["testnets_deleted"] += 1
                    self._stats["forks_deleted"] += 1  # Legacy compatibility
                    logger.info(f"Deleted Tenderly Virtual TestNet: {testnet_id}")
                    return True
                    
                else:
                    logger.warning(f"Failed to delete Virtual TestNet {testnet_id}: {response.status}")
                    return False
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error deleting Virtual TestNet {testnet_id}: {e}")
            return False
    
    # Legacy compatibility methods
    async def create_fork(
        self,
        network_id: TenderlyNetworkId = TenderlyNetworkId.ETHEREUM,
        block_number: Optional[int] = None,
        alias: Optional[str] = None,
        description: Optional[str] = None
    ) -> TenderlyVirtualTestnet:
        """
        Legacy compatibility method for creating forks.
        Creates a Virtual TestNet instead.
        
        Args:
            network_id: Network to fork from
            block_number: Block number to fork from
            alias: Used as slug for the testnet
            description: Used as display name for the testnet
            
        Returns:
            TenderlyVirtualTestnet object (compatible with TenderlyFork)
        """
        return await self.create_virtual_testnet(
            network_id=network_id,
            block_number=block_number,
            slug=alias,
            display_name=description or alias
        )
    
    async def delete_fork(self, fork_id: str) -> bool:
        """
        Legacy compatibility method for deleting forks.
        Deletes a Virtual TestNet instead.
        
        Args:
            fork_id: TestNet ID to delete (same as testnet_id)
            
        Returns:
            True if successful
        """
        return await self.delete_virtual_testnet(fork_id)
    
    async def get_virtual_testnet_info(self, testnet_id: str) -> Optional[TenderlyVirtualTestnet]:
        """Get information about a Virtual TestNet."""
        if testnet_id in self._active_testnets:
            return self._active_testnets[testnet_id]
        
        # Could implement API call to fetch testnet info
        return None
    
    async def get_fork_info(self, fork_id: str) -> Optional[TenderlyVirtualTestnet]:
        """Legacy compatibility method for getting fork info."""
        return await self.get_virtual_testnet_info(fork_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = self._stats.copy()
        stats["active_testnets"] = len(self._active_testnets)
        stats["active_forks"] = len(self._active_testnets)  # Legacy compatibility
        stats["session_active"] = self.session is not None
        return stats
    
    async def _validate_api_access(self) -> None:
        """Validate API access by making a test request."""
        url = f"{self.base_url}/account/{self.username}/project/{self.project_slug}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 401:
                    raise TenderlyAuthError("Invalid API key or insufficient permissions")
                elif response.status == 404:
                    raise TenderlyAPIError(f"Project {self.project_slug} not found")
                elif response.status != 200:
                    raise TenderlyAPIError(f"API validation failed: {response.status}")
                    
        except aiohttp.ClientError as e:
            raise TenderlyNetworkError(f"Network error during API validation: {e}")
    
    async def _ensure_rate_limit(self) -> None:
        """Ensure we don't exceed rate limits."""
        now = datetime.utcnow()
        
        # Remove requests older than 1 minute
        cutoff = now - timedelta(minutes=1)
        self._last_requests = [req_time for req_time in self._last_requests if req_time > cutoff]
        
        # Check if we're at the limit
        if len(self._last_requests) >= 50:  # Conservative limit
            # Wait until the oldest request is more than 1 minute old
            oldest_request = min(self._last_requests)
            wait_time = 60 - (now - oldest_request).total_seconds()
            
            if wait_time > 0:
                logger.info(f"Rate limit approaching, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self._last_requests.append(now)
    
    def _parse_simulation_response(
        self, 
        data: Dict[str, Any], 
        simulation_time_ms: float
    ) -> TenderlySimulationResult:
        """Parse Tenderly API response into TenderlySimulationResult."""
        transaction = data.get("transaction", {})
        
        # Determine success
        success = transaction.get("status", False)
        if isinstance(success, str):
            success = success.lower() == "true"
        
        # Extract gas information
        gas_used_raw = transaction.get("gas_used")
        if gas_used_raw:
            if isinstance(gas_used_raw, str) and gas_used_raw.startswith("0x"):
                gas_used = int(gas_used_raw, 16)
            elif isinstance(gas_used_raw, str):
                gas_used = int(gas_used_raw)
            else:
                gas_used = int(gas_used_raw)
        else:
            gas_used = 0
        
        # Extract error information
        error_message = transaction.get("error_message")
        revert_reason = None
        
        # Try to extract revert reason from call trace
        if not success and "call_trace" in transaction:
            call_trace = transaction["call_trace"]
            if "output" in call_trace and call_trace["output"]:
                # Try to decode revert reason
                revert_reason = self._decode_revert_reason(call_trace["output"])
        
        return TenderlySimulationResult(
            success=success,
            gas_used=gas_used,
            transaction_hash=transaction.get("hash"),
            block_number=transaction.get("block_number"),
            logs=transaction.get("logs", []),
            trace=transaction.get("call_trace"),
            state_changes=data.get("state_changes"),
            error_message=error_message,
            revert_reason=revert_reason,
            simulation_time_ms=simulation_time_ms
        )
    
    def _decode_revert_reason(self, output: str) -> Optional[str]:
        """Attempt to decode revert reason from transaction output."""
        if not output or output == "0x":
            return None
            
        try:
            # Standard revert reason encoding
            if output.startswith("0x08c379a0"):  # Error(string) selector
                # Skip selector (4 bytes) and offset (32 bytes)
                data = output[2 + 8 + 64:]
                
                # Get string length
                if len(data) >= 64:
                    length = int(data[:64], 16)
                    
                    # Extract string data
                    string_data = data[64:64 + length * 2]
                    if len(string_data) == length * 2:
                        return bytes.fromhex(string_data).decode('utf-8').rstrip('\x00')
                        
        except Exception:
            pass
            
        return f"Raw output: {output[:100]}..."


# Custom exceptions
class TenderlyError(Exception):
    """Base exception for Tenderly-related errors."""
    pass


class TenderlyAPIError(TenderlyError):
    """API-related errors."""
    pass


class TenderlyAuthError(TenderlyError):
    """Authentication-related errors."""
    pass


class TenderlyNetworkError(TenderlyError):
    """Network-related errors."""
    pass


class TenderlyRateLimitError(TenderlyError):
    """Rate limit exceeded."""
    pass