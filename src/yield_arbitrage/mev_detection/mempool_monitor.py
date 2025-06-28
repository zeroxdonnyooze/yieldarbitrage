"""
Mempool Monitor for MEV Opportunity Detection.

This module provides real-time monitoring of blockchain mempools to detect
pending transactions that may create MEV opportunities.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum

import aiohttp
# import websockets  # Optional - for WebSocket connections
# from web3 import AsyncWeb3  # Optional - for Web3 connections
# from web3.types import TxParams

logger = logging.getLogger(__name__)


class TransactionEventType(str, Enum):
    """Types of transaction events."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    DROPPED = "dropped"


@dataclass
class TransactionEvent:
    """Event representing a transaction in the mempool."""
    event_type: TransactionEventType
    transaction_hash: str
    transaction_data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    
    # Additional analysis
    gas_price_gwei: Optional[float] = None
    estimated_confirmation_time: Optional[int] = None
    mev_potential_score: Optional[float] = None


@dataclass
class MempoolConfig:
    """Configuration for mempool monitoring."""
    
    # Network configuration
    chain_id: int = 1  # Ethereum mainnet
    rpc_urls: List[str] = field(default_factory=lambda: ["wss://eth-mainnet.g.alchemy.com/v2/"])
    mempool_streams: List[str] = field(default_factory=list)
    
    # Filtering options
    min_gas_price_gwei: float = 10.0
    max_gas_price_gwei: float = 1000.0
    min_value_eth: float = 0.01
    watch_addresses: Set[str] = field(default_factory=set)
    watch_contract_methods: Set[str] = field(default_factory=set)
    
    # Performance settings
    max_pending_transactions: int = 10000
    transaction_ttl_seconds: int = 300  # 5 minutes
    batch_size: int = 100
    update_interval_ms: int = 100
    
    # Analysis settings
    enable_gas_analysis: bool = True
    enable_mev_scoring: bool = True
    enable_duplicate_detection: bool = True


class MempoolMonitor:
    """
    Real-time mempool monitor for MEV opportunity detection.
    
    Monitors blockchain mempools through WebSocket connections and analyzes
    pending transactions for MEV opportunities.
    """
    
    def __init__(self, config: MempoolConfig):
        """Initialize mempool monitor with configuration."""
        self.config = config
        self.is_running = False
        
        # Transaction tracking
        self.pending_transactions: Dict[str, TransactionEvent] = {}
        self.processed_hashes: Set[str] = set()
        
        # Event handlers
        self.event_handlers: Dict[TransactionEventType, List[Callable]] = {
            TransactionEventType.PENDING: [],
            TransactionEventType.CONFIRMED: [],
            TransactionEventType.FAILED: [],
            TransactionEventType.DROPPED: []
        }
        
        # Network connections
        self.web3_connections: List[Any] = []  # Would be AsyncWeb3 instances
        self.websocket_connections: List[Any] = []  # Would be WebSocket connections
        
        # Statistics
        self.stats = {
            "transactions_processed": 0,
            "mev_opportunities_detected": 0,
            "gas_price_average": 0.0,
            "last_block_processed": 0,
            "uptime_start": time.time()
        }
    
    async def start(self):
        """Start monitoring mempool transactions."""
        if self.is_running:
            logger.warning("Mempool monitor already running")
            return
        
        logger.info(f"Starting mempool monitor for chain {self.config.chain_id}")
        self.is_running = True
        
        # Initialize connections
        await self._initialize_connections()
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_pending_transactions()),
            asyncio.create_task(self._monitor_confirmed_transactions()),
            asyncio.create_task(self._cleanup_expired_transactions()),
            asyncio.create_task(self._update_statistics())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Mempool monitor error: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop monitoring and cleanup connections."""
        logger.info("Stopping mempool monitor")
        self.is_running = False
        
        # Close WebSocket connections
        for ws in self.websocket_connections:
            await ws.close()
        self.websocket_connections.clear()
        
        # Close Web3 connections
        for w3 in self.web3_connections:
            # Note: AsyncWeb3 doesn't have explicit close method
            pass
        self.web3_connections.clear()
    
    def add_event_handler(self, event_type: TransactionEventType, handler: Callable):
        """Add event handler for transaction events."""
        self.event_handlers[event_type].append(handler)
        logger.debug(f"Added handler for {event_type} events")
    
    def remove_event_handler(self, event_type: TransactionEventType, handler: Callable):
        """Remove event handler."""
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
    
    async def _initialize_connections(self):
        """Initialize Web3 and WebSocket connections."""
        # In a real implementation, this would initialize actual Web3 connections
        # For testing, we'll create mock connections
        logger.info("Initializing mock connections for testing")
        
        for rpc_url in self.config.rpc_urls:
            try:
                # Mock connection for testing
                mock_connection = {
                    "url": rpc_url,
                    "type": "mock",
                    "connected": True
                }
                self.web3_connections.append(mock_connection)
                logger.info(f"Mock connection created for {rpc_url}")
                
            except Exception as e:
                logger.error(f"Failed to create mock connection for {rpc_url}: {e}")
        
        if not self.web3_connections:
            logger.warning("No connections established - using offline mode")
    
    async def _monitor_pending_transactions(self):
        """Monitor pending transactions from mempool."""
        logger.info("Starting pending transaction monitoring (mock mode)")
        
        while self.is_running:
            try:
                # In testing mode, we'll just wait
                # Real implementation would monitor actual mempool
                await asyncio.sleep(self.config.update_interval_ms / 1000)
                
            except Exception as e:
                logger.error(f"Pending transaction monitoring error: {e}")
                await asyncio.sleep(1)
    
    async def _process_pending_transaction(self, w3: Any, tx_hash: str):
        """Process a single pending transaction."""
        if tx_hash in self.processed_hashes:
            return
        
        try:
            # Get transaction details
            tx = await w3.eth.get_transaction(tx_hash)
            if not tx:
                return
            
            # Convert transaction to dict
            tx_data = dict(tx)
            
            # Apply filters
            if not self._passes_filters(tx_data):
                return
            
            # Create transaction event
            event = TransactionEvent(
                event_type=TransactionEventType.PENDING,
                transaction_hash=tx_hash,
                transaction_data=tx_data,
                gas_price_gwei=float(tx_data.get('gasPrice', 0)) / 1e9
            )
            
            # Analyze transaction for MEV potential
            if self.config.enable_mev_scoring:
                event.mev_potential_score = await self._calculate_mev_potential(tx_data)
            
            # Store and process
            self.pending_transactions[tx_hash] = event
            self.processed_hashes.add(tx_hash)
            self.stats["transactions_processed"] += 1
            
            # Trigger event handlers
            await self._trigger_event_handlers(TransactionEventType.PENDING, event)
            
            logger.debug(f"Processed pending transaction: {tx_hash[:10]}...")
            
        except Exception as e:
            logger.error(f"Error processing pending transaction {tx_hash}: {e}")
    
    async def _monitor_confirmed_transactions(self):
        """Monitor confirmed transactions to update pending transaction status."""
        logger.info("Starting confirmed transaction monitoring")
        
        while self.is_running:
            try:
                for w3 in self.web3_connections:
                    try:
                        # Get latest block
                        latest_block = await w3.eth.get_block('latest', full_transactions=True)
                        
                        if latest_block.number > self.stats["last_block_processed"]:
                            await self._process_block_transactions(latest_block)
                            self.stats["last_block_processed"] = latest_block.number
                        
                    except Exception as e:
                        logger.error(f"Error monitoring confirmed transactions: {e}")
                        continue
                
                await asyncio.sleep(1)  # Check for new blocks every second
                
            except Exception as e:
                logger.error(f"Confirmed transaction monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _process_block_transactions(self, block):
        """Process transactions in a confirmed block."""
        for tx in block.transactions:
            tx_hash = tx.hash.hex()
            
            if tx_hash in self.pending_transactions:
                # Update pending transaction to confirmed
                pending_event = self.pending_transactions[tx_hash]
                
                confirmed_event = TransactionEvent(
                    event_type=TransactionEventType.CONFIRMED,
                    transaction_hash=tx_hash,
                    transaction_data=dict(tx),
                    gas_price_gwei=pending_event.gas_price_gwei,
                    mev_potential_score=pending_event.mev_potential_score
                )
                
                # Remove from pending
                del self.pending_transactions[tx_hash]
                
                # Trigger confirmed event handlers
                await self._trigger_event_handlers(TransactionEventType.CONFIRMED, confirmed_event)
    
    async def _cleanup_expired_transactions(self):
        """Remove expired pending transactions."""
        while self.is_running:
            try:
                current_time = time.time()
                expired_hashes = []
                
                for tx_hash, event in self.pending_transactions.items():
                    if current_time - event.timestamp > self.config.transaction_ttl_seconds:
                        expired_hashes.append(tx_hash)
                
                # Remove expired transactions
                for tx_hash in expired_hashes:
                    event = self.pending_transactions[tx_hash]
                    del self.pending_transactions[tx_hash]
                    
                    # Create dropped event
                    dropped_event = TransactionEvent(
                        event_type=TransactionEventType.DROPPED,
                        transaction_hash=tx_hash,
                        transaction_data=event.transaction_data
                    )
                    
                    await self._trigger_event_handlers(TransactionEventType.DROPPED, dropped_event)
                
                if expired_hashes:
                    logger.debug(f"Cleaned up {len(expired_hashes)} expired transactions")
                
                # Cleanup processed hashes periodically
                if len(self.processed_hashes) > self.config.max_pending_transactions * 2:
                    self.processed_hashes.clear()
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _update_statistics(self):
        """Update monitoring statistics."""
        while self.is_running:
            try:
                # Calculate average gas price from pending transactions
                if self.pending_transactions:
                    gas_prices = [event.gas_price_gwei for event in self.pending_transactions.values() 
                                if event.gas_price_gwei is not None]
                    if gas_prices:
                        self.stats["gas_price_average"] = sum(gas_prices) / len(gas_prices)
                
                await asyncio.sleep(10)  # Update stats every 10 seconds
                
            except Exception as e:
                logger.error(f"Statistics update error: {e}")
                await asyncio.sleep(10)
    
    def _passes_filters(self, tx_data: Dict[str, Any]) -> bool:
        """Check if transaction passes configured filters."""
        # Gas price filter
        gas_price_gwei = float(tx_data.get('gasPrice', 0)) / 1e9
        if gas_price_gwei < self.config.min_gas_price_gwei:
            return False
        if gas_price_gwei > self.config.max_gas_price_gwei:
            return False
        
        # Value filter
        value_eth = float(tx_data.get('value', 0)) / 1e18
        if value_eth < self.config.min_value_eth:
            return False
        
        # Address filter
        if self.config.watch_addresses:
            to_address = tx_data.get('to', '').lower()
            from_address = tx_data.get('from', '').lower()
            if not (to_address in self.config.watch_addresses or from_address in self.config.watch_addresses):
                return False
        
        return True
    
    async def _calculate_mev_potential(self, tx_data: Dict[str, Any]) -> float:
        """Calculate MEV potential score for a transaction."""
        score = 0.0
        
        try:
            # High gas price indicates urgency/MEV opportunity
            gas_price_gwei = float(tx_data.get('gasPrice', 0)) / 1e9
            if gas_price_gwei > 50:
                score += 0.3
            elif gas_price_gwei > 20:
                score += 0.1
            
            # Large value transactions have higher MEV potential
            value_eth = float(tx_data.get('value', 0)) / 1e18
            if value_eth > 100:
                score += 0.4
            elif value_eth > 10:
                score += 0.2
            
            # Contract interactions have higher MEV potential
            if tx_data.get('to') and tx_data.get('input', '0x') != '0x':
                score += 0.3
            
            # DEX interactions (simplified detection)
            data = tx_data.get('input', '')
            if data.startswith('0xa9059cbb'):  # ERC20 transfer
                score += 0.2
            elif data.startswith('0x38ed1739'):  # Uniswap swapExactTokensForTokens
                score += 0.5
            elif data.startswith('0x7ff36ab5'):  # Uniswap swapExactETHForTokens
                score += 0.5
            
        except Exception as e:
            logger.error(f"Error calculating MEV potential: {e}")
        
        return min(1.0, score)
    
    async def _trigger_event_handlers(self, event_type: TransactionEventType, event: TransactionEvent):
        """Trigger registered event handlers."""
        for handler in self.event_handlers[event_type]:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def get_pending_count(self) -> int:
        """Get count of pending transactions."""
        return len(self.pending_transactions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            **self.stats,
            "pending_transactions": len(self.pending_transactions),
            "uptime_seconds": time.time() - self.stats["uptime_start"],
            "connections_active": len(self.web3_connections)
        }
    
    def get_high_mev_transactions(self, min_score: float = 0.5) -> List[TransactionEvent]:
        """Get pending transactions with high MEV potential."""
        return [
            event for event in self.pending_transactions.values()
            if event.mev_potential_score and event.mev_potential_score >= min_score
        ]


# Convenience functions

async def create_mempool_monitor(
    chain_id: int = 1,
    rpc_urls: Optional[List[str]] = None,
    **config_kwargs
) -> MempoolMonitor:
    """Create and configure a mempool monitor."""
    
    if rpc_urls is None:
        # Default RPC URLs by chain
        default_rpcs = {
            1: ["wss://eth-mainnet.g.alchemy.com/v2/"],  # Ethereum
            56: ["wss://bsc-ws-node.nariox.org:443"],    # BSC
            137: ["wss://polygon-mainnet.g.alchemy.com/v2/"],  # Polygon
            42161: ["wss://arb1.arbitrum.io/ws"],        # Arbitrum
            10: ["wss://opt-mainnet.g.alchemy.com/v2/"]  # Optimism
        }
        rpc_urls = default_rpcs.get(chain_id, [])
    
    config = MempoolConfig(
        chain_id=chain_id,
        rpc_urls=rpc_urls,
        **config_kwargs
    )
    
    return MempoolMonitor(config)


async def start_basic_mempool_monitoring(
    chain_id: int = 1,
    event_handler: Optional[Callable] = None
) -> MempoolMonitor:
    """Start basic mempool monitoring with optional event handler."""
    
    monitor = await create_mempool_monitor(chain_id)
    
    if event_handler:
        monitor.add_event_handler(TransactionEventType.PENDING, event_handler)
    
    # Start monitoring in background
    asyncio.create_task(monitor.start())
    
    return monitor