"""Protocol adapter registry for managing multiple protocol integrations."""
import logging
from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass

from .base_adapter import ProtocolAdapter, ProtocolError

logger = logging.getLogger(__name__)


@dataclass
class AdapterInfo:
    """Information about a registered protocol adapter."""
    adapter_class: Type[ProtocolAdapter]
    protocol_name: str
    supported_chains: List[str]
    description: str
    is_enabled: bool = True


class ProtocolAdapterRegistry:
    """
    Registry for managing protocol adapters across multiple chains.
    
    This class provides a centralized way to register, initialize, and manage
    protocol adapters for different blockchains and DeFi protocols.
    """
    
    def __init__(self):
        """Initialize the adapter registry."""
        self._adapters: Dict[str, Dict[str, ProtocolAdapter]] = {}  # protocol -> chain -> adapter
        self._adapter_info: Dict[str, AdapterInfo] = {}  # protocol -> info
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_adapter(
        self,
        adapter_class: Type[ProtocolAdapter],
        protocol_name: str,
        supported_chains: List[str],
        description: str = "",
        enabled: bool = True
    ) -> bool:
        """
        Register a protocol adapter class.
        
        Args:
            adapter_class: The adapter class to register
            protocol_name: Name of the protocol
            supported_chains: List of supported chain names
            description: Description of the protocol
            enabled: Whether the adapter is enabled
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if not issubclass(adapter_class, ProtocolAdapter):
                raise ValueError(f"Adapter class must inherit from ProtocolAdapter")
            
            protocol_key = protocol_name.lower()
            
            if protocol_key in self._adapter_info:
                self.logger.warning(f"Protocol {protocol_name} already registered, overwriting")
            
            self._adapter_info[protocol_key] = AdapterInfo(
                adapter_class=adapter_class,
                protocol_name=protocol_name,
                supported_chains=[chain.lower() for chain in supported_chains],
                description=description,
                is_enabled=enabled
            )
            
            # Initialize empty adapter storage for this protocol
            if protocol_key not in self._adapters:
                self._adapters[protocol_key] = {}
            
            self.logger.info(
                f"Registered {protocol_name} adapter for chains: {', '.join(supported_chains)}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register adapter for {protocol_name}: {e}")
            return False
    
    async def initialize_adapter(
        self,
        protocol_name: str,
        chain_name: str,
        provider: Any,
        **kwargs
    ) -> Optional[ProtocolAdapter]:
        """
        Initialize a specific protocol adapter for a chain.
        
        Args:
            protocol_name: Name of the protocol
            chain_name: Name of the chain
            provider: Blockchain provider instance
            **kwargs: Additional arguments for adapter initialization
            
        Returns:
            Initialized adapter instance or None if failed
        """
        try:
            protocol_key = protocol_name.lower()
            chain_key = chain_name.lower()
            
            # Check if protocol is registered
            if protocol_key not in self._adapter_info:
                self.logger.error(f"Protocol {protocol_name} not registered")
                return None
            
            adapter_info = self._adapter_info[protocol_key]
            
            # Check if protocol is enabled
            if not adapter_info.is_enabled:
                self.logger.warning(f"Protocol {protocol_name} is disabled")
                return None
            
            # Check if chain is supported
            if chain_key not in adapter_info.supported_chains:
                self.logger.error(
                    f"Chain {chain_name} not supported by {protocol_name}. "
                    f"Supported chains: {', '.join(adapter_info.supported_chains)}"
                )
                return None
            
            # Check if adapter already exists
            if chain_key in self._adapters[protocol_key]:
                existing_adapter = self._adapters[protocol_key][chain_key]
                if existing_adapter.is_initialized:
                    self.logger.info(f"Adapter {protocol_name}@{chain_name} already initialized")
                    return existing_adapter
            
            # Create and initialize new adapter
            adapter = adapter_info.adapter_class(chain_name, provider, **kwargs)
            
            success = await adapter.initialize()
            if not success:
                self.logger.error(f"Failed to initialize {protocol_name} adapter for {chain_name}")
                return None
            
            # Store initialized adapter
            self._adapters[protocol_key][chain_key] = adapter
            
            self.logger.info(f"Successfully initialized {protocol_name} adapter for {chain_name}")
            return adapter
            
        except Exception as e:
            self.logger.error(f"Error initializing {protocol_name} adapter for {chain_name}: {e}")
            return None
    
    def get_adapter(self, protocol_name: str, chain_name: str) -> Optional[ProtocolAdapter]:
        """
        Get an initialized adapter for a specific protocol and chain.
        
        Args:
            protocol_name: Name of the protocol
            chain_name: Name of the chain
            
        Returns:
            Adapter instance or None if not found
        """
        protocol_key = protocol_name.lower()
        chain_key = chain_name.lower()
        
        if protocol_key in self._adapters:
            return self._adapters[protocol_key].get(chain_key)
        
        return None
    
    def list_protocols(self) -> List[str]:
        """
        Get list of registered protocols.
        
        Returns:
            List of protocol names
        """
        return [info.protocol_name for info in self._adapter_info.values()]
    
    def list_chains_for_protocol(self, protocol_name: str) -> List[str]:
        """
        Get list of supported chains for a protocol.
        
        Args:
            protocol_name: Name of the protocol
            
        Returns:
            List of supported chain names
        """
        protocol_key = protocol_name.lower()
        
        if protocol_key in self._adapter_info:
            return self._adapter_info[protocol_key].supported_chains.copy()
        
        return []
    
    def list_initialized_adapters(self) -> List[tuple[str, str]]:
        """
        Get list of initialized adapters.
        
        Returns:
            List of (protocol_name, chain_name) tuples
        """
        initialized = []
        
        for protocol_key, chain_adapters in self._adapters.items():
            protocol_name = self._adapter_info[protocol_key].protocol_name
            
            for chain_key, adapter in chain_adapters.items():
                if adapter.is_initialized:
                    initialized.append((protocol_name, chain_key))
        
        return initialized
    
    def get_adapter_stats(self) -> Dict[str, Any]:
        """
        Get statistics about registered and initialized adapters.
        
        Returns:
            Dictionary with adapter statistics
        """
        stats = {
            "registered_protocols": len(self._adapter_info),
            "enabled_protocols": sum(1 for info in self._adapter_info.values() if info.is_enabled),
            "initialized_adapters": len(self.list_initialized_adapters()),
            "protocols": {}
        }
        
        for protocol_key, info in self._adapter_info.items():
            protocol_stats = {
                "protocol_name": info.protocol_name,
                "description": info.description,
                "is_enabled": info.is_enabled,
                "supported_chains": info.supported_chains,
                "initialized_chains": []
            }
            
            if protocol_key in self._adapters:
                for chain_key, adapter in self._adapters[protocol_key].items():
                    if adapter.is_initialized:
                        protocol_stats["initialized_chains"].append(chain_key)
                        
                        # Get adapter-specific stats
                        adapter_stats = adapter.get_discovery_stats()
                        protocol_stats[f"{chain_key}_stats"] = adapter_stats
            
            stats["protocols"][info.protocol_name] = protocol_stats
        
        return stats
    
    def enable_protocol(self, protocol_name: str) -> bool:
        """
        Enable a protocol.
        
        Args:
            protocol_name: Name of the protocol
            
        Returns:
            True if successful, False otherwise
        """
        protocol_key = protocol_name.lower()
        
        if protocol_key not in self._adapter_info:
            self.logger.error(f"Protocol {protocol_name} not registered")
            return False
        
        self._adapter_info[protocol_key].is_enabled = True
        self.logger.info(f"Enabled protocol {protocol_name}")
        return True
    
    def disable_protocol(self, protocol_name: str) -> bool:
        """
        Disable a protocol.
        
        Args:
            protocol_name: Name of the protocol
            
        Returns:
            True if successful, False otherwise
        """
        protocol_key = protocol_name.lower()
        
        if protocol_key not in self._adapter_info:
            self.logger.error(f"Protocol {protocol_name} not registered")
            return False
        
        self._adapter_info[protocol_key].is_enabled = False
        self.logger.info(f"Disabled protocol {protocol_name}")
        return True
    
    async def shutdown_all(self):
        """Shutdown all initialized adapters."""
        self.logger.info("Shutting down all protocol adapters...")
        
        for protocol_key, chain_adapters in self._adapters.items():
            for chain_key, adapter in chain_adapters.items():
                try:
                    # Call adapter shutdown if it exists
                    if hasattr(adapter, 'shutdown'):
                        await adapter.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down {protocol_key}@{chain_key}: {e}")
        
        # Clear all adapters
        self._adapters.clear()
        self.logger.info("All protocol adapters shut down")


# Global registry instance
protocol_registry = ProtocolAdapterRegistry()