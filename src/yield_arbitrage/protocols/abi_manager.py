"""ABI Manager for protocol contract interactions."""
import logging
from typing import Dict, List, Optional

from .abis.uniswap_v3 import (
    UNISWAP_V3_FACTORY_ABI,
    UNISWAP_V3_QUOTER_ABI,
    UNISWAP_V3_POOL_ABI,
    ERC20_ABI
)

logger = logging.getLogger(__name__)


class ABIManager:
    """Manages contract ABIs for protocol interactions."""
    
    def __init__(self):
        """Initialize ABI manager with protocol ABIs."""
        self._abis = {
            "uniswap_v3": {
                "factory": UNISWAP_V3_FACTORY_ABI,
                "quoter": UNISWAP_V3_QUOTER_ABI,
                "pool": UNISWAP_V3_POOL_ABI,
            },
            "erc20": ERC20_ABI
        }
    
    def get_abi(self, protocol: str, contract_type: str = None) -> Optional[List[Dict]]:
        """
        Get ABI for a specific protocol and contract type.
        
        Args:
            protocol: Protocol name (e.g., 'uniswap_v3', 'erc20')
            contract_type: Contract type within protocol (e.g., 'factory', 'quoter', 'pool')
            
        Returns:
            ABI dictionary list or None if not found
        """
        try:
            protocol_abis = self._abis.get(protocol.lower())
            
            if not protocol_abis:
                logger.warning(f"Protocol '{protocol}' not found in ABI manager")
                return None
            
            # For simple protocols like ERC20, return directly
            if isinstance(protocol_abis, list):
                return protocol_abis
            
            # For complex protocols with multiple contract types
            if contract_type:
                contract_abi = protocol_abis.get(contract_type.lower())
                if not contract_abi:
                    logger.warning(f"Contract type '{contract_type}' not found for protocol '{protocol}'")
                    return None
                return contract_abi
            
            logger.warning(f"Contract type required for protocol '{protocol}'")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving ABI for {protocol}.{contract_type}: {e}")
            return None
    
    def get_uniswap_v3_factory_abi(self) -> List[Dict]:
        """Get Uniswap V3 Factory ABI."""
        return self.get_abi("uniswap_v3", "factory")
    
    def get_uniswap_v3_quoter_abi(self) -> List[Dict]:
        """Get Uniswap V3 Quoter ABI."""
        return self.get_abi("uniswap_v3", "quoter")
    
    def get_uniswap_v3_pool_abi(self) -> List[Dict]:
        """Get Uniswap V3 Pool ABI."""
        return self.get_abi("uniswap_v3", "pool")
    
    def get_erc20_abi(self) -> List[Dict]:
        """Get ERC20 ABI."""
        return self.get_abi("erc20")
    
    def list_protocols(self) -> List[str]:
        """List available protocols."""
        return list(self._abis.keys())
    
    def list_contract_types(self, protocol: str) -> List[str]:
        """List available contract types for a protocol."""
        protocol_abis = self._abis.get(protocol.lower())
        
        if not protocol_abis:
            return []
        
        if isinstance(protocol_abis, list):
            return ["main"]  # Simple protocol with single ABI
        
        return list(protocol_abis.keys())
    
    def validate_abi(self, abi: List[Dict]) -> bool:
        """
        Validate that an ABI is properly formatted.
        
        Args:
            abi: ABI to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(abi, list):
                return False
            
            for item in abi:
                if not isinstance(item, dict):
                    return False
                
                # Check required fields
                if "type" not in item:
                    return False
                
                # Validate function/event structure
                if item["type"] in ["function", "event"]:
                    if "name" not in item:
                        return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating ABI: {e}")
            return False


# Global ABI manager instance
abi_manager = ABIManager()