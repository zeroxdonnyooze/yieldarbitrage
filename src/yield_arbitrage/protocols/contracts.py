"""Contract addresses for different protocols and chains."""
from typing import Dict, Optional

# Uniswap V3 contract addresses by chain
UNISWAP_V3_CONTRACTS = {
    "ethereum": {
        "factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "quoter": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
        "quoter_v2": "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        "nonfungible_position_manager": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
        "router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        "router_v2": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"
    },
    "arbitrum": {
        "factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "quoter": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
        "quoter_v2": "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        "nonfungible_position_manager": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
        "router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        "router_v2": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"
    },
    "base": {
        "factory": "0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
        "quoter": "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a",
        "quoter_v2": "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a",
        "nonfungible_position_manager": "0x03a520b32C04BF3bEEf7BF5754C4Fef0c2DB7EA2",
        "router": "0x2626664c2603336E57B271c5C0b26F421741e481",
        "router_v2": "0x2626664c2603336E57B271c5C0b26F421741e481"
    }
}

# Standard fee tiers for Uniswap V3
UNISWAP_V3_FEE_TIERS = [
    100,    # 0.01%
    500,    # 0.05%
    3000,   # 0.3%
    10000   # 1%
]

# Well-known token addresses by chain
WELL_KNOWN_TOKENS = {
    "ethereum": {
        "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "USDC": "0xA0b86a33E6441b9435B654f6D26Cc98b6e1D0a3A",
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    },
    "arbitrum": {
        "WETH": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
        "USDC": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        "USDC_E": "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
        "USDT": "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",
        "DAI": "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1",
        "WBTC": "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f",
    },
    "base": {
        "WETH": "0x4200000000000000000000000000000000000006",
        "USDC": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "DAI": "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb",
        "CBETH": "0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22",
    }
}


def get_uniswap_v3_contract(chain_name: str, contract_type: str) -> Optional[str]:
    """Get Uniswap V3 contract address for a specific chain and contract type."""
    chain_contracts = UNISWAP_V3_CONTRACTS.get(chain_name.lower())
    if not chain_contracts:
        return None
    
    return chain_contracts.get(contract_type.lower())


def get_well_known_tokens(chain_name: str) -> Dict[str, str]:
    """Get well-known token addresses for a specific chain."""
    return WELL_KNOWN_TOKENS.get(chain_name.lower(), {})


def is_supported_chain(chain_name: str) -> bool:
    """Check if a chain is supported for Uniswap V3 integration."""
    return chain_name.lower() in UNISWAP_V3_CONTRACTS


def get_supported_chains() -> list[str]:
    """Get list of supported chains for Uniswap V3."""
    return list(UNISWAP_V3_CONTRACTS.keys())