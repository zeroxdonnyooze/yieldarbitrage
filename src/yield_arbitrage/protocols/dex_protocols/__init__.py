"""
DEX Protocol Math Implementations.

This package contains accurate mathematical implementations for various
decentralized exchange protocols, replacing simplified approximations
with real formulas used on-chain.
"""
from .uniswap_v2_math import UniswapV2Math
from .uniswap_v3_math import UniswapV3Math
from .curve_stableswap_math import CurveStableSwapMath
from .balancer_weighted_math import BalancerWeightedMath

__all__ = [
    "UniswapV2Math",
    "UniswapV3Math", 
    "CurveStableSwapMath",
    "BalancerWeightedMath"
]