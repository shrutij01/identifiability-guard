"""
Data Generating Processes (DGPs) module.

Provides four types of DGPs:
- D1: Independent, non-redundant
- D2: Correlated, non-redundant  
- D3: Single-factor redundant
- D4: Multi-factor redundant
"""

from .base import BaseDGP
from .d1_independent import D1Independent
from .d2_correlated import D2Correlated
from .d3_single_redundant import D3SingleRedundant
from .d4_multi_redundant import D4MultiRedundant

__all__ = [
    "BaseDGP",
    "D1Independent",
    "D2Correlated",
    "D3SingleRedundant",
    "D4MultiRedundant",
]
