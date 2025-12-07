"""
Encoder Mixings module.

Provides six types of encoders:
- E1: Exact, elementwise linear
- E2: Exact, elementwise invertible nonlinear
- E3: Exact, linearly entangled
- E4: Undercomplete, elementwise linear
- E5: Overcomplete, elementwise linear
- E6: Overcomplete, multiple codes per factor
"""

from .base import BaseEncoder
from .e1_elementwise_linear import E1ElementwiseLinear
from .e2_elementwise_nonlinear import E2ElementwiseNonlinear
from .e3_linearly_entangled import E3LinearlyEntangled
from .e4_undercomplete_linear import E4UndercompleteLinear
from .e5_overcomplete_linear import E5OvercompleteLinear
from .e6_overcomplete_multicodes import E6OvercompleteMulticodes

__all__ = [
    "BaseEncoder",
    "E1ElementwiseLinear",
    "E2ElementwiseNonlinear",
    "E3LinearlyEntangled",
    "E4UndercompleteLinear",
    "E5OvercompleteLinear",
    "E6OvercompleteMulticodes",
]
