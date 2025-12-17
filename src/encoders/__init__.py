"""
Encoder Mixings module.

Provides eight types of encoders:
- E1: Exact, elementwise linear
- E2: Exact, elementwise invertible nonlinear
- E3: Exact, linearly entangled
- E4: Undercomplete, elementwise linear
- E5: Overcomplete, elementwise linear
- E6: Overcomplete, multiple codes per factor
- E7: Overcomplete, linearly entangled
- E8: Overcomplete, nonlinear disjoint subsets
"""

from .base import BaseEncoder
from .e1_elementwise_linear import E1ElementwiseLinear
from .e2_elementwise_nonlinear import E2ElementwiseNonlinear
from .e3_linearly_entangled import E3LinearlyEntangled
from .e4_undercomplete_linear import E4UndercompleteLinear
from .e5_overcomplete_linear import E5OvercompleteLinear
from .e6_overcomplete_multicodes import E6OvercompleteMulticodes
from .e7_overcomplete_entangled import E7OvercompleteEntangled
from .e8_overcomplete_disjoint import E8OvercompleteDisjoint

__all__ = [
    "BaseEncoder",
    "E1ElementwiseLinear",
    "E2ElementwiseNonlinear",
    "E3LinearlyEntangled",
    "E4UndercompleteLinear",
    "E5OvercompleteLinear",
    "E6OvercompleteMulticodes",
    "E7OvercompleteEntangled",
    "E8OvercompleteDisjoint",
]
