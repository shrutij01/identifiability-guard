"""
Identifiability Guard: A framework for evaluating identifiability metrics.

This package provides:
- Data Generating Processes (DGPs): D1-D4
- Encoder Mixings: E1-E6
- Identifiability Metrics: MCC, DCI
"""

from . import dgp
from . import encoders
from . import metrics

__version__ = "0.1.0"
