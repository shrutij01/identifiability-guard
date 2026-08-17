"""
Identifiability Guard: A framework for evaluating identifiability metrics.

This package provides:
- Data Generating Processes (DGPs): D1-D4
- Encoder Mixings: E1-E10
- Identifiability Metrics: MCC, DCI, R², MIG, T-MEX, and InfoMEC
"""

from . import dgp
from . import encoders
from . import metrics

__version__ = "0.1.0"
