"""Pytest config to ensure local package imports resolve."""

import sys
from pathlib import Path

# Add repository src/ to sys.path so `identifiability_guard` imports work in tests.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
