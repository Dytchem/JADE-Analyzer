"""
ele_time module for JADE-Analyzer.

Provides classes for reading and analyzing electronic time data from JADE-NAMD simulations.
The ele_time.out file contains detailed information about electronic state evolution,
including density matrices, hopping probabilities, and state transitions.
"""

from .EleTimeSingle import EleTimeSingle
from .EleTimeMulti import EleTimeMulti

__all__ = [
    "EleTimeSingle",
    "EleTimeMulti",
]
