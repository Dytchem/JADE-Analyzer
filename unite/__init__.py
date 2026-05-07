"""
Unite module for JADE-Analyzer.

Provides classes to integrate multiple data sources and enable unified time handling.
"""

from .base import BaseData, BaseMultiData
from .unite import DataUniter, MultiTrajectoryUniter

__all__ = ["BaseData", "BaseMultiData", "DataUniter", "MultiTrajectoryUniter"]
