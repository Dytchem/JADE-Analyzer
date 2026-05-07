"""
JADE-Analyzer - Molecular Dynamics Simulation Data Analysis Library

This package provides tools for analyzing excited state dynamics simulation data
from JADE-NAMD simulations.
"""

__version__ = "1.0.0"
__author__ = "JADE Team"

# Export main classes from submodules
from .state import StateSingle, StateMulti
from .coordinate import CoordSingle, CoordMulti, Geometry, PlotGeom, CountGeom, PlotCount
from .di import DiSingle, DiMulti
from .energy import EnergySingle, EnergyMulti
from .hop_coord import HopCoordSingle, HopCoordMulti
from .unite import DataUniter, MultiTrajectoryUniter, BaseData, BaseMultiData

__all__ = [
    # State
    "StateSingle",
    "StateMulti",
    # Coordinate
    "CoordSingle",
    "CoordMulti",
    "Geometry",
    "PlotGeom",
    "CountGeom",
    "PlotCount",
    # DI
    "DiSingle",
    "DiMulti",
    # Energy
    "EnergySingle",
    "EnergyMulti",
    # Hop Coord
    "HopCoordSingle",
    "HopCoordMulti",
    # Unite
    "DataUniter",
    "MultiTrajectoryUniter",
    "BaseData",
    "BaseMultiData",
]
