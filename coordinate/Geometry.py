import re
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path for direct execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from coordinate import CoordSingle, CoordMulti
from unite.base import BaseData, BaseMultiData


class Geometry:
    """
    Geometry analysis class for JADE-NAMD simulations.
    
    Supports both single and multi-trajectory coordinate data.
    Automatically detects input type and handles accordingly.
    Compatible with BaseData/BaseMultiData for Unite operations.
    """

    def __init__(self, coord_data, *atoms, type: str = "auto", unwrap_dihedral: bool = True):
        """
        Initialize Geometry.
        
        Args:
            coord_data: CoordSingle, CoordMulti instance, or path to CSV/Pickle file
            *atoms: Atom labels for geometry calculation (2 for distance, 3 for angle, 4 for dihedral)
            type: Input type ('auto' for auto-detection, 'coord_single', 'coord_multi', 'csv', 'pickle')
            unwrap_dihedral: Whether to unwrap dihedral angles (default: True)
        
        Raises:
            ValueError: If input type cannot be determined or atom count is invalid
        """
        self.coord_single = None
        self.coord_multi = None
        self.atoms = tuple(atoms)
        self.unwrap_dihedral = bool(unwrap_dihedral)
        self.source_type = "geometry"
        self.max_i_time = None
        self.n = 0  # Number of trajectories (0 for single)
        
        if type == "auto":
            # Auto-detect input type
            if isinstance(coord_data, CoordSingle):
                self._init_from_coord_single(coord_data)
            elif isinstance(coord_data, CoordMulti):
                self._init_from_coord_multi(coord_data)
            elif isinstance(coord_data, str) or isinstance(coord_data, Path):
                # Check file extension
                path_str = str(coord_data)
                if path_str.endswith('.csv'):
                    self._init_from_csv(path_str)
                elif path_str.endswith('.pkl') or path_str.endswith('.pickle'):
                    self._init_from_pickle(path_str)
                else:
                    raise ValueError("Cannot determine file type from extension")
            else:
                raise ValueError(f"Unsupported input type: {type(coord_data)}")
        
        elif type == "coord_single":
            if isinstance(coord_data, CoordSingle):
                self._init_from_coord_single(coord_data)
            else:
                raise ValueError("coord_data must be CoordSingle instance")
        
        elif type == "coord_multi":
            if isinstance(coord_data, CoordMulti):
                self._init_from_coord_multi(coord_data)
            else:
                raise ValueError("coord_data must be CoordMulti instance")
        
        elif type == "csv":
            self._init_from_csv(coord_data)
        
        elif type == "pickle":
            self._init_from_pickle(coord_data)
        
        else:
            raise ValueError("type must be 'auto', 'coord_single', 'coord_multi', 'csv', or 'pickle'")
    
    def _init_from_coord_single(self, coord_single: CoordSingle):
        """Initialize from CoordSingle instance."""
        self.coord_single = coord_single
        self.max_i_time = coord_single.max_i_time
        self.n = 0  # Single trajectory
        
        if len(self.atoms) not in (2, 3, 4):
            raise ValueError(
                "Please provide 2 (distance), 3 (angle), or 4 (dihedral) atoms"
            )
        
        if len(self.atoms) == 2:
            self.kind = "distance"
        elif len(self.atoms) == 3:
            self.kind = "angle"
        else:
            self.kind = "dihedral"
        
        self.time = coord_single.data["time"].copy()
        self.data = self._build_single_dataframe()
    
    def _init_from_coord_multi(self, coord_multi: CoordMulti):
        """Initialize from CoordMulti instance."""
        self.coord_multi = coord_multi
        self.max_i_time = coord_multi.max_i_time
        self.n = coord_multi.n
        
        if len(self.atoms) not in (2, 3, 4):
            raise ValueError(
                "Please provide 2 (distance), 3 (angle), or 4 (dihedral) atoms"
            )
        
        if len(self.atoms) == 2:
            self.kind = "distance"
        elif len(self.atoms) == 3:
            self.kind = "angle"
        else:
            self.kind = "dihedral"
        
        self.time = coord_multi.data["time"].copy()
        self._traj_indices = list(range(1, self.n + 1))
        self.data = self._build_multi_dataframe()
    
    def _init_from_csv(self, path: str):
        """Initialize from CSV file."""
        self.data = pd.read_csv(path)
        self.time = self.data["time"].copy()
        self._traj_indices = self._detect_traj_indices(self.data)
        self.n = len(self._traj_indices) if self._traj_indices else 0
        self.kind = self._infer_kind_from_data()
        self._apply_unwrap_config_to_loaded_data()
    
    def _init_from_pickle(self, path: str):
        """Initialize from Pickle file."""
        self.data = pd.read_pickle(path)
        self.time = self.data["time"].copy()
        self._traj_indices = self._detect_traj_indices(self.data)
        self.n = len(self._traj_indices) if self._traj_indices else 0
        self.kind = self._infer_kind_from_data()
        self._apply_unwrap_config_to_loaded_data()
    
    def _detect_traj_indices(self, df: pd.DataFrame) -> List[int]:
        """Detect trajectory indices from column names."""
        traj_indices = set()
        pattern = re.compile(r"_No\.(\d+)$")
        for col in df.columns:
            m = pattern.search(col)
            if m:
                traj_indices.add(int(m.group(1)))
        return sorted(traj_indices)
    
    def _infer_kind_from_data(self) -> str:
        """Infer geometry kind from data column names."""
        value_cols = [c for c in self.data.columns if c != "time"]
        if any(c.startswith("distance_") for c in value_cols):
            return "distance"
        if any(c.startswith("angle_") for c in value_cols):
            return "angle"
        if any(c.startswith("dihedral_") for c in value_cols):
            return "dihedral"
        return "geometry"
    
    def _normalize_atom_label(self, atom_label: str) -> str:
        """Normalize atom label to standard format."""
        atom_label = str(atom_label).strip()
        
        m = re.fullmatch(r"([A-Za-z]+)_?(\d+)", atom_label)
        if m:
            symbol, index = m.group(1), m.group(2)
            return f"{symbol}_{index}"
        
        m = re.fullmatch(r"([A-Za-z]+)_(\d+)", atom_label)
        if m:
            symbol, index = m.group(1), m.group(2)
            return f"{symbol}_{index}"
        
        raise ValueError(
            f"Invalid atom label: {atom_label}. Use formats like N1 or N_1"
        )
    
    def _atom_xyz_single(self, atom_label: str) -> np.ndarray:
        """Get atom coordinates for single trajectory."""
        atom_prefix = self._normalize_atom_label(atom_label)
        
        x_col = f"{atom_prefix}_x"
        y_col = f"{atom_prefix}_y"
        z_col = f"{atom_prefix}_z"
        
        missing = [
            c for c in (x_col, y_col, z_col) if c not in self.coord_single.data.columns
        ]
        if missing:
            raise KeyError(f"Missing atom columns for {atom_label}: {missing}")
        
        return self.coord_single.data[[x_col, y_col, z_col]].to_numpy(dtype=float)
    
    def _atom_xyz_multi(self, atom_label: str, traj_index: int) -> np.ndarray:
        """Get atom coordinates for specific trajectory in multi-trajectory data."""
        atom_prefix = self._normalize_atom_label(atom_label)
        
        x_col = f"{atom_prefix}_x_No.{traj_index}"
        y_col = f"{atom_prefix}_y_No.{traj_index}"
        z_col = f"{atom_prefix}_z_No.{traj_index}"
        
        missing = [
            c for c in (x_col, y_col, z_col) if c not in self.coord_multi.data.columns
        ]
        if missing:
            raise KeyError(
                f"Missing atom columns for {atom_label} in trajectory No.{traj_index}: {missing}"
            )
        
        return self.coord_multi.data[[x_col, y_col, z_col]].to_numpy(dtype=float)
    
    @staticmethod
    def _distance(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Calculate distance between two points."""
        return np.linalg.norm(p2 - p1, axis=1)
    
    @staticmethod
    def _angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        """Calculate angle between three points."""
        v1 = p1 - p2
        v2 = p3 - p2
        cross_norm = np.linalg.norm(np.cross(v1, v2), axis=1)
        dot_val = np.einsum("ij,ij->i", v1, v2)
        angle_rad = np.arctan2(cross_norm, dot_val)
        return np.degrees(angle_rad)
    
    @staticmethod
    def _dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
        """Calculate dihedral angle between four points."""
        ab = p2 - p1
        bc = p3 - p2
        cd = p4 - p3
        
        ab_norm = np.linalg.norm(ab, axis=1, keepdims=True)
        bc_norm = np.linalg.norm(bc, axis=1, keepdims=True)
        cd_norm = np.linalg.norm(cd, axis=1, keepdims=True)
        
        with np.errstate(invalid="ignore", divide="ignore"):
            ab_unit = ab / ab_norm
            bc_unit = bc / bc_norm
            cd_unit = cd / cd_norm
        
        m = np.cross(ab_unit, bc_unit)
        n = np.cross(bc_unit, cd_unit)
        
        m_norm = np.linalg.norm(m, axis=1, keepdims=True)
        n_norm = np.linalg.norm(n, axis=1, keepdims=True)
        
        with np.errstate(invalid="ignore", divide="ignore"):
            m_unit = m / m_norm
            n_unit = n / n_norm
        
        x = np.einsum("ij,ij->i", m_unit, n_unit)
        y = np.einsum("ij,ij->i", np.cross(m_unit, n_unit), bc_unit)
        
        angle_rad = np.arctan2(y, x)
        return np.degrees(angle_rad)
    
    @staticmethod
    def _unwrap_degrees(angle_deg: np.ndarray) -> np.ndarray:
        """Unwrap angle values across 360-degree boundaries."""
        angle_deg = np.asarray(angle_deg, dtype=float)
        unwrapped = angle_deg.copy()
        
        finite_mask = np.isfinite(angle_deg)
        if not finite_mask.any():
            return unwrapped
        
        valid_idx = np.where(finite_mask)[0]
        
        start = 0
        while start < len(valid_idx):
            end = start
            while end + 1 < len(valid_idx) and valid_idx[end + 1] == valid_idx[end] + 1:
                end += 1
            
            seg = valid_idx[start : end + 1]
            rad = np.deg2rad(angle_deg[seg])
            unwrapped[seg] = np.rad2deg(np.unwrap(rad, discont=np.pi))
            start = end + 1
        
        return unwrapped
    
    @staticmethod
    def _rewrap_degrees(angle_deg: np.ndarray) -> np.ndarray:
        """Wrap angle values to [-180, 180] range."""
        angle_deg = np.asarray(angle_deg, dtype=float)
        wrapped = angle_deg.copy()
        
        finite_mask = np.isfinite(angle_deg)
        if finite_mask.any():
            wrapped[finite_mask] = ((angle_deg[finite_mask] + 180.0) % 360.0) - 180.0
        
        return wrapped
    
    def _apply_unwrap_config_to_loaded_data(self):
        """Apply unwrap configuration to loaded data."""
        if self.kind != "dihedral":
            return
        
        dihedral_cols = [c for c in self.data.columns if c.startswith("dihedral_")]
        for col in dihedral_cols:
            values = self.data[col].to_numpy(dtype=float)
            if self.unwrap_dihedral:
                self.data[col] = self._unwrap_degrees(values)
            else:
                self.data[col] = self._rewrap_degrees(values)
    
    def _build_single_dataframe(self) -> pd.DataFrame:
        """Build DataFrame for single trajectory."""
        time_series = self.time.reset_index(drop=True)
        
        points = [self._atom_xyz_single(atom) for atom in self.atoms]
        
        if self.kind == "distance":
            value = self._distance(points[0], points[1])
        elif self.kind == "angle":
            value = self._angle(points[0], points[1], points[2])
        else:
            value = self._dihedral(points[0], points[1], points[2], points[3])
            if self.unwrap_dihedral:
                value = self._unwrap_degrees(value)
        
        atom_text = "-".join(self.atoms)
        value_df = pd.DataFrame({f"{self.kind}_{atom_text}": value}, index=time_series.index)
        return pd.concat([time_series.rename("time"), value_df], axis=1)
    
    def _build_multi_dataframe(self) -> pd.DataFrame:
        """Build DataFrame for multi trajectory."""
        time_series = self.time.reset_index(drop=True)
        value_columns = {}
        
        atom_text = "-".join(self.atoms)
        for traj_index in self._traj_indices:
            points = [self._atom_xyz_multi(atom, traj_index) for atom in self.atoms]
            
            if self.kind == "distance":
                value = self._distance(points[0], points[1])
            elif self.kind == "angle":
                value = self._angle(points[0], points[1], points[2])
            else:
                value = self._dihedral(points[0], points[1], points[2], points[3])
                if self.unwrap_dihedral:
                    value = self._unwrap_degrees(value)
            
            value_columns[f"{self.kind}_{atom_text}_No.{traj_index}"] = value
        
        value_df = pd.DataFrame(value_columns, index=time_series.index)
        return pd.concat([time_series.rename("time"), value_df], axis=1)
    
    def has_real_time(self) -> bool:
        """Check if data has real time values."""
        time_vals = self.data['time'].values
        if len(time_vals) < 2:
            return False
        return np.issubdtype(time_vals.dtype, np.floating)
    
    def get_time_series(self) -> np.ndarray:
        """Get the time series as numpy array."""
        return self.data['time'].values
    
    def save_to_csv(self, path: str):
        """Save data to CSV file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data.to_csv(path, index=False)
    
    def save_to_pickle(self, path: str):
        """Save data to Pickle file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data.to_pickle(path)
    
    def get_trajectory_columns(self, trajectory_idx: int) -> List[str]:
        """Get columns for a specific trajectory (multi-trajectory only)."""
        if self.n == 0:
            raise ValueError("This is a single trajectory instance")
        
        suffix = f'_No.{trajectory_idx}'
        return [col for col in self.data.columns if col == 'time' or col.endswith(suffix)]
    
    def get_trajectory_data(self, trajectory_idx: int) -> pd.DataFrame:
        """Extract data for a specific trajectory (multi-trajectory only)."""
        if self.n == 0:
            raise ValueError("This is a single trajectory instance")
        
        cols = self.get_trajectory_columns(trajectory_idx)
        df = self.data[cols].copy()
        df.columns = [col.replace(f'_No.{trajectory_idx}', '') for col in df.columns]
        return df
    
    def __len__(self) -> int:
        """Get length of data."""
        return len(self.data)
    
    def __repr__(self) -> str:
        """Get string representation."""
        if self.n > 0:
            return f"Geometry(kind='{self.kind}', n_trajectories={self.n}, max_i_time={self.max_i_time})"
        else:
            return f"Geometry(kind='{self.kind}', single_trajectory, max_i_time={self.max_i_time})"


if __name__ == "__main__":
    # Test with CoordSingle
    print("Testing with CoordSingle...")
    coord_single = CoordSingle(r"E:\GitHub\JADE-Analyzer\sample\1", 500)
    geom_single = Geometry(coord_single, "C3", "N2", "N1", "C7")
    print(f"Single trajectory geometry: {geom_single}")
    print(f"Data shape: {geom_single.data.shape}")
    print(f"Columns: {geom_single.data.columns.tolist()}")
    print()
    
    # Test with CoordMulti
    print("Testing with CoordMulti...")
    paths = [
        r"E:\GitHub\JADE-Analyzer\sample\1",
        r"E:\GitHub\JADE-Analyzer\sample\2",
    ]
    coord_multi = CoordMulti(paths, 500)
    geom_multi = Geometry(coord_multi, "C3", "N2", "N1", "C7")
    print(f"Multi trajectory geometry: {geom_multi}")
    print(f"Data shape: {geom_multi.data.shape}")
    print(f"Columns: {geom_multi.data.columns.tolist()}")
    print()
    
    # Test Unite compatibility
    print("Testing Unite compatibility...")
    from unite import DataUniter, MultiTrajectoryUniter
    
    # Test DataUniter with single trajectory
    uniter_single = DataUniter()
    uniter_single.add_source("geometry", geom_single)
    print(f"DataUniter with Geometry: {uniter_single}")
    
    # Test MultiTrajectoryUniter with multi trajectory
    uniter_multi = MultiTrajectoryUniter()
    uniter_multi.add_source("geometry", geom_multi)
    print(f"MultiTrajectoryUniter with Geometry: {uniter_multi}")
    
    print("\nAll tests passed!")
