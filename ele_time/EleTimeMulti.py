"""
Multi-trajectory electronic time data handler for JADE-NAMD simulations.

This module provides classes to read and analyze multiple ele_time.out files
from different trajectories.
"""

from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from .EleTimeSingle import EleTimeSingle


class EleTimeMulti:
    """
    Multi-trajectory electronic time data handler.
    
    Reads and analyzes multiple ele_time.out files from different trajectories.
    """

    def __init__(self, paths: Union[str, List[str]], max_i_time: int = None, type: str = "folder"):
        """
        Initialize the EleTimeMulti instance.
        
        Args:
            paths: Path to directory containing trajectory folders or list of file paths
            max_i_time: Maximum time step to consider (optional)
            type: "folder" for directory containing trajectory folders, "csv" for single file mode
        
        Raises:
            FileNotFoundError: If specified paths do not exist
        """
        self.paths = paths
        self.max_i_time = max_i_time
        self.type = type
        self.trajectories: List[EleTimeSingle] = []
        self._load_trajectories()
    
    def _load_trajectories(self):
        """Load all trajectory data."""
        if self.type == "folder":
            # Path is a directory containing trajectory subdirectories
            base_path = Path(self.paths)
            if not base_path.exists():
                raise FileNotFoundError(f"Directory not found: {base_path}")
            
            # Find all subdirectories containing ele_time.out
            for item in base_path.iterdir():
                if item.is_dir():
                    ele_time_path = item / "ele_time.out"
                    if ele_time_path.exists():
                        try:
                            traj = EleTimeSingle(str(item), type="folder")
                            self.trajectories.append(traj)
                        except Exception as e:
                            print(f"Warning: Failed to load {item}: {e}")
        
        elif self.type == "file_list":
            # paths is a list of file paths
            for path in self.paths:
                if Path(path).exists():
                    try:
                        traj = EleTimeSingle(path, type="file")
                        self.trajectories.append(traj)
                    except Exception as e:
                        print(f"Warning: Failed to load {path}: {e}")
        
        if not self.trajectories:
            raise ValueError("No trajectories found")
    
    def get_num_trajectories(self) -> int:
        """Get number of loaded trajectories."""
        return len(self.trajectories)
    
    def get_trajectory(self, index: int) -> EleTimeSingle:
        """Get trajectory by index."""
        if index < 0 or index >= len(self.trajectories):
            raise IndexError(f"Trajectory index {index} out of range")
        return self.trajectories[index]
    
    def get_all_populations(self) -> List[np.ndarray]:
        """Get populations from all trajectories."""
        return [traj.get_populations() for traj in self.trajectories]
    
    def get_average_populations(self) -> np.ndarray:
        """
        Get time-averaged populations across all trajectories.
        
        Returns:
            Array of shape (n_steps, 2) with average populations
        """
        all_pops = self.get_all_populations()
        if not all_pops:
            return np.array([])
        
        # Find minimum length to align trajectories
        min_len = min(len(p) for p in all_pops)
        aligned = [p[:min_len] for p in all_pops]
        
        return np.mean(aligned, axis=0)
    
    def get_std_populations(self) -> np.ndarray:
        """
        Get standard deviation of populations across all trajectories.
        
        Returns:
            Array of shape (n_steps, 2) with population standard deviations
        """
        all_pops = self.get_all_populations()
        if not all_pops:
            return np.array([])
        
        min_len = min(len(p) for p in all_pops)
        aligned = [p[:min_len] for p in all_pops]
        
        return np.std(aligned, axis=0)
    
    def get_all_hopping_probabilities(self) -> List[np.ndarray]:
        """Get hopping probabilities from all trajectories."""
        return [traj.get_hopping_probabilities() for traj in self.trajectories]
    
    def get_average_hopping_probabilities(self) -> np.ndarray:
        """Get average hopping probabilities across all trajectories."""
        all_probs = self.get_all_hopping_probabilities()
        if not all_probs:
            return np.array([])
        
        min_len = min(len(p) for p in all_probs)
        aligned = [p[:min_len] for p in all_probs]
        
        return np.mean(aligned, axis=0)
    
    def get_all_times(self) -> np.ndarray:
        """Get time array (assuming all trajectories have same time grid)."""
        if self.trajectories:
            return self.trajectories[0].get_times()
        return np.array([])
    
    def count_state_transitions(self) -> dict:
        """Count state transitions across all trajectories."""
        transitions = {'1->2': 0, '2->1': 0, '1->1': 0, '2->2': 0}
        
        for traj in self.trajectories:
            states = traj.get_states()
            new_states = traj.get_new_states()
            
            for curr, new in zip(states, new_states):
                if curr == 1 and new == 2:
                    transitions['1->2'] += 1
                elif curr == 2 and new == 1:
                    transitions['2->1'] += 1
                elif curr == 1 and new == 1:
                    transitions['1->1'] += 1
                elif curr == 2 and new == 2:
                    transitions['2->2'] += 1
        
        return transitions
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all trajectory data to a single DataFrame."""
        all_rows = []
        
        for traj_idx, traj in enumerate(self.trajectories):
            df = traj.to_dataframe()
            df['trajectory'] = traj_idx
            all_rows.append(df)
        
        return pd.concat(all_rows, ignore_index=True)
    
    def save_to_csv(self, output_path: str):
        """Save combined data to CSV file."""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
    
    def description(self) -> dict:
        """Get summary description of all trajectories."""
        transitions = self.count_state_transitions()
        avg_pops = self.get_average_populations()
        
        return {
            'n_trajectories': self.get_num_trajectories(),
            'transitions': transitions,
            'total_transitions': sum(transitions.values()),
            'average_final_population': avg_pops[-1] if len(avg_pops) > 0 else None,
            'time_range': (self.get_all_times()[0], self.get_all_times()[-1]) if len(self.get_all_times()) > 0 else None
        }
