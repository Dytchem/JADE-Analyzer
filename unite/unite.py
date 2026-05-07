"""
Unite module for integrating multiple data sources.

This module provides classes to combine different types of trajectory data
while ensuring data consistency and enabling shared time series access.
"""

from typing import List, Tuple, Type, Union

import numpy as np
import pandas as pd

from .base import BaseData, BaseMultiData


class DataUniter:
    """
    Class for uniting multiple data sources from the same trajectory.
    
    Ensures data consistency by:
    - Validating matching time indices
    - Synchronizing time series across sources
    - Providing unified access to combined data
    """
    
    def __init__(self):
        self.data_sources = {}
        self.unified_data = None
    
    def add_source(self, name: str, data: BaseData):
        """
        Add a data source to the uniter.
        
        Args:
            name: Name to identify this data source
            data: BaseData instance to add
        
        Raises:
            ValueError: If data is not a BaseData instance or if dimensions don't match
        """
        if not isinstance(data, BaseData):
            raise ValueError("Data must be an instance of BaseData")
        
        if self.data_sources:
            first_source = next(iter(self.data_sources.values()))
            if len(data) != len(first_source):
                raise ValueError(
                    f"Data length {len(data)} does not match existing sources ({len(first_source)})"
                )
            if data.max_i_time != first_source.max_i_time:
                raise ValueError(
                    f"max_i_time {data.max_i_time} does not match existing sources ({first_source.max_i_time})"
                )
        
        self.data_sources[name] = data
        self._update_unified_data()
    
    def remove_source(self, name: str):
        """
        Remove a data source from the uniter.
        
        Args:
            name: Name of the data source to remove
        """
        if name in self.data_sources:
            del self.data_sources[name]
            self._update_unified_data()
    
    def _update_unified_data(self):
        """Update the unified DataFrame when sources change."""
        if not self.data_sources:
            self.unified_data = None
            return
        
        time_source = None
        for name, data in self.data_sources.items():
            if data.has_real_time():
                time_source = name
                break
        
        merged = None
        for name, data in self.data_sources.items():
            if merged is None:
                merged = data.data.copy()
                non_time_cols = [col for col in merged.columns if col != 'time']
                merged.columns = ['time'] + [f"{name}_{col}" for col in non_time_cols]
            else:
                temp = data.data.copy()
                non_time_cols = [col for col in temp.columns if col != 'time']
                temp.columns = ['time'] + [f"{name}_{col}" for col in non_time_cols]
                merged = merged.merge(temp, on='time', how='outer')
        
        self.unified_data = merged
        
        if time_source and self.unified_data is not None:
            real_time = self.unified_data['time'].values
            for name, data in self.data_sources.items():
                if not data.has_real_time():
                    data.set_time_series(real_time)
    
    def get_unified_data(self) -> pd.DataFrame:
        """Get the merged DataFrame containing all data sources."""
        return self.unified_data
    
    def get_source(self, name: str) -> BaseData:
        """Get a specific data source by name."""
        return self.data_sources.get(name)
    
    def get_time_series(self) -> np.ndarray:
        """Get the unified time series."""
        if self.unified_data is None:
            return np.array([])
        return self.unified_data['time'].values
    
    def has_real_time(self) -> bool:
        """Check if any source has real time values."""
        return any(data.has_real_time() for data in self.data_sources.values())
    
    def plot_correlation(self, x_col: str, y_col: str, **kwargs):
        """
        Create a correlation plot between two columns.
        
        Args:
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            **kwargs: Additional plotting arguments
        
        Returns:
            matplotlib figure and axis objects
        """
        if self.unified_data is None:
            raise ValueError("No data sources added")
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
        ax.scatter(self.unified_data[x_col], self.unified_data[y_col], **kwargs)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{x_col} vs {y_col}')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        return fig, ax
    
    def save_unified_to_csv(self, path: str):
        """Save unified data to CSV."""
        if self.unified_data is None:
            raise ValueError("No data sources to save")
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.unified_data.to_csv(path, index=False)
    
    def save_unified_to_pickle(self, path: str):
        """Save unified data to Pickle."""
        if self.unified_data is None:
            raise ValueError("No data sources to save")
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.unified_data.to_pickle(path)
    
    def __repr__(self):
        sources = ", ".join(self.data_sources.keys())
        return f"DataUniter(sources=[{sources}], has_real_time={self.has_real_time()})"
    
    def __len__(self):
        if not self.data_sources:
            return 0
        return len(next(iter(self.data_sources.values())))


class MultiTrajectoryUniter:
    """
    Class for uniting multiple trajectory data from multiple sources.
    
    Handles multi-trajectory data (Multi classes) and provides consolidated access.
    """
    
    def __init__(self):
        self.data_sources = {}
        self.n_trajectories = 0
    
    def add_source(self, name: str, data: BaseMultiData):
        """
        Add a multi-trajectory data source.
        
        Args:
            name: Name to identify this data source
            data: BaseMultiData instance to add
        
        Raises:
            ValueError: If data is not a BaseMultiData instance or if trajectory count doesn't match
        """
        if not isinstance(data, BaseMultiData):
            raise ValueError("Data must be an instance of BaseMultiData")
        
        if self.data_sources:
            if data.n != self.n_trajectories:
                raise ValueError(
                    f"Number of trajectories {data.n} does not match existing sources ({self.n_trajectories})"
                )
        else:
            self.n_trajectories = data.n
        
        self.data_sources[name] = data
    
    def get_trajectory_uniter(self, trajectory_idx: int) -> DataUniter:
        """
        Get a DataUniter for a specific trajectory.
        
        Args:
            trajectory_idx: 1-based trajectory index
        
        Returns:
            DataUniter: Uniter containing data for the specified trajectory
        """
        if trajectory_idx < 1 or trajectory_idx > self.n_trajectories:
            raise ValueError(f"Invalid trajectory index {trajectory_idx}")
        
        uniter = DataUniter()
        for name, data in self.data_sources.items():
            traj_data = data.get_trajectory_data(trajectory_idx)
            
            class TempBaseData(BaseData):
                def __init__(self, df, max_i_time, source_type):
                    super().__init__(df, max_i_time, source_type)
            
            temp = TempBaseData(traj_data, data.max_i_time, data.source_type)
            uniter.add_source(name, temp)
        
        return uniter
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics across all trajectories and sources."""
        stats = []
        for name, data in self.data_sources.items():
            for traj_idx in range(1, self.n_trajectories + 1):
                traj_data = data.get_trajectory_data(traj_idx)
                for col in traj_data.columns:
                    if col == 'time':
                        continue
                    stats.append({
                        'source': name,
                        'trajectory': traj_idx,
                        'column': col,
                        'mean': traj_data[col].mean(),
                        'std': traj_data[col].std(),
                        'min': traj_data[col].min(),
                        'max': traj_data[col].max()
                    })
        return pd.DataFrame(stats)
    
    def __repr__(self):
        sources = ", ".join(self.data_sources.keys())
        return f"MultiTrajectoryUniter(sources=[{sources}], n_trajectories={self.n_trajectories})"
