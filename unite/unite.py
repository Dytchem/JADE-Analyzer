"""
Unite module for integrating multiple data sources.

This module provides classes to combine different types of trajectory data
while ensuring data consistency and enabling shared time series access.

Key design:
- Step is represented by DataFrame index (0, 1, 2, ...)
- Some data sources have a 'time' column (real time values)
- Merge is done on index (step)
- Common time axis is derived from sources with real time
"""

from typing import List, Tuple, Type, Union

import numpy as np
import pandas as pd

from .base import BaseData, BaseMultiData


class DataUniter:
    """
    Class for uniting multiple data sources from the same trajectory.
    
    Ensures data consistency by:
    - Validating matching step indices
    - Synchronizing time series across sources
    - Providing unified access to combined data
    
    Time axis matching rules:
    - When both sources have real time: time axes must match exactly
    - When one has real time and one doesn't: step counts must match
    - When neither has real time: step counts must match, use step as time
    """
    
    def __init__(self):
        self.data_sources = {}
        self.unified_data = None
    
    def add_source(self, name: str, data):
        """
        Add a data source to the uniter.
        
        Args:
            name: Name to identify this data source
            data: BaseData instance or object with compatible interface (data, max_i_time, has_real_time())
        
        Raises:
            ValueError: If data doesn't have required interface or if dimensions don't match
        """
        # Check for BaseData interface (duck typing)
        if not hasattr(data, 'data') or not hasattr(data, 'max_i_time') or not hasattr(data, 'has_real_time'):
            raise ValueError("Data must have BaseData interface (data, max_i_time, has_real_time)")
        
        if self.data_sources:
            first_source = next(iter(self.data_sources.values()))
            
            # Check step count consistency (using DataFrame index)
            if len(data) != len(first_source):
                raise ValueError(
                    f"Data length {len(data)} does not match existing sources ({len(first_source)})"
                )
            if data.max_i_time != first_source.max_i_time:
                raise ValueError(
                    f"max_i_time {data.max_i_time} does not match existing sources ({first_source.max_i_time})"
                )
            
            # Check time axis consistency if both have real time
            if data.has_real_time() and first_source.has_real_time():
                if not np.allclose(data.data['time'].values, first_source.data['time'].values):
                    raise ValueError(
                        f"Time axes do not match between {name} and existing sources"
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
        
        # Find all sources with real time
        real_time_sources = [(name, data) for name, data in self.data_sources.items() if data.has_real_time()]
        
        # Validate all real time sources have matching time axes
        if len(real_time_sources) > 1:
            first_time = real_time_sources[0][1].data['time'].values
            for name, data in real_time_sources[1:]:
                if not np.allclose(data.data['time'].values, first_time):
                    raise ValueError(f"Time axes do not match between {real_time_sources[0][0]} and {name}")
        
        # Determine the reference time axis
        if real_time_sources:
            # Use the first real time source as reference
            ref_name, ref_data = real_time_sources[0]
            ref_time = ref_data.data['time'].values
        else:
            # No real time sources, use step indices as time
            first_source = next(iter(self.data_sources.values()))
            ref_time = first_source.data.index.values.astype(float)
        
        # Merge all data sources
        # First source sets the index (step), subsequent sources align by index
        merged = None
        time_added = False  # Track if time column has been added
        
        for name, data in self.data_sources.items():
            temp = data.data.copy()
            
            # Remove any existing time columns from all sources except the first one
            # This prevents duplicate time columns when sources have been modified by set_time_series
            if time_added and 'time' in temp.columns:
                temp = temp.drop('time', axis=1)
            
            # Only add time column from the first source or from reference
            if not time_added:
                if 'time' not in temp.columns:
                    temp.insert(0, 'time', ref_time)
                else:
                    temp['time'] = ref_time
                time_added = True
            
            # Rename columns to include source name (except time)
            cols_to_rename = [col for col in temp.columns if col != 'time']
            rename_map = {col: f"{name}_{col}" for col in cols_to_rename}
            temp = temp.rename(columns=rename_map)
            
            if merged is None:
                merged = temp
            else:
                # Concatenate horizontally, aligning by index
                merged = pd.concat([merged, temp], axis=1)
        
        self.unified_data = merged
        
        # Update time series for sources without real time
        if real_time_sources and self.unified_data is not None and len(self.unified_data) > 0:
            real_time = self.unified_data['time'].values
            if len(real_time) > 0:
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
    
    def get_step_series(self) -> np.ndarray:
        """Get the unified step series (DataFrame index)."""
        if self.unified_data is None:
            return np.array([])
        return self.unified_data.index.values
    
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
        self._unified_data = None
    
    def add_source(self, name: str, data):
        """
        Add a multi-trajectory data source.
        
        Args:
            name: Name to identify this data source
            data: BaseMultiData instance or object with compatible interface
        
        Raises:
            ValueError: If data doesn't have required interface or if trajectory count doesn't match
        """
        # Check for BaseMultiData interface (duck typing)
        required_attrs = ['data', 'max_i_time', 'n', 'get_trajectory_columns', 'get_trajectory_data']
        if not all(hasattr(data, attr) for attr in required_attrs):
            raise ValueError(f"Data must have BaseMultiData interface: {required_attrs}")
        
        if self.data_sources:
            if data.n != self.n_trajectories:
                raise ValueError(
                    f"Number of trajectories {data.n} does not match existing sources ({self.n_trajectories})"
                )
        else:
            self.n_trajectories = data.n
        
        self.data_sources[name] = data
        self._unified_data = None
    
    def get_unified_data(self) -> pd.DataFrame:
        """Get a merged DataFrame of all data sources."""
        if self._unified_data is not None:
            return self._unified_data
        
        if not self.data_sources:
            return pd.DataFrame()
        
        merged = None
        time_added = False
        for name, data in self.data_sources.items():
            temp = data.data.copy()
            if time_added and 'time' in temp.columns:
                temp = temp.drop('time', axis=1)
            time_added = True
            cols_to_rename = [c for c in temp.columns if c != 'time']
            rename_map = {col: f"{name}_{col}" for col in cols_to_rename}
            temp = temp.rename(columns=rename_map)
            if merged is None:
                merged = temp
            else:
                merged = pd.concat([merged, temp], axis=1)
        
        self._unified_data = merged
        return merged
    
    def save_unified_to_csv(self, path: str):
        """Save unified data to CSV."""
        df = self.get_unified_data()
        if df.empty:
            raise ValueError("No data to save")
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    
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
