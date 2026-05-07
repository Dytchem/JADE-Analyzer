"""
Base classes for all data modules in JADE-Analyzer.

This module provides a common base class that all data classes should inherit from,
enabling unified time handling and data integration capabilities.
"""

import os

import numpy as np
import pandas as pd


class BaseData:
    """
    Abstract base class for all trajectory data types.
    
    Provides common functionality for:
    - Time series management
    - Data validation
    - Saving to various formats
    - Data integration support
    """
    
    def __init__(self, data: pd.DataFrame, max_i_time: int, source_type: str):
        """
        Initialize the base data class.
        
        Args:
            data: DataFrame containing the data with at least a 'time' column
            max_i_time: Maximum time index
            source_type: Type identifier (e.g., 'state', 'coordinate', 'di', 'energy')
        """
        self.data = data
        self.max_i_time = max_i_time
        self.source_type = source_type
        self._validate_time_column()
        self.time_interval = self._calculate_time_interval()
    
    def _validate_time_column(self):
        """Validate that the DataFrame has a 'time' column."""
        if 'time' not in self.data.columns:
            raise ValueError("DataFrame must contain a 'time' column")
    
    def _calculate_time_interval(self):
        """Calculate the time interval between consecutive frames."""
        if len(self.data) < 2:
            return np.nan
        return float(self.data['time'].iloc[1] - self.data['time'].iloc[0])
    
    def set_time_series(self, time_series):
        """
        Set a custom time series for the data.
        
        Args:
            time_series: 1D sequence of time values
            
        Raises:
            ValueError: If time_series is not 1D or length doesn't match
        """
        time_array = np.asarray(time_series, dtype=float)
        if time_array.ndim != 1:
            raise ValueError("time_series must be a 1D sequence")
        if len(time_array) != len(self.data):
            raise ValueError(
                f"time_series length {len(time_array)} does not match data length {len(self.data)}"
            )

        self.data.loc[:, "time"] = time_array
        self.time_interval = self._calculate_time_interval()
    
    def has_real_time(self):
        """
        Check if the data has real time values (float) or just indices (integer).
        
        Returns:
            bool: True if time values are floats (indicating real time data)
        """
        time_vals = self.data['time'].values
        if len(time_vals) < 2:
            return False
        
        return np.issubdtype(time_vals.dtype, np.floating)
    
    def get_time_series(self):
        """Get the time series as a numpy array."""
        return self.data['time'].values
    
    def save_to_csv(self, path):
        """Save data to CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data.to_csv(path, index=False)
    
    def save_to_pickle(self, path):
        """Save data to Pickle file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data.to_pickle(path)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(source_type='{self.source_type}', max_i_time={self.max_i_time}, has_real_time={self.has_real_time()})"
    
    def __len__(self):
        return len(self.data)


class BaseMultiData(BaseData):
    """
    Abstract base class for multi-trajectory data.
    
    Extends BaseData with support for multiple trajectories and trajectory-specific
    column naming.
    """
    
    def __init__(self, data: pd.DataFrame, max_i_time: int, source_type: str, n_trajectories: int):
        super().__init__(data, max_i_time, source_type)
        self.n = n_trajectories
    
    def get_trajectory_columns(self, trajectory_idx: int):
        """
        Get all columns belonging to a specific trajectory.
        
        Args:
            trajectory_idx: 1-based trajectory index
        
        Returns:
            list: Column names for the specified trajectory
        """
        suffix = f'_No.{trajectory_idx}'
        return [col for col in self.data.columns if col == 'time' or col.endswith(suffix)]
    
    def get_trajectory_data(self, trajectory_idx: int):
        """
        Extract data for a specific trajectory.
        
        Args:
            trajectory_idx: 1-based trajectory index
        
        Returns:
            pd.DataFrame: Data for the specified trajectory
        """
        cols = self.get_trajectory_columns(trajectory_idx)
        df = self.data[cols].copy()
        df.columns = [col.replace(f'_No.{trajectory_idx}', '') for col in df.columns]
        return df
