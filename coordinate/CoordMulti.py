from typing import List

import numpy as np
import pandas as pd

from .CoordSingle import CoordSingle
from unite.base import BaseMultiData


class CoordMulti(BaseMultiData):
    """
    Multi-trajectory coordinate data handler for JADE-NAMD simulations.
    
    Manages atomic coordinate data across multiple trajectories, providing
    methods for geometry analysis and visualization.
    """
    
    def __init__(self, path: List[str], max_i_time: int, type: str = "folder"):
        """
        Initialize CoordMulti.
        
        Args:
            path: List of paths to trajectory folders or path to CSV/Pickle file
            max_i_time: Maximum time index
            type: Data source type ('folder', 'csv', or 'pickle')
        """
        self.max_i_time = max_i_time
        
        if type == "folder":
            frames = []
            for i, p in enumerate(path):
                coord_single = CoordSingle(p, max_i_time, type="folder")
                if i == 0:
                    first = coord_single.data.copy()
                    coord_columns = [c for c in first.columns if c != "time"]
                    rename_map = {c: f"{c}_No.1" for c in coord_columns}
                    first = first.rename(columns=rename_map)
                    frames.append(first)
                else:
                    one = coord_single.data.drop(columns=["time"]).copy()
                    rename_map = {c: f"{c}_No.{i+1}" for c in one.columns}
                    one = one.rename(columns=rename_map)
                    frames.append(one)

            if not frames:
                raise ValueError("path is empty when type is 'folder'")
            data = pd.concat(frames, axis=1)
            n_trajectories = len(path)

        elif type == "csv":
            data = pd.read_csv(path)
            # Count trajectories by finding unique suffixes
            suffix_counts = {}
            for col in data.columns:
                if col != "time" and "_No." in col:
                    suffix = col.split("_No.")[-1]
                    suffix_counts[suffix] = 1
            n_trajectories = len(suffix_counts)
        elif type == "pickle":
            data = pd.read_pickle(path)
            suffix_counts = {}
            for col in data.columns:
                if col != "time" and "_No." in col:
                    suffix = col.split("_No.")[-1]
                    suffix_counts[suffix] = 1
            n_trajectories = len(suffix_counts)
        else:
            raise ValueError("type must be 'folder' or 'csv' or 'pickle'")
        
        super().__init__(data, max_i_time, "coordinate", n_trajectories)


if __name__ == "__main__":
    path = [
        r"E:\GitHub\JADE-Analyzer\sample\1_del",
        r"E:\GitHub\JADE-Analyzer\sample\1",
        r"E:\GitHub\JADE-Analyzer\sample\2",
    ]
    max_i_time = 500
    coord = CoordMulti(path, max_i_time)
    coord.save_to_csv(r"E:\GitHub\JADE-Analyzer\output\coord_multi.csv")

    print(coord.data)
