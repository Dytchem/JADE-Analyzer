from typing import List

import numpy as np
import pandas as pd

from unite.base import BaseMultiData
from .EnergySingle import EnergySingle


class EnergyMulti(BaseMultiData):
    def __init__(self, path, max_i_time, type="folder"):
        self.max_i_time = max_i_time
        
        if type == "folder":
            frames = []
            for i, p in enumerate(path):
                energy_single = EnergySingle(p, max_i_time, type="folder")
                if i == 0:
                    first = energy_single.data.copy()
                    energy_columns = [c for c in first.columns if c != "time"]
                    rename_map = {c: f"{c}_No.{i+1}" for c in energy_columns}
                    first = first.rename(columns=rename_map)
                    frames.append(first)
                else:
                    one = energy_single.data.drop(columns=["time"]).copy()
                    rename_map = {c: f"{c}_No.{i+1}" for c in one.columns}
                    one = one.rename(columns=rename_map)
                    frames.append(one)

            if not frames:
                raise ValueError("path is empty when type is 'folder'")
            data = pd.concat(frames, axis=1)

        elif type == "csv":
            data = pd.read_csv(path)
        elif type == "pickle":
            data = pd.read_pickle(path)
        else:
            raise ValueError("type must be 'folder' or 'csv' or 'pickle'")

        n_trajectories = (data.shape[1] - 1) // 7
        super().__init__(data, max_i_time, "energy", n_trajectories)

    def save_to_csv(self, path):
        self.data.to_csv(path, index=False)

    def save_to_pickle(self, path):
        self.data.to_pickle(path)


if __name__ == "__main__":
    path = [
        r"E:\GitHub\JADE-Analyzer\sample\1",
        r"E:\GitHub\JADE-Analyzer\sample\2",
        r"E:\GitHub\JADE-Analyzer\sample\3",
    ]
    max_i_time = 500
    energy = EnergyMulti(path, max_i_time)
    energy.save_to_csv(r"E:\GitHub\JADE-Analyzer\output\energy_multi.csv")

    print(energy)
    print(energy.data.head())
