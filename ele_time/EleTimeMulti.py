from typing import List

import pandas as pd

from .EleTimeSingle import EleTimeSingle
from unite.base import BaseMultiData


class EleTimeMulti(BaseMultiData):
    def __init__(self, path: List[str], max_i_time: int, type: str = "folder"):
        self.max_i_time = max_i_time

        if type == "folder":
            frames = []
            for i, p in enumerate(path):
                ele_single = EleTimeSingle(p, max_i_time, type="folder")
                df = ele_single.to_dataframe()
                df = df.iloc[:max_i_time + 1]
                if i == 0:
                    first = df.copy()
                    ele_columns = [c for c in first.columns if c != "time"]
                    rename_map = {c: f"{c}_No.{i+1}" for c in ele_columns}
                    first = first.rename(columns=rename_map)
                    frames.append(first)
                else:
                    one = df.drop(columns=["time"]).copy()
                    rename_map = {c: f"{c}_No.{i+1}" for c in one.columns}
                    one = one.rename(columns=rename_map)
                    frames.append(one)

            if not frames:
                raise ValueError("path is empty when type is 'folder'")
            data = pd.concat(frames, axis=1)
            n_trajectories = len(path)
        super().__init__(data, max_i_time, "ele", n_trajectories)

    def save_to_csv(self, path):
        self.data.to_csv(path, index=False)

    def save_to_pickle(self, path):
        self.data.to_pickle(path)
