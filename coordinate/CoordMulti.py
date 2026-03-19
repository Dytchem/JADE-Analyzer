import sys
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from CoordSingle import CoordSingle


class CoordMulti:
    def __init__(self, path, max_i_time, type="folder"):
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
            self.data = pd.concat(frames, axis=1)

        elif type == "csv":
            self.data = pd.read_csv(path)
        elif type == "pickle":
            self.data = pd.read_pickle(path)
        else:
            raise ValueError("type must be 'folder' or 'csv' or 'pickle'")

        self.n = len([c for c in self.data.columns if c.endswith("_No.1")]) // 3
        self.time_interval = self.data["time"][1] - self.data["time"][0]

    def set_time_series(self, time_series):
        time_array = np.asarray(time_series, dtype=float)
        if time_array.ndim != 1:
            raise ValueError("time_series must be a 1D sequence")
        if len(time_array) != len(self.data):
            raise ValueError(
                f"time_series length {len(time_array)} does not match data length {len(self.data)}"
            )

        self.data.loc[:, "time"] = time_array
        self.time_interval = (
            time_array[1] - time_array[0] if len(time_array) > 1 else np.nan
        )

    def save_to_csv(self, path):
        self.data.to_csv(path, index=False)

    def save_to_pickle(self, path):
        self.data.to_pickle(path)


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
