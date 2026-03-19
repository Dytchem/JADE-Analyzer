import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


class CoordSingle:
    def __init__(self, path, max_i_time, type="folder"):
        self.max_i_time = max_i_time
        if type == "folder":
            sym_frames, coord_frames, time_frames = self._single_coord(path)
            self.data = self._build_dataframe(sym_frames, coord_frames, time_frames)
        elif type == "csv":
            self.data = pd.read_csv(path)
        elif type == "pickle":
            self.data = pd.read_pickle(path)
        else:
            raise ValueError("type must be 'folder' or 'csv' or 'pickle'")
        self.time_interval = self.data["time"][1]

    def _single_coord(self, path):
        sym_frames = []
        coord_frames = []
        time_frames = []
        traj_path = os.path.join(path, "traj_time.out")

        with open(traj_path, "r", encoding="utf-8") as f:
            n = int(f.readline().strip())
            while True:
                sym = []
                coord = []
                flag = False
                frame_time = np.nan
                for i in range(0, n + 2):
                    if i == 0:
                        header = f.readline()
                        header_parts = header.strip().replace("=", " ").split()
                        for part in header_parts:
                            try:
                                frame_time = float(part)
                                break
                            except ValueError:
                                continue
                    elif i <= n:
                        a = f.readline().strip().split()
                        sym.append(a[0])
                        coord.append(np.array([float(a[1]), float(a[2]), float(a[3])]))
                    else:
                        if not f.readline():
                            flag = True
                            break
                sym_frames.append(sym)
                coord_frames.append(coord)
                time_frames.append(frame_time)
                if flag:
                    break

        return sym_frames, coord_frames, time_frames

    def _build_dataframe(self, sym_frames, coord_frames, time_frames):
        if not sym_frames or not coord_frames:
            return pd.DataFrame()

        base_symbols = sym_frames[0]
        columns = []
        axes = ["x", "y", "z"]
        for atom_idx, symbol in enumerate(base_symbols, start=1):
            for axis in axes:
                columns.append(f"{symbol}_{atom_idx}_{axis}")

        frame_count = self.max_i_time + 1
        values = np.full((frame_count, len(columns)), np.nan)

        valid_frames = min(len(coord_frames), frame_count)
        for frame_idx in range(valid_frames):
            flat = []
            for atom_coord in coord_frames[frame_idx]:
                flat.extend(atom_coord.tolist())
            values[frame_idx, :] = flat

        if len(time_frames) > 1 and not np.isnan(time_frames[1]) and time_frames[1] > 0:
            time = np.arange(
                0, time_frames[1] * (self.max_i_time + 0.5), time_frames[1]
            )
        else:
            time = np.arange(0, frame_count, 1.0)

        df = pd.DataFrame(values, columns=columns)
        df.insert(0, "time", time)
        return df

    def save_to_csv(self, path):
        self.data.to_csv(path, index=False)

    def save_to_pickle(self, path):
        self.data.to_pickle(path)


if __name__ == "__main__":
    path = r"E:\GitHub\JADE-Analyzer\sample\1_del"
    max_i_time = 500
    coord = CoordSingle(path, max_i_time)
    coord.save_to_csv(r"E:\GitHub\JADE-Analyzer\output\coord_single.csv")

    print(coord.data)
