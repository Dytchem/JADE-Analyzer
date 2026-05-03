import sys
from pathlib import Path
from importlib import import_module

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for module_dir in (ROOT_DIR / "state", ROOT_DIR / "coordinate"):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

CoordSingle = import_module("CoordSingle").CoordSingle
StateSingle = import_module("StateSingle").StateSingle

try:
    from ._core import build_geometry_series
    from ._core import extract_hop_time
    from ._core import geometry_label
    from ._core import normalize_geometries
    from ._core import select_value_at_time
except ImportError:
    from _core import build_geometry_series
    from _core import extract_hop_time
    from _core import geometry_label
    from _core import normalize_geometries
    from _core import select_value_at_time


DEFAULT_FONT_PATH = (
    Path(__file__).resolve().parent.parent / "font" / "SourceHanSansSC-Regular.otf"
)


class HopCoordSingle:
    def __init__(
        self,
        path,
        max_i_time,
        *geometries,
        type="folder",
        coord_path=None,
        unwrap_dihedral=True,
    ):
        self.path = path
        self.coord_path = path if coord_path is None else coord_path
        self.max_i_time = max_i_time
        self.geometries = normalize_geometries(*geometries)
        self.unwrap_dihedral = bool(unwrap_dihedral)
        self.geometry_labels = [
            geometry_label(len(atoms), atoms) for atoms in self.geometries
        ]
        self.font = fm.FontProperties(fname=str(DEFAULT_FONT_PATH))

        self.state = StateSingle(path, max_i_time, type=type)
        self.coord = CoordSingle(self.coord_path, max_i_time, type=type)

        self.data = self._build_result_frame()

        plt.rcParams["axes.unicode_minus"] = False

    def _build_result_frame(self):
        state_values = self.state.data["state"].to_numpy(dtype=int)
        state_time = self.state.data["time"].to_numpy(dtype=float)
        hop_time, hop_status = extract_hop_time(
            state_values, state_time, self.state.time_interval
        )

        row = {
            "traj": 1,
            "hop_time": hop_time,
            "hop_status": hop_status,
        }

        coord_time = self.coord.data["time"].to_numpy(dtype=float)
        for atoms, label in zip(self.geometries, self.geometry_labels):
            try:
                geom_series = build_geometry_series(
                    self.coord.data,
                    atoms,
                    unwrap_dihedral=self.unwrap_dihedral,
                )
                (
                    geom_value,
                    geom_time,
                    frame_index,
                    geom_status,
                ) = select_value_at_time(
                    coord_time,
                    geom_series,
                    hop_time,
                )
            except KeyError:
                geom_value = np.nan
                geom_time = np.nan
                frame_index = np.nan
                geom_status = "missing_columns"

            if hop_status == "no_hop" and geom_status == "out_of_range":
                geom_status = "no_hop"

            row[f"{label}_time"] = geom_time
            row[label] = geom_value
            row[f"{label}_status"] = geom_status
            row[f"{label}_frame"] = frame_index

        return pd.DataFrame([row])

    def result(self):
        row = self.data.iloc[0]
        values = [row[label] for label in self.geometry_labels]
        return (row["hop_time"], *values)

    def save_to_csv(self, path):
        self.data.to_csv(path, index=False)

    def save_to_pickle(self, path):
        self.data.to_pickle(path)

    def plot(self, save_path=None, show=True, title=None, y_column=None):
        row = self.data.iloc[0]

        if y_column is None:
            y_column = self.geometry_labels[0]
        if y_column not in self.data.columns:
            raise ValueError(f"Unknown y_column: {y_column}")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([row["hop_time"]], [row[y_column]], s=70, zorder=3)
        ax.text(
            row["hop_time"],
            row[y_column],
            "  No.1",
            fontproperties=self.font,
            va="center",
        )

        ax.set_xlabel("退激发时间 (fs)", fontproperties=self.font)
        ax.set_ylabel(y_column, fontproperties=self.font)
        if title is None:
            title = "单条轨迹的退激发时间-几何参量"
        ax.set_title(title, fontproperties=self.font)
        ax.grid(True, linestyle="--", alpha=0.6)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(self.font)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300)
        if show:
            plt.show()

        return fig, ax


if __name__ == "__main__":
    path = r"E:\GitHub\JADE-Analyzer\sample\1_del"
    max_i_time = 500

    hop_coord = HopCoordSingle(
        path,
        max_i_time,
        ("C3", "N2", "N1", "C7"),
        ("N1", "N2", "C3"),
        unwrap_dihedral=False,
    )
    print(hop_coord.data)
    print(hop_coord.result())
    hop_coord.save_to_csv(r"E:\GitHub\JADE-Analyzer\output\hop_coord_single.csv")
    hop_coord.plot(y_column="angle_N1-N2-C3")
