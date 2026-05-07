from pathlib import Path
from typing import Tuple, Optional

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from state import StateMulti
from coordinate import CoordMulti
from ._core import build_geometry_series, extract_hop_time, geometry_label, normalize_geometries, select_value_at_time


DEFAULT_FONT_PATH = (
    Path(__file__).resolve().parent.parent / "font" / "SourceHanSansSC-Regular.otf"
)


class HopCoordMulti:
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

        self.state = StateMulti(path, max_i_time, type=type)
        self.coord = CoordMulti(self.coord_path, max_i_time, type=type)

        self.data = self._build_result_frame()

        plt.rcParams["axes.unicode_minus"] = False

    def _build_result_frame(self):
        time_values = self.state.data["time"].to_numpy(dtype=float)
        coord_time = self.coord.data["time"].to_numpy(dtype=float)
        result_rows = []

        for traj_index in range(1, self.state.n + 1):
            state_col = f"state_No.{traj_index}"

            state_values = self.state.data[state_col].to_numpy(dtype=int)
            hop_time, hop_status = extract_hop_time(
                state_values, time_values, self.state.time_interval
            )

            row = {
                "traj": traj_index,
                "hop_time": hop_time,
                "hop_status": hop_status,
            }

            for atoms, label in zip(self.geometries, self.geometry_labels):
                try:
                    geom_series = build_geometry_series(
                        self.coord.data,
                        atoms,
                        traj_index=traj_index,
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

            result_rows.append(row)

        return pd.DataFrame(result_rows)

    def result(self):
        return self.data.copy()

    def save_to_csv(self, path):
        self.data.to_csv(path, index=False)

    def save_to_pickle(self, path):
        self.data.to_pickle(path)

    def plot_pair(
        self,
        x_column,
        y_column,
        save_path=None,
        show=True,
        title=None,
        annotate=True,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ):
        if x_column not in self.data.columns:
            raise ValueError(f"Unknown x_column: {x_column}")
        if y_column not in self.data.columns:
            raise ValueError(f"Unknown y_column: {y_column}")

        fig, ax = plt.subplots(figsize=(7, 5))
        valid = self.data[x_column].notna() & self.data[y_column].notna()
        plot_df = self.data.loc[valid].copy()

        ax.scatter(plot_df[x_column], plot_df[y_column], s=36, alpha=0.85)

        if annotate:
            for _, row in plot_df.iterrows():
                ax.text(
                    row[x_column],
                    row[y_column],
                    f"  {int(row['traj'])}",
                    fontproperties=self.font,
                    va="center",
                    fontsize=9,
                )

        ax.set_xlabel(x_column, fontproperties=self.font)
        ax.set_ylabel(y_column, fontproperties=self.font)
        if title is None:
            title = "多条轨迹参数散点图"
        ax.set_title(title, fontproperties=self.font)
        ax.grid(True, linestyle="--", alpha=0.6)

        if x_min is not None or x_max is not None:
            ax.set_xlim(left=x_min, right=x_max)
        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(self.font)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300)
        if show:
            plt.show()

        return fig, ax

    def plot(
        self,
        save_path=None,
        show=True,
        title=None,
        annotate=True,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ):
        return self.plot_pair(
            x_column="hop_time",
            y_column=self.geometry_labels[0],
            save_path=save_path,
            show=show,
            title=title,
            annotate=annotate,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    def plot_hop_vs(self, y_column, **kwargs):
        return self.plot_pair("hop_time", y_column, **kwargs)


if __name__ == "__main__":
    path = [f"E:\\GitHub\\JADE-Analyzer\\sample\\{i}" for i in range(1, 201)]
    max_i_time = 500

    hop_coord = HopCoordMulti(
        path,
        max_i_time,
        ("C3", "N2", "N1", "C7"),
        ("N1", "N2", "C3"),
        unwrap_dihedral=False,
    )
    print(hop_coord.data)
    hop_coord.save_to_csv(r"E:\GitHub\JADE-Analyzer\output\hop_coord_multi.csv")
    hop_coord.plot_hop_vs("angle_N1-N2-C3")
    hop_coord.plot_pair("dihedral_C3-N2-N1-C7", "angle_N1-N2-C3")
