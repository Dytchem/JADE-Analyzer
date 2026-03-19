import re
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from CoordMulti import CoordMulti
from Geometry import Geometry


DEFAULT_FONT_PATH = (
    Path(__file__).resolve().parent.parent / "font" / "SourceHanSansSC-Regular.otf"
)


class PlotGeom:
    def __init__(
        self,
        geometry: Geometry,
        font=fm.FontProperties(fname=str(DEFAULT_FONT_PATH)),
    ):
        self.origin = geometry
        self.data = geometry.data
        self.kind = geometry.kind
        self.font = font

        plt.rcParams["axes.unicode_minus"] = False

    def _value_columns(self):
        cols = [c for c in self.data.columns if c != "time"]

        def _sort_key(col):
            m = re.search(r"_No\.(\d+)$", col)
            return int(m.group(1)) if m else 10**9

        return sorted(cols, key=_sort_key)

    def _series_list(self):
        return [self.data[c].to_numpy(dtype=float) for c in self._value_columns()]

    @staticmethod
    def _draw_dihedral_bands(ax, y_min, y_max, band_start=-90, band_width=180):
        color_even = "#8dd3c7"
        color_odd = "#ffffb3"

        vis_min_idx = int((y_min - band_start) // band_width) - 1
        vis_max_idx = int((y_max - band_start) // band_width) + 1

        for band_idx in range(vis_min_idx, vis_max_idx + 1):
            start = band_start + band_idx * band_width
            end = start + band_width
            if end <= y_min or start >= y_max:
                continue

            color = color_even if band_idx % 2 == 0 else color_odd
            ax.axhspan(start, end, facecolor=color, alpha=0.3, zorder=0)

    @staticmethod
    def _count_dihedral_bands(data_list, band_start=-90, band_width=180):
        band_counts = {}
        for data in data_list:
            if len(data) == 0:
                continue
            final_angle = data[-1]
            if np.isnan(final_angle):
                continue
            band_idx = int((final_angle - band_start) // band_width)
            band_counts[band_idx] = band_counts.get(band_idx, 0) + 1
        return band_counts

    def _annotate_band_counts(
        self, ax, band_counts, y_min, y_max, band_start=-90, band_width=180
    ):
        for band_idx, count in band_counts.items():
            start = band_start + band_idx * band_width
            center_y = start + band_width / 2
            if center_y < y_min or center_y > y_max:
                continue

            ax.text(
                1.02,
                center_y,
                f"{count}",
                verticalalignment="center",
                horizontalalignment="left",
                transform=ax.get_yaxis_transform(),
                color="#333333",
                fontsize=11,
                weight="bold",
                fontproperties=self.font,
            )

    def _set_labels(self, ax):
        ylabel = {
            "distance": "键长 (Angstrom)",
            "angle": "键角 (deg)",
            "dihedral": "二面角 (deg)",
        }.get(self.kind, "几何参数")

        ax.set_xlabel("时间 (fs)", fontproperties=self.font)
        ax.set_ylabel(ylabel, fontproperties=self.font)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(self.font)

    def print_band_summary(self, band_counts, band_start=-90, band_width=180):
        print("\n" + "=" * 50)
        print("完整条带统计 (所有数据)")
        print("=" * 50)

        total = 0
        even_count = 0
        odd_count = 0

        for band_idx in reversed(sorted(band_counts.keys())):
            start = band_start + band_idx * band_width
            end = start + band_width
            count = band_counts[band_idx]

            total += count
            if band_idx % 2 == 0:
                even_count += count
            else:
                odd_count += count

            print(
                f"条带 {band_idx:3d}: {start:6.0f} deg ~ {end:6.0f} deg  ->  {count:4d} 条轨迹"
            )

        print("=" * 50)
        print(f"总计: {total}")
        print(f"偶数条带: {even_count}")
        print(f"奇数条带: {odd_count}")
        print("=" * 50)

    def plot(self, y_min=None, y_max=None, show_band_stat=True, save_path=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        time = self.data["time"].to_numpy(dtype=float)

        value_cols = self._value_columns()
        data_list = self._series_list()

        for i, (col, values) in enumerate(zip(value_cols, data_list), start=1):
            ax.plot(time, values, linewidth=1.2, alpha=0.8, label=f"No.{i}")

        band_counts = {}
        if self.kind == "dihedral" and show_band_stat:
            y_min = -810 if y_min is None else y_min
            y_max = 810 if y_max is None else y_max

            self._draw_dihedral_bands(ax, y_min, y_max)
            band_counts = self._count_dihedral_bands(data_list)
            self._annotate_band_counts(ax, band_counts, y_min, y_max)

            yticks = np.arange(-900, 901, 90)
            ax.set_yticks(yticks)
            ax.set_ylim(y_min, y_max)
            self.print_band_summary(band_counts)

            plt.tight_layout()
        else:
            if y_min is not None or y_max is not None:
                ax.set_ylim(y_min, y_max)
            plt.tight_layout()

        self._set_labels(ax)
        ax.grid(True, linestyle="--", alpha=0.6)

        if save_path is not None:
            fig.savefig(save_path)

        return fig, ax, band_counts


def load_and_plot(
    paths,
    max_i_time,
    *atoms,
    y_min=None,
    y_max=None,
    show_band_stat=True,
    save_path=None,
):
    coord = CoordMulti(paths, max_i_time)
    geom = Geometry(coord, *atoms)
    plotter = PlotGeom(geom)
    return plotter.plot(
        y_min=y_min,
        y_max=y_max,
        show_band_stat=show_band_stat,
        save_path=save_path,
    )


if __name__ == "__main__":
    path = [f"E:\\GitHub\\JADE-Analyzer\\sample\\{i}" for i in range(1, 201)]
    max_i_time = 500

    coord = CoordMulti(path, max_i_time)
    geom = Geometry(coord, "C3", "N2", "N1", "C7")
    geom.save_to_csv("./output/geom.csv")

    plotter = PlotGeom(geom)
    fig, ax, band_counts = plotter.plot(y_min=-810, y_max=810)
    plt.show()
