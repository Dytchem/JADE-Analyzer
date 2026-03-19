import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from StateMulti import StateMulti


DEFAULT_FONT_PATH = (
    Path(__file__).resolve().parent.parent / "font" / "SourceHanSansSC-Regular.otf"
)


class PlotDistribution:
    def __init__(
        self,
        state_multi: StateMulti,
        font=fm.FontProperties(fname=str(DEFAULT_FONT_PATH)),
    ):
        self.origin = state_multi
        self.distribution = state_multi.distribution_change()
        self.font = font

    @staticmethod
    def _build_hist_bins(data, bin_width, x_min=None, x_max=None):
        if bin_width <= 0:
            raise ValueError("bin_width must be positive")

        arr = np.asarray(data, dtype=float)
        arr = arr[np.isfinite(arr)]

        if x_min is not None and x_max is not None:
            start, end = float(x_min), float(x_max)
        elif arr.size > 0:
            start = float(np.nanmin(arr)) if x_min is None else float(x_min)
            end = float(np.nanmax(arr)) if x_max is None else float(x_max)
        else:
            start = float(0 if x_min is None else x_min)
            end = float(start + bin_width if x_max is None else x_max)

        if end <= start:
            end = start + bin_width

        return np.arange(start, end + bin_width, bin_width)

    def plot_single(
        self,
        name,
        data,
        normalized=False,
        total=None,
        save_path=None,
        show=True,
        title=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        bin_width=10.0,
    ):
        # print(data)
        plt.figure()
        bins = self._build_hist_bins(
            data, bin_width=bin_width, x_min=x_min, x_max=x_max
        )

        if normalized:
            # Normalize by event count: each sample contributes 1 / total.
            weights = [1 / total] * len(data) if total and total > 0 else None
            plt.hist(data, bins=bins, edgecolor="black", weights=weights)
            plt.ylabel("比例", fontproperties=self.font)
        else:
            plt.hist(data, bins=bins, edgecolor="black")
            plt.ylabel("数量", fontproperties=self.font)
        plt.xlabel("时间 (fs)", fontproperties=self.font)
        if x_min is not None or x_max is not None:
            plt.xlim(left=x_min, right=x_max)
        if y_min is not None or y_max is not None:
            plt.ylim(bottom=y_min, top=y_max)
        if title is None:
            title = f"{name}的直方图"
        plt.title(title, fontproperties=self.font)
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        if show:
            plt.show()

    def plot(
        self,
        hop_save_path=None,
        crash_save_path=None,
        save_path=None,
        show=True,
        hop_title=None,
        crash_title=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        bin_width=10.0,
    ):
        if save_path is not None and hop_save_path is None and crash_save_path is None:
            hop_save_path = save_path
        self.plot_single(
            "退激发",
            self.distribution["hop_time"],
            save_path=hop_save_path,
            show=show,
            title=hop_title,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            bin_width=bin_width,
        )
        self.plot_single(
            "崩溃",
            self.distribution["crash_time"],
            save_path=crash_save_path,
            show=show,
            title=crash_title,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            bin_width=bin_width,
        )

    def plot_normalized(
        self,
        hop_save_path=None,
        crash_save_path=None,
        save_path=None,
        show=True,
        hop_title=None,
        crash_title=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        bin_width=10.0,
    ):
        if save_path is not None and hop_save_path is None and crash_save_path is None:
            hop_save_path = save_path
        total = self.origin.n
        self.plot_single(
            "退激发",
            self.distribution["hop_time"],
            normalized=True,
            total=total,
            save_path=hop_save_path,
            show=show,
            title=hop_title,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            bin_width=bin_width,
        )
        self.plot_single(
            "崩溃",
            self.distribution["crash_time"],
            normalized=True,
            total=total,
            save_path=crash_save_path,
            show=show,
            title=crash_title,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            bin_width=bin_width,
        )


if __name__ == "__main__":
    path = [f"E:\\GitHub\\JADE-Analyzer\\sample\\{i}" for i in range(1, 201)]
    max_i_time = 500
    state = StateMulti(path, max_i_time)

    plot = PlotDistribution(state)
    print(plot.distribution)
    plot.plot_normalized()
