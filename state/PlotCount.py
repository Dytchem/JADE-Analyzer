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


class PlotCount:
    def __init__(
        self,
        state_multi: StateMulti,
        font=fm.FontProperties(fname=str(DEFAULT_FONT_PATH)),
    ):
        self.origin = state_multi
        self.count_state = state_multi.count_state()
        self.font = font

    def plot(
        self,
        save_path=None,
        show=True,
        title=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ):
        plt.figure()
        plt.plot(self.count_state["time"], self.count_state["count_state1"], label="S0")
        plt.plot(self.count_state["time"], self.count_state["count_state2"], label="S1")
        plt.plot(
            self.count_state["time"], self.count_state["count_crash"], label="Crash"
        )
        plt.xlabel("时间 (fs)", fontproperties=self.font)
        plt.ylabel("数量", fontproperties=self.font)
        plt.legend()
        if x_min is not None or x_max is not None:
            plt.xlim(left=x_min, right=x_max)
        if y_min is not None or y_max is not None:
            plt.ylim(bottom=y_min, top=y_max)
        if title is None:
            title = "S0、S1和Crash数量随时间的变化"
        plt.title(title, fontproperties=self.font)
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        if show:
            plt.show()

    def plot_normalized(
        self,
        save_path=None,
        show=True,
        title=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ):
        plt.figure()
        total = self.origin.n
        plt.plot(
            self.count_state["time"],
            self.count_state["count_state1"] / total,
            label="S0",
        )
        plt.plot(
            self.count_state["time"],
            self.count_state["count_state2"] / total,
            label="S1",
        )
        plt.plot(
            self.count_state["time"],
            self.count_state["count_crash"] / total,
            label="Crash",
        )
        plt.xlabel("时间 (fs)", fontproperties=self.font)
        plt.ylabel("比例", fontproperties=self.font)
        plt.legend()
        if x_min is not None or x_max is not None:
            plt.xlim(left=x_min, right=x_max)
        if y_min is not None or y_max is not None:
            plt.ylim(bottom=y_min, top=y_max)
        if title is None:
            title = "S0、S1和Crash占比随时间的变化"
        plt.title(title, fontproperties=self.font)
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        if show:
            plt.show()

    def plot_exp_decay_fit(
        self,
        save_path=None,
        show=True,
        title="S1归一化真实曲线与指数拟合",
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ):
        total = self.origin.n
        if total <= 0:
            raise ValueError("Total trajectory number must be positive")

        time = self.count_state["time"].to_numpy(dtype=float)
        y = self.count_state["count_state2"].to_numpy(dtype=float) / float(total)

        start_idx_candidates = np.where(np.abs(y - 1.0) > 1e-12)[0]
        if len(start_idx_candidates) == 0:
            raise ValueError("Normalized count_state2 never deviates from 1")

        start_idx = int(start_idx_candidates[0])
        t0 = float(time[start_idx])

        t_seg = time[start_idx:]
        y_seg = y[start_idx:]
        valid = np.isfinite(t_seg) & np.isfinite(y_seg) & (y_seg > 0)
        if np.count_nonzero(valid) < 2:
            raise ValueError("Not enough positive points for exponential fitting")

        x = t_seg[valid] - t0
        log_y = np.log(y_seg[valid])
        # Constrained model: y = exp(-k * (t - t0)), ensuring y(t0)=1.
        denom = np.dot(x, x)
        if denom <= 0:
            raise ValueError("Insufficient x-span for constrained exponential fitting")

        slope = np.dot(x, log_y) / denom
        intercept = 0.0
        if slope >= 0:
            raise ValueError(
                "Fitted slope is non-negative, exponential decay is invalid"
            )

        k = float(-slope)
        half_life = float(np.log(2.0) / k)
        fit_hop_time = t0 + half_life

        y_fit = np.full_like(y, np.nan, dtype=float)
        fit_mask = time >= t0
        y_fit[fit_mask] = np.exp(-k * (time[fit_mask] - t0))

        plt.figure(figsize=(8, 6))
        plt.plot(time, y, lw=2, label="真实曲线: count_state2 / n")
        plt.plot(time, y_fit, "--", lw=2, label="指数拟合曲线")
        plt.axvline(t0, color="gray", linestyle=":", linewidth=1.5, label="拟合起点")
        plt.axvline(
            fit_hop_time,
            color="tab:red",
            linestyle=":",
            linewidth=1.5,
            label="拟合退激发时间",
        )

        text = (
            f"t0 = {t0:.2f} fs\n"
            f"半衰期 = {half_life:.2f} fs\n"
            f"拟合退激发时间 = {fit_hop_time:.2f} fs"
        )
        plt.text(
            0.02,
            0.98,
            text,
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            fontproperties=self.font,
        )

        plt.xlabel("时间 (fs)", fontproperties=self.font)
        plt.ylabel("S1 归一化数量", fontproperties=self.font)
        if x_min is not None or x_max is not None:
            plt.xlim(left=x_min, right=x_max)
        if y_min is not None or y_max is not None:
            plt.ylim(bottom=y_min, top=y_max)
        plt.title(title, fontproperties=self.font)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(prop=self.font)
        plt.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()

        return {
            "t0": t0,
            "half_life": half_life,
            "fit_hop_time": fit_hop_time,
            "slope": slope,
            "intercept": intercept,
        }


if __name__ == "__main__":
    path = [f"E:\\GitHub\\JADE-Analyzer\\sample\\{i}" for i in range(1, 201)]
    max_i_time = 500
    state = StateMulti(path, max_i_time)
    print(state.description())

    plot = PlotCount(state)
    plot.plot_normalized()
    plot.plot_exp_decay_fit()
