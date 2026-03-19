import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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


if __name__ == "__main__":
    path = [f"E:\\GitHub\\JADE-Analyzer\\sample\\{i}" for i in range(1, 201)]
    max_i_time = 500
    state = StateMulti(path, max_i_time)
    print(state.description())

    plot = PlotCount(state)
    plot.plot_normalized()
