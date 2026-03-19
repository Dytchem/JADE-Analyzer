import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from StateMulti import StateMuti


DEFAULT_FONT_PATH = (
    Path(__file__).resolve().parent.parent / "font" / "SourceHanSansSC-Regular.otf"
)


class PlotCount:
    def __init__(
        self,
        state_muti: StateMuti,
        font=fm.FontProperties(fname=str(DEFAULT_FONT_PATH)),
    ):
        self.origin = state_muti
        self.count_state = state_muti.count_state()
        self.font = font

    def plot(self, save_path=None):
        plt.figure()
        plt.plot(self.count_state["time"], self.count_state["count_state1"], label="S0")
        plt.plot(self.count_state["time"], self.count_state["count_state2"], label="S1")
        plt.plot(
            self.count_state["time"], self.count_state["count_crash"], label="Crash"
        )
        plt.xlabel("时间 (fs)", fontproperties=self.font)
        plt.ylabel("数量", fontproperties=self.font)
        plt.legend()
        plt.title("S0、S1和Crash数量随时间的变化", fontproperties=self.font)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    def plot_normalized(self, save_path=None):
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
        plt.title("S0、S1和Crash占比随时间的变化", fontproperties=self.font)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    path = [f"E:\\GitHub\\JADE-Analyzer\\sample\\{i}" for i in range(1, 201)]
    max_i_time = 500
    state = StateMuti(path, max_i_time)

    plot = PlotCount(state)
    plot.plot_normalized()
