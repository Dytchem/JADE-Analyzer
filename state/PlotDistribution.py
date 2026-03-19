import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from state.StateMulti import StateMuti


class PlotDistribution:
    def __init__(
        self,
        state_muti: StateMuti,
        font=fm.FontProperties(
            fname=r"E:\GitHub\JADE-Analyzer\SourceHanSansSC-Regular.otf"
        ),
    ):
        self.origin = state_muti
        self.distribution = state_muti.distribution_change()
        self.font = font

    def plot_single(self, name, data, normalized=False, total=None):
        plt.figure()
        if normalized:
            # Normalize by event count: each sample contributes 1 / total.
            weights = [1 / total] * len(data) if total and total > 0 else None
            plt.hist(data, bins=50, edgecolor="black", weights=weights)
            plt.ylabel("比例", fontproperties=self.font)
        else:
            plt.hist(data, bins=50, edgecolor="black")
            plt.ylabel("数量", fontproperties=self.font)
        plt.xlabel("时间 (fs)", fontproperties=self.font)
        plt.title(f"{name}的直方图", fontproperties=self.font)
        plt.show()

    def plot(self):
        self.plot_single("退激发", self.distribution["hop_time"])
        self.plot_single("崩溃", self.distribution["crash_time"])

    def plot_normalized(self):
        total = self.origin.n
        self.plot_single(
            "退激发", self.distribution["hop_time"], normalized=True, total=total
        )
        self.plot_single(
            "崩溃", self.distribution["crash_time"], normalized=True, total=total
        )


if __name__ == "__main__":
    path = [f"E:\\GitHub\\JADE-Analyzer\\sample\\{i}" for i in range(1, 201)]
    max_i_time = 500
    state = StateMuti(path, max_i_time)

    plot = PlotDistribution(state)
    print(plot.distribution)
    plot.plot_normalized()
