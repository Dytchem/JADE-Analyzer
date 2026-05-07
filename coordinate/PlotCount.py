import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from pathlib import Path

from coordinate import CoordMulti, CountGeom, Geometry


DEFAULT_FONT_PATH = (
    Path(__file__).resolve().parent.parent / "font" / "SourceHanSansSC-Regular.otf"
)


class PlotCount:
    def __init__(
        self,
        count_geom: CountGeom,
        font=fm.FontProperties(fname=str(DEFAULT_FONT_PATH)),
    ):
        self.origin = count_geom
        self.count_data = count_geom.data
        self.font = font

        self.count_columns = [
            c for c in self.count_data.columns if c.startswith("count_")
        ]
        if not self.count_columns:
            raise ValueError("CountGeom.data has no count_ columns to plot")

        plt.rcParams["axes.unicode_minus"] = False

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

        for col in self.count_columns:
            label = col.replace("count_", "")
            plt.plot(self.count_data["time"], self.count_data[col], label=label)

        plt.xlabel("时间 (fs)", fontproperties=self.font)
        plt.ylabel("数量", fontproperties=self.font)
        if x_min is not None or x_max is not None:
            plt.xlim(left=x_min, right=x_max)
        if y_min is not None or y_max is not None:
            plt.ylim(bottom=y_min, top=y_max)
        if title is None:
            title = "几何构型数量随时间变化"
        plt.title(title, fontproperties=self.font)
        plt.legend(prop=self.font)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
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
        if total <= 0:
            raise ValueError("Total trajectory number must be positive")

        for col in self.count_columns:
            label = col.replace("count_", "")
            plt.plot(self.count_data["time"], self.count_data[col] / total, label=label)

        plt.xlabel("时间 (fs)", fontproperties=self.font)
        plt.ylabel("比例", fontproperties=self.font)
        if x_min is not None or x_max is not None:
            plt.xlim(left=x_min, right=x_max)
        if y_min is not None or y_max is not None:
            plt.ylim(bottom=y_min, top=y_max)
        if title is None:
            title = "几何构型占比随时间变化"
        plt.title(title, fontproperties=self.font)
        plt.legend(prop=self.font)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        if show:
            plt.show()


if __name__ == "__main__":
    path = [f"E:\\GitHub\\JADE-Analyzer\\sample\\{i}" for i in range(1, 201)]
    max_i_time = 500

    coord = CoordMulti(path, max_i_time)
    geom = Geometry(coord, "C3", "N2", "N1", "C7")
    count = CountGeom(geom)
    count.save_to_csv("./output/count.csv")

    plot = PlotCount(count)
    plot.plot()
