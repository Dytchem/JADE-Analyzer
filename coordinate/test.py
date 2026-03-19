import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from pathlib import Path

# === 字体配置（必须在绘图前设置）===
font_path = str(
    Path(__file__).resolve().parent.parent / "font" / "SourceHanSansSC-Regular.otf"
)
zh_font = fm.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
# ================================


def plot_band_statistics(data_list, y_min=-810, y_max=810):
    """
    绘制条带统计图

    参数:
        data_list: list of array-like, 每个元素是一条轨迹的角度数据
        y_min: float, Y 轴显示下限
        y_max: float, Y 轴显示上限

    返回:
        fig: matplotlib figure 对象
        ax: matplotlib axes 对象
        band_counts: dict, 完整统计字典 {条带索引：计数}
    """
    # --- 1. 定义基础参数 ---
    BAND_WIDTH = 180  # 条带宽度 180°
    BAND_START = -90  # 第一个条带从 -90 开始

    # 颜色定义
    color_even = "#8dd3c7"
    color_odd = "#ffffb3"

    # --- 2. 创建图形 ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- 3. 绘制背景色块（仅可视范围） ---
    vis_min_idx = int((y_min - BAND_START) // BAND_WIDTH) - 1
    vis_max_idx = int((y_max - BAND_START) // BAND_WIDTH) + 1

    for band_idx in range(vis_min_idx, vis_max_idx + 1):
        start = BAND_START + band_idx * BAND_WIDTH
        end = start + BAND_WIDTH

        if end <= y_min or start >= y_max:
            continue

        color = color_even if band_idx % 2 == 0 else color_odd
        ax.axhspan(start, end, facecolor=color, alpha=0.3, zorder=0)

    # --- 4. 统计所有数据（不限范围） ---
    band_counts = {}

    for data in data_list:
        if len(data) > 0:
            final_angle = data[-1]
            # 计算所属条带索引
            band_idx = int((final_angle - BAND_START) // BAND_WIDTH)

            # 统计所有数据，不管是否在可视范围内
            if band_idx not in band_counts:
                band_counts[band_idx] = 0
            band_counts[band_idx] += 1

    # --- 5. 绘制轨迹（仅显示可视范围内的部分） ---
    for data in data_list:
        ax.plot(data, linewidth=1.5, alpha=0.8)

    # --- 6. 在可视条带右侧标注计数 ---
    for band_idx, count in band_counts.items():
        start = BAND_START + band_idx * BAND_WIDTH
        center_y = start + BAND_WIDTH / 2

        # 只在可视范围内的条带显示标记
        if center_y < y_min or center_y > y_max:
            continue

        # 确定条带颜色（用于参考，文字用深色）
        color = color_even if band_idx % 2 == 0 else color_odd

        ax.text(
            1.02,  # X 坐标：画布右侧 102% 处
            center_y,  # Y 坐标：条带中心
            f"{count}",  # 显示计数
            verticalalignment="center",
            horizontalalignment="left",
            transform=ax.get_yaxis_transform(),
            color="#333333",  # 深灰色字体
            fontsize=11,
            weight="bold",
            fontproperties=zh_font,  # 【关键】添加中文字体
        )

    # --- 7. 设置坐标轴 ---
    yticks = np.arange(-900, 900, 90)
    ax.set_yticks(yticks)
    ax.set_ylim(y_min, y_max)

    # 【关键】所有文本都要指定 fontproperties
    ax.set_xlabel("时间（fs）", fontproperties=zh_font)
    ax.set_ylabel("CNNC 二面角（°）", fontproperties=zh_font)

    ax.grid(True, linestyle="--", alpha=0.6)

    # 设置刻度标签字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(zh_font)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, ax, band_counts


def load_and_plot(data_dir="/data/huangdeqing/azm/nn/0.000/", n_files=200):
    """
    加载数据并绘制统计图

    参数:
        data_dir: str, 数据目录路径
        n_files: int, 加载的文件数量

    返回:
        fig, ax, band_counts
    """
    # 加载所有轨迹数据
    data_list = []
    print("Loading trajectories...", end=" ")

    for i in range(1, n_files + 1):
        print(i, end=" ")
        data = mol_dihedral(f"{data_dir}{i}", 3, 2, 1, 7)
        data_list.append(data)

    print("\nDone!")

    # 调用绘图函数
    fig, ax, band_counts = plot_band_statistics(data_list)

    # 打印完整统计信息
    print("\n" + "=" * 50)
    print("完整条带统计（所有数据）")
    print("=" * 50)

    total, cis, trans = 0, 0, 0
    for band_idx in reversed(sorted(band_counts.keys())):
        start = -90 + band_idx * 180
        end = start + 180
        count = band_counts[band_idx]
        total += count
        if band_idx % 2 == 0:
            cis += count
        else:
            trans += count
        print(f"条带 {band_idx:3d}: {start:6.0f}° ~ {end:6.0f}°  ->  {count:4d} 条轨迹")

    print("=" * 50)
    print(f"总计: \t{total} ")
    print(f"cis：\t{cis}")
    print(f"trans：\t{trans}")
    print("=" * 50)

    return fig, ax, band_counts
