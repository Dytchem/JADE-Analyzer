import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path


DEFAULT_FONT_PATH = (
    Path(__file__).resolve().parent.parent / "font" / "SourceHanSansSC-Regular.otf"
)


def read_from_folder_single(path, max_i_time):
    t = []  # 时间序列，单位fs
    state = []  # 状态序列，1表示S0，2表示S1
    t0 = -1  # S1->S0的最终退激发时间，-2bi，-1表示max_i_time时间内没有退激发
    is_crash = False  # 在max_i_time时间内是否发生了碰撞
    with open(os.path.join(path, "hop_all_time.out")) as f:
        i = 1
        while True:
            a = f.readline()
            # print(a)
            if not a:
                break
            if i % 10 == 2:
                b = a.strip().split()
                t.append(float(b[1]))
                state.append(int(b[-1]))
            i += 1
    if len(t) < max_i_time + 1:
        is_crash = True
    if state[-1] == 2:
        t0 = -1
    else:
        for i in range(len(t) - 2, -1, -1):
            if state[i] == 2:
                t0 = t[i + 1]
                break
    return np.array(t), np.array(state), t0, is_crash


def read_from_folder(paths, max_i_time, store_path=None):
    time = np.zeros(max_i_time + 1)
    count_state1 = np.zeros(max_i_time + 1)
    count_state2 = np.zeros(max_i_time + 1)
    t0s = np.zeros(len(paths))
    count_crash = 0
    for i, path in enumerate(paths):
        time, state, t0, is_crash = read_from_folder_single(path, max_i_time)
        count_state1 += (state == 1).astype(int)
        count_state2 += (state == 2).astype(int)
        t0s[i] = t0
        if is_crash:
            count_crash += 1
    t0s.sort()
    if store_path is not None:
        with open(store_path, "wb") as f:
            pickle.dump([time, count_state1, count_state2, t0s, count_crash], f)
    return time, count_state1, count_state2, t0s, count_crash


def read_from_pikle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def plot_count_state(
    total,
    time,
    count_state1,
    count_state2,
    font=fm.FontProperties(fname=str(DEFAULT_FONT_PATH)),
    save_path=None,
):
    plt.figure()
    plt.plot(time, count_state1, label="S0")
    plt.plot(time, count_state2, label="S1")
    plt.plot(time, total - count_state1 - count_state2, label="Crash")
    plt.xlabel("时间 (fs)", fontproperties=font)
    plt.ylabel("数量", fontproperties=font)
    plt.legend()
    plt.title("S0、S1和Crash数量随时间的变化", fontproperties=font)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_t0_hist(
    t0s,
    font=fm.FontProperties(fname=str(DEFAULT_FONT_PATH)),
    save_path=None,
):
    plt.figure()
    plt.hist(t0s[t0s >= 0], bins=50, edgecolor="black")
    plt.xlabel("退激发时间 (fs)", fontproperties=font)
    plt.ylabel("数量", fontproperties=font)
    plt.title("退激发时间的直方图", fontproperties=font)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def describe_information(time, count_state1, count_state2, t0s, count_crash):
    print("总时间:", time[-1])
    print("S0,S1数量相等的时间点:", time[count_state1 == count_state2])
    print("平均退激发时间（不考虑未退激发）:", np.mean(t0s[t0s >= 0]))
    print(
        "平均退激发时间（未退激发算做最大时间）:",
        np.mean(np.where(t0s >= 0, t0s, time[-1])),
    )


if __name__ == "__main__":
    paths = [f"./sample/{i}" for i in range(1, 201)]
    time, count_state1, count_state2, t0s, count_crash = read_from_folder(
        paths, 500, store_path="./sample.pkl"
    )
    print(time)
    print(count_state1)
    print(count_state2)
    print(t0s)
    print(count_crash)

    time1, count_state11, count_state21, t0s1, count_crash1 = read_from_pikle(
        "./sample.pkl"
    )
    print(time == time1)
    print(count_state1 == count_state11)
    print(count_state2 == count_state21)
    print(t0s == t0s1)
    print(count_crash == count_crash1)

    plot_count_state(
        len(paths),
        time,
        count_state1,
        count_state2,
        save_path="./output/count_state.png",
    )
    plot_t0_hist(t0s, save_path="./output/t0_hist.png")
    describe_information(time, count_state1, count_state2, t0s, count_crash)
