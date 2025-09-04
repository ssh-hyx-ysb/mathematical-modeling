import matplotlib.pyplot as plt
import numpy as np


import matplotlib

# 添加中文字体包（华文宋体）
fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")
fm.addfont("./STSONG.TTF")

# 设置中文字体和负号正常显示
plt.rcParams["font.sans-serif"] = ["仿宋_GB2312.TTF", "STSONG"]
plt.rcParams["axes.unicode_minus"] = False


def line(
    Point1: np.ndarray,
    Point2: np.ndarray,
    points: int,
    Color: np.ndarray | str,
    title: str,
    x_range_l: int,
    x_range_r: int,
    y_range_l: int,
    y_range_r: int,
):
    Xs = np.arange(Point1[0], Point2[0], (Point1[0] - Point2[0]) / points)
    Ys = np.arange(Point1[1], Point2[1], (Point1[1] - Point2[1] / points))
    plt.plot([Xs, Ys], color=Color)
    plt.xlim([x_range_l, x_range_r])
    plt.ylim([y_range_l, y_range_r])
    plt.title(title)
    plt.legend()
    plt.grid(True)
    pass


def circle():
    pass
