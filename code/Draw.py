import matplotlib.pyplot as plt
import numpy as np


def plot_line(base_point, other_points):
    for p in other_points:
        plt.plot([base_point[0], p[0]], [base_point[1], p[1]])


def circle(center, radius, color: str):
    xs = [
        (radius * np.cos(theta) + center[0]) for theta in np.arange(0, 2 * np.pi, 0.01)
    ]
    ys = [
        (radius * np.sin(theta) + center[1]) for theta in np.arange(0, 2 * np.pi, 0.01)
    ]
    plt.plot(xs, ys, color=color)


if __name__ == "__main__":
    center = (0, 0)
    radius = 10
    color = "#FF0000"
    plt.figure(figsize=(8, 8))
    circle(center, radius, color)
    plt.savefig("figure/test.out.png")
