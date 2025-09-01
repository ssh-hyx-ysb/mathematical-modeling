import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import plotly.express as px


def r2xy(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


# 参数
radius = 100
np.random.seed(42)

# 生成数据
plane_series = np.array(
    [
        (np.float64(0), np.float64(0)),
        *[(radius, x) for x in np.arange(0, 2 * np.pi, 40 * 2 * np.pi / 360)],
    ]
)
plane_series_xy = [r2xy(*p) for p in plane_series]


plane_positions_truth = [
    (0, 0),
    (100, 0),
    *[
        (i[0] + np.random.rand() * 20 - 0.5, i[1] + np.random.rand() * 20 - 0.5)
        for i in plane_series_xy[2:]
    ],
]
# 生成模拟数据：初始位置（角度、半径）和调整后位置（角度、半径）
# 角度单位为度，需转换为弧度用于计算
angles_initial = np.array([0, 45, 90, 135, 180, 225, 270, 315])
radii_initial = np.array([95, 90, 92, 88, 90, 85, 88, 90])

angles_adjusted = np.array([0, 45, 90, 135, 180, 225, 270, 315])
radii_adjusted = np.array([80, 85, 82, 78, 80, 75, 78, 80])

# 将角度转换为弧度
theta_initial = np.radians(angles_initial)
theta_adjusted = np.radians(angles_adjusted)

# 创建极坐标图
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="polar")

# 绘制初始位置（用黄色填充的多边形）
ax.fill(theta_initial, radii_initial, color="#FFFF000F", alpha=0.3, label="初始位置")

# 绘制调整后位置（用蓝色填充的多边形）
ax.fill(
    theta_adjusted, radii_adjusted, color="#FF00000F", alpha=0.3, label="调整后位置"
)

# 绘制初始位置的散点
ax.scatter(theta_initial, radii_initial, color="yellow", s=50, marker="o")

# 绘制调整后位置的散点
ax.scatter(theta_adjusted, radii_adjusted, color="blue", s=50, marker="o")

# 设置极坐标的刻度和标签
ax.set_rmax(100)  # 极径最大值
ax.set_rticks(np.arange(20, 101, 20))  # 极径刻度
ax.set_thetagrids(np.arange(0, 360, 45))  # 极角刻度，每45度一个刻度

# 添加标题和图例
ax.set_title("无人机位置变化", fontsize=15)
ax.legend(loc="upper right")

# 显示图形
plt.show()
# 我们取索引 3 到 10（即第4个到第11个点），但注意边界
start_points = plane_positions_truth[3:11]  # 索引 3~10（含头不含尾）→ 8 个点
end_points = plane_series_xy[3:11]  # 同样取 8 个点

assert len(start_points) == len(end_points), "起点和终点数量必须一致"

# 插值生成 20 帧
num_frames = 20
t = np.linspace(0, 1, num_frames)

positions = np.array(
    [
        (1 - t[frm]) * np.array(start_points) + t[frm] * np.array(end_points)
        for frm in range(num_frames)
    ]
)

# 构建主数据 DataFrame
data = []
for frame in range(num_frames):
    for point_id, (x, y) in enumerate(positions[frame]):
        data.append(
            {"frame": frame, "x": x, "y": y, "point_id": point_id, "stage": "moving"}
        )

df = pd.DataFrame(data)

# -------------------------------
# ✅ 使用 pd.concat() 添加初始和终止状态
# -------------------------------

# 创建初始状态数据
initial_data = []
for point_id, (x, y) in enumerate(start_points):
    initial_data.append(
        {"frame": -1, "x": x, "y": y, "point_id": point_id, "stage": "initial"}
    )

# 创建终止状态数据
final_data = []
for point_id, (x, y) in enumerate(end_points):
    final_data.append(
        {"frame": num_frames, "x": x, "y": y, "point_id": point_id, "stage": "final"}
    )

# 转为 DataFrame
df_initial = pd.DataFrame(initial_data)
df_final = pd.DataFrame(final_data)

# 使用 pd.concat 拼接
df = pd.concat([df, df_initial, df_final], ignore_index=True)

# 排序 frame，确保动画顺序
# df = df.sort_values("frame").reset_index(drop=True)

# -------------------------------
# 绘制动画
# -------------------------------
fig = px.scatter(
    df,
    x="x",
    y="y",
    animation_frame="frame",
    symbol="stage",
    symbol_map={"initial": "x", "moving": "circle", "final": "circle"},
    color="stage",
    color_discrete_map={"initial": "green", "moving": "blue", "final": "red"},
    hover_name="point_id",
    range_x=[df["x"].min() - 10, df["x"].max() + 10],
    range_y=[df["y"].min() - 10, df["y"].max() + 10],
    # title="点3-点10移动动画（初始:X → 终点:O）"
)

fig.update_traces(marker=dict(size=12))
fig.update_layout(width=800, height=800)

# 设置动画速度（可选）
if fig.layout.updatemenus:
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 50

fig.show()
