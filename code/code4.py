import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# 输入参数
# -------------------------------
row = 5
length = 50

# -------------------------------
# 生成初始点 x_ps_1 和扰动后点 x_ps_2
# -------------------------------
np.random.seed(42)  # 可重复性

x_ps_1 = []
x_ps_2 = []

for i in range(row, 0, -1):
    # 初始：规则三角晶格（每行居中）
    y_coords = np.arange(-(i - 1) * length / 2, (i - 1) * length / 2 + 0.1, length)
    x_row = (row - i) * length * np.sqrt(3) / 2
    row_init = [(x_row, y) for y in y_coords]
    x_ps_2.append(row_init)

    # 终点：随机扰动 ±10
    row_final = [
        (
            x + np.random.rand() * 10 - np.random.rand() * 10,
            y + np.random.rand() * 10 - np.random.rand() * 10,
        )
        for x, y in row_init
    ]
    x_ps_1.append(row_final)

# -------------------------------
# 插值生成 20 帧的中间路径
# -------------------------------
num_frames = 20
t = np.linspace(0, 1, num_frames)  # 插值参数

# 展平所有点，便于处理
points_init = []
points_final = []
labels = []  # 标记每个点的行和列（用于区分）

idx = 0
for i, (row1, row2) in enumerate(zip(x_ps_1, x_ps_2)):
    for j, (p1, p2) in enumerate(zip(row1, row2)):
        points_init.append(p1)
        points_final.append(p2)
        labels.append(f"Row{i}_Point{j}")
        idx += 1

points_init = np.array(points_init)
points_final = np.array(points_final)
num_points = len(points_init)

# 插值得到每帧位置 (num_frames, num_points, 2)
positions = np.array(
    [(1 - t[frm]) * points_init + t[frm] * points_final for frm in range(num_frames)]
)

# -------------------------------
# 构建 DataFrame
# -------------------------------
data = []

# 添加动画帧（移动过程）
for frame in range(num_frames):
    for i in range(num_points):
        x, y = positions[frame, i]
        data.append(
            {"frame": frame, "x": x, "y": y, "point_id": labels[i], "stage": "moving"}
        )

# 添加初始状态（X 形状）—— 用 frame=-1 表示
for i in range(num_points):
    x, y = points_init[i]
    data.append(
        {"frame": -1, "x": x, "y": y, "point_id": labels[i], "stage": "initial"}
    )

# 添加终止状态（O 形状）—— 用 frame=max+1 表示
for i in range(num_points):
    x, y = points_final[i]
    data.append(
        {"frame": num_frames, "x": x, "y": y, "point_id": labels[i], "stage": "final"}
    )

df = pd.DataFrame(data)
# df = df.sort_values("frame")  # 确保动画顺序正确

# -------------------------------
# 创建动画
# -------------------------------
fig = px.scatter(
    df,
    x="x",
    y="y",
    animation_frame="frame",
    symbol="stage",
    symbol_map={
        "initial": "x",  # 初始点为 X
        "moving": "circle",  # 移动中为 O
        "final": "circle",  # 终点为 O
    },
    color="stage",
    color_discrete_map={"initial": "green", "moving": "blue", "final": "red"},
    hover_name="point_id",
    range_x=[df["x"].min() - 10, df["x"].max() + 10],
    range_y=[df["y"].min() - 10, df["y"].max() + 10],
    title="三角晶格点阵移动动画（初始:X → 终点:O）",
    labels={"stage": "状态", "point_id": "点编号"},
)

# 调整点的大小
fig.update_traces(marker=dict(size=12))

# 添加轨迹线（可选：显示路径）
for i in range(num_points):
    trace_data = df[(df["point_id"] == labels[i]) & (df["stage"] == "moving")]
    fig.add_scatter(
        x=trace_data["x"],
        y=trace_data["y"],
        mode="lines",
        line=dict(color="lightgray", width=1),
        showlegend=False,
        hoverinfo="none",
    )

# 布局美化
fig.update_layout(
    xaxis_title="X",
    yaxis_title="Y",
    legend_title="状态",
    hovermode="closest",
    width=800,
    height=700,
)

# 设置动画速度
if fig.layout.updatemenus:
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 500  # 每帧500ms
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 100

fig.show()
