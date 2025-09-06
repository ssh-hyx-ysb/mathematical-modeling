import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

import matplotlib

fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")
fm.addfont("./times.ttf")
print(fm)
# 设置中文字体和负号正常显示
plt.rcParams["font.sans-serif"] = ["FangSong_GB2312"]
plt.rcParams["axes.unicode_minus"] = False


# 读取数据
print("正在读取数据...")
df = pd.read_excel("../data/附件.xlsx")
print(f"数据形状: {df.shape}")
print("\n前5行数据:")
print(df.head())
print("\n数据基本信息:")
print(df.info())
print("\n数据描述性统计:")
print(df.describe())

# 数据清洗和质量检查
print("=== 数据质量检查 ===")

# 检查缺失值
print("\n缺失值统计:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# 检查重复值
print(f"\n重复行数量: {df.duplicated().sum()}")

# 检查异常值
print("\n异常值检查 (Z-score > 3):")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    outliers = z_scores > 3
    if outliers.any():
        print(f"{col}: {outliers.sum()} 个异常值")

# 数据清洗
print("\n=== 数据清洗 ===")

# 处理缺失值 - 对于数值列用中位数填充，分类列用众数填充
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ["int64", "float64"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

print("缺失值已处理")

# 删除完全重复的行
df_cleaned = df.drop_duplicates()
print(f"清洗后数据形状: {df_cleaned.shape}")

# 保存清洗后的数据
df_cleaned.to_csv("cleaned_data.csv", index=False, encoding="utf-8-sig")
print("清洗后的数据已保存为 'cleaned_data.csv'")


# 1) 绘制Y染色体密度与孕周、BMI的散点图矩阵
print("正在绘制散点图矩阵...")

# 计算孕周（从末次月经到检测日期的周数）
df_cleaned["检测日期"] = pd.to_datetime(df_cleaned["检测日期"], format="%Y%m%d")
df_cleaned["末次月经"] = pd.to_datetime(df_cleaned["末次月经"])
df_cleaned["孕周"] = (df_cleaned["检测日期"] - df_cleaned["末次月经"]).dt.days / 7

# 选择相关变量
scatter_vars = ["Y染色体浓度", "孕周", "孕妇BMI"]
scatter_df = df_cleaned[scatter_vars].dropna()

# 绘制散点图矩阵
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Y染色体浓度与孕周、BMI的相关性分析")

# Y染色体浓度 vs 孕周
axes[0, 0].scatter(scatter_df["孕周"], scatter_df["Y染色体浓度"], alpha=0.6, s=30)
axes[0, 0].set_xlabel("孕周")
axes[0, 0].set_ylabel("Y染色体浓度")
axes[0, 0].grid(True, alpha=0.3)

# Y染色体浓度 vs BMI
axes[0, 1].scatter(
    scatter_df["孕妇BMI"], scatter_df["Y染色体浓度"], alpha=0.6, s=30, color="orange"
)
axes[0, 1].set_xlabel("孕妇BMI")
axes[0, 1].set_ylabel("Y染色体浓度")
axes[0, 1].grid(True, alpha=0.3)

# 孕周 vs BMI
axes[1, 0].scatter(
    scatter_df["孕妇BMI"], scatter_df["孕周"], alpha=0.6, s=30, color="green"
)
axes[1, 0].set_xlabel("孕妇BMI")
axes[1, 0].set_ylabel("孕周")
axes[1, 0].grid(True, alpha=0.3)

# 相关系数矩阵热图（简化版）
corr_matrix = scatter_df.corr()
im = axes[1, 1].imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
axes[1, 1].set_yticklabels(corr_matrix.columns)

# 添加相关系数值
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        axes[1, 1].text(
            j,
            i,
            f"{corr_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

plt.colorbar(im, ax=axes[1, 1])
axes[1, 1].set_title("相关系数矩阵")

plt.tight_layout()
plt.savefig("y_chromosome_correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

print("散点图矩阵已保存为 'y_chromosome_correlation_matrix.png'")

# 2) 绘制不同BMI分组的Y染色体密度随孕周变化曲线
print("正在绘制BMI分组曲线...")

# 创建BMI分组
bmi_bins = [0, 18.5, 24, 28, 50]  # 偏瘦, 正常, 超重, 肥胖
df_cleaned["BMI分组"] = pd.cut(
    df_cleaned["孕妇BMI"], bins=bmi_bins, labels=["偏瘦", "正常", "超重", "肥胖"]
)

# 按BMI分组和孕周分组计算Y染色体浓度的均值
bmi_week_analysis = (
    df_cleaned.groupby(["BMI分组", "孕周"])["Y染色体浓度"].mean().reset_index()
)

plt.figure(figsize=(12, 8))

# 为每个BMI分组绘制曲线
bmi_groups = bmi_week_analysis["BMI分组"].unique()
colors = ["blue", "green", "orange", "red"]

for i, group in enumerate(bmi_groups):
    group_data = bmi_week_analysis[bmi_week_analysis["BMI分组"] == group]
    if len(group_data) > 1:  # 确保有足够的数据点
        plt.plot(
            group_data["孕周"],
            group_data["Y染色体浓度"],
            label=f"{group}",
            color=colors[i],
            linewidth=2.5,
            marker="o",
            markersize=4,
        )

plt.xlabel("孕周", fontsize=14)
plt.ylabel("Y染色体浓度均值", fontsize=14)
plt.title("不同BMI分组的Y染色体浓度随孕周变化趋势", fontsize=16, fontweight="bold")
plt.legend(title="BMI分组", fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig("y_chromosome_by_bmi_week.png", dpi=300, bbox_inches="tight")
plt.show()

print("BMI分组曲线图已保存为 'y_chromosome_by_bmi_week.png'")

# 3) 绘制各染色体Z值的分布直方图
print("正在绘制染色体Z值分布直方图...")

# 选择染色体Z值列
z_value_cols = [
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值",
    "X染色体的Z值",
    "Y染色体的Z值",
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("各染色体Z值分布直方图", fontsize=16, fontweight="bold")

axes = axes.flatten()

for i, col in enumerate(z_value_cols):
    if i < len(axes):
        # 移除极端异常值以更好地显示分布
        data = df_cleaned[col].dropna()
        q1 = data.quantile(0.01)
        q3 = data.quantile(0.99)
        filtered_data = data[(data >= q1) & (data <= q3)]

        axes[i].hist(
            filtered_data, bins=30, alpha=0.7, color=f"C{i}", edgecolor="black"
        )
        axes[i].set_xlabel("Z值")
        axes[i].set_ylabel("频数")
        axes[i].set_title(f"{col}分布")
        axes[i].grid(True, alpha=0.3)

        # 添加统计信息
        mean_val = filtered_data.mean()
        std_val = filtered_data.std()
        axes[i].axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"均值: {mean_val:.2f}",
        )
        axes[i].axvline(mean_val + std_val, color="orange", linestyle=":", linewidth=1)
        axes[i].axvline(mean_val - std_val, color="orange", linestyle=":", linewidth=1)
        axes[i].legend()

# 移除多余的子图
if len(z_value_cols) < len(axes):
    for j in range(len(z_value_cols), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("chromosome_zvalue_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

print("染色体Z值分布直方图已保存为 'chromosome_zvalue_distribution.png'")

# 4) 绘制GC含量、读段数等质量控制指标的分布图（修正版）
print("正在绘制质量控制指标分布图...")

# 修正后的质量控制相关列（注意：'唯一比对的读段数  '后面有空格）
qc_cols = [
    "GC含量",
    "原始读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "唯一比对的读段数  ",
    "被过滤掉读段数的比例",
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("质量控制指标分布图", fontsize=16, fontweight="bold")

axes = axes.flatten()

for i, col in enumerate(qc_cols):
    if i < len(axes):
        data = df_cleaned[col].dropna()

        # 对于比例数据，使用百分比显示
        if "比例" in col or "含量" in col:
            data = data * 100  # 转换为百分比
            xlabel = f"{col.strip()} (%)"  # 去除列名末尾空格
        else:
            xlabel = col.strip()  # 去除列名末尾空格

        axes[i].hist(data, bins=30, alpha=0.7, color=f"C{i+2}", edgecolor="black")
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel("频数")
        axes[i].set_title(f"{col.strip()}分布")  # 去除列名末尾空格
        axes[i].grid(True, alpha=0.3)

        # 添加统计信息
        mean_val = data.mean()
        median_val = data.median()
        axes[i].axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"均值: {mean_val:.2f}",
        )
        axes[i].axvline(
            median_val,
            color="green",
            linestyle="-.",
            linewidth=2,
            label=f"中位数: {median_val:.2f}",
        )
        axes[i].legend()

plt.tight_layout()
plt.savefig("quality_control_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

print("质量控制指标分布图已保存为 'quality_control_distribution.png'")

# First, let's check what files are available in the working directory


print("Files in working directory:")
files = os.listdir(".")
csv_files = [f for f in files if f.endswith(".csv")]
print(csv_files)

# Try to load the data with correct filename
if csv_files:
    # Use the first CSV file found
    data_file = csv_files[0]
    print(f"\nLoading data from: {data_file}")

    # Load the data with proper encoding and check for problematic columns
    df = pd.read_csv(data_file)

    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check data types for the numeric columns we want to use
    numeric_cols_to_check = [
        "年龄",
        "身高",
        "体重",
        "检测孕周",
        "孕妇BMI",
        "原始读段数",
        "在参考基因组上比对的比例",
        "重复读段的比例",
        "唯一比对的读段数  ",
        "GC含量",
        "13号染色体的Z值",
        "18号染色体的Z值",
        "21号染色体的Z值",
        "X染色体的Z值",
        "Y染色体的Z值",
        "Y染色体浓度",
        "X染色体浓度",
        "13号染色体的GC含量",
        "18号染色体的GC含量",
        "21号染色体的GC含量",
        "被过滤掉读段数的比例",
        "怀孕次数",
        "生产次数",
        "孕周",
    ]

    print("\nData types of numeric columns:")
    for col in numeric_cols_to_check:
        if col in df.columns:
            print(f"{col}: {df[col].dtype}")
            # Show sample values if object type
            if df[col].dtype == "object":
                unique_vals = df[col].dropna().unique()
                print(
                    f"  Sample values: {unique_vals[:5] if len(unique_vals) > 5 else unique_vals}"
                )
        else:
            print(f"{col}: Column not found")

    # Check for specific problematic columns
    print("\nChecking '检测孕周' column:")
    if "检测孕周" in df.columns:
        print(f"Unique values in '检测孕周': {df['检测孕周'].dropna().unique()[:10]}")

    print("\nChecking '孕周' column:")
    if "孕周" in df.columns:
        print(f"Unique values in '孕周': {df['孕周'].dropna().unique()[:10]}")
else:
    print("No CSV files found in the directory")

    # Convert problematic columns to numeric values
