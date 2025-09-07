# -*- coding: utf-8 -*-
"""
2025高教社杯C题 - 问题2：NIPT最佳时点建模
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
# 设置中文字体和图形样式
import matplotlib

fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")
fm.addfont("./times.ttf")
# 设置中文字体和负号正常显示
plt.rcParams["font.sans-serif"] = ["FangSong_GB2312", "times"]
plt.rcParams["axes.unicode_minus"] = False


# 创建输出目录
output_dir = Path("C2_Output")
output_dir.mkdir(exist_ok=True)

print("✅ 正在读取清洗后的男胎数据...")
# 假设问题1已生成 clean_male_data.csv
df = pd.read_csv("C1_Output/clean_male_data.csv")

# 确保关键列存在
assert "Y染色体浓度" in df.columns, "缺少Y染色体浓度列"
assert "检测孕周" in df.columns, "缺少检测孕周列"
assert "孕妇BMI" in df.columns, "缺少BMI列"

print(f"📊 当前男胎样本数：{len(df)}")


# 提取孕周数值（如 '16w+4' -> 16 + 4/7 ≈ 16.57）
def parse_gestational_week(gw_str):
    if pd.isna(gw_str):
        return np.nan
    try:
        if "w+" in gw_str:
            week, day = gw_str.split("w+")
            return float(week) + float(day) / 7
        elif "w" in gw_str:
            return float(gw_str.replace("w", ""))
        else:
            return float(gw_str)
    except:
        return np.nan


df["孕周"] = df["检测孕周"].apply(parse_gestational_week)
df = df.dropna(subset=["孕周", "Y染色体浓度", "孕妇BMI"])
df = df[(df["孕周"] >= 8) & (df["孕周"] <= 28)]  # 合理孕周范围

print(f"🧹 清洗后有效样本数：{len(df)}")

# 标记Y浓度是否达标
THRESHOLD = 0.04  # 4%
df["Y达标"] = df["Y染色体浓度"] >= THRESHOLD

# 按孕妇代码分组，找到每个孕妇首次达标的时间
print("🔍 正在计算每位孕妇Y浓度首次达标时间...")
first达标 = (
    df[df["Y达标"]]
    .groupby("孕妇代码")
    .agg({"孕周": "min"})
    .rename(columns={"孕周": "首次达标孕周"})
)

# 合并回原数据
df = df.merge(first达标, on="孕妇代码", how="left")

# 只保留每个孕妇的最早一次检测记录用于分组分析
df = df.sort_values("孕周").groupby("孕妇代码").first().reset_index()

print(f"📌 可用于分组的孕妇数：{len(df)}")

# 定义BMI分组区间
bmi_bins = [20, 28, 32, 36, 40, 100]  # 扩展范围
bmi_labels = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "≥40"]
df["BMI组"] = pd.cut(
    df["孕妇BMI"], bins=bmi_bins, labels=bmi_labels, include_lowest=True
)

# 计算每组的首次达标孕周统计
group_stats = (
    df.groupby("BMI组")
    .agg(
        n_samples=("孕妇代码", "size"),
        mean_attainment=("首次达标孕周", "mean"),
        median_attainment=("首次达标孕周", "median"),
        q80_attainment=("首次达标孕周", lambda x: x.quantile(0.8)),
        q90_attainment=("首次达标孕周", lambda x: x.quantile(0.9)),
    )
    .round(2)
)

# 重命名为中文便于输出
group_stats.columns = [
    "孕妇数量",
    "平均首次达标孕周",
    "中位首次达标孕周",
    "80分位达标孕周",
    "90分位达标孕周",
]
print(group_stats)

# 保存结果
group_stats.to_csv(output_dir / "bmi_group_first_attainment.csv")
df.to_csv(output_dir / "bmi_grouped_data.csv", index=False)

# 绘制：各BMI组首次达标孕周分布
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="BMI组", y="首次达标孕周")
plt.title("各BMI组Y染色体浓度首次≥4%的孕周分布")
plt.ylabel("首次达标孕周（周）")
plt.xlabel("BMI分组")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / "first_attainment_by_bmi.png", dpi=300)
plt.show()

# 绘制：达标比例随孕周变化（按BMI组）
plt.figure(figsize=(12, 7))
for label in bmi_labels:
    data = df[df["BMI组"] == label]
    if len(data) == 0:
        continue
    weeks = np.arange(10, 26, 0.5)
    attainment_ratio = []
    for w in weeks:
        ratio = (data[data["孕周"] <= w]["Y达标"].mean()) * 100
        attainment_ratio.append(ratio)
    plt.plot(weeks, attainment_ratio, label=f"{label}", linewidth=2.5)

plt.axhline(y=90, color="r", linestyle="--", label="90%达标线")
plt.axhline(y=80, color="orange", linestyle="--", label="80%达标线")
plt.xlabel("孕周（周）")
plt.ylabel("Y浓度≥4%的比例（%）")
plt.title("不同BMI组Y染色体浓度达标比例随孕周变化")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "attainment_ratio_by_bmi.png", dpi=300)
plt.show()

# 最佳NIPT时点建议（取80%分位）
best_timing = group_stats[["80分位达标孕周"]].copy()

print("80分位达标孕周（含NaN）：")
print(group_stats["80分位达标孕周"])

best_timing.columns = ["最佳NIPT时点（周）"]
best_timing["建议检测时间"] = best_timing["最佳NIPT时点（周）"].apply(
    lambda x: f"{int(x)}周{int((x-int(x))*7)}天"
)

print("\n🎯 问题2：最佳NIPT时点建议（基于80%分位）")
print(best_timing)

best_timing.to_csv(output_dir / "best_nipt_timing_q2.csv")

# 检测误差影响分析
print("\n🔍 检测误差影响分析...")
# 假设测量误差：±0.5周（时间误差）或 ±5%浓度误差
# 时间误差：最佳时点 ±0.5周
best_timing["时点下限"] = best_timing["最佳NIPT时点（周）"] - 0.5
best_timing["时点上限"] = best_timing["最佳NIPT时点（周）"] + 0.5
print("考虑±0.5周时间误差的置信区间：")
print(best_timing[["最佳NIPT时点（周）", "时点下限", "时点上限"]])

# 浓度误差：若真实浓度有±5%波动，达标时间可能延迟
# 模拟：在原浓度上加噪声，重新计算达标时间
np.random.seed(42)
df_noisy = df.copy()
df_noisy["Y染色体浓度_噪声"] = df_noisy["Y染色体浓度"] * np.random.uniform(
    0.95, 1.05, len(df_noisy)
)
df_noisy["Y达标_噪声"] = df_noisy["Y染色体浓度_噪声"] >= THRESHOLD

first达标_噪声 = df_noisy[df_noisy["Y达标_噪声"]].groupby("孕妇代码")["孕周"].min()
df = df.merge(first达标_噪声, on="孕妇代码", how="left", suffixes=("", "_噪声"))

# 重新计算各组80%分位
noise_impact = df.groupby("BMI组")["首次达标孕周"].quantile(0.8).round(2)
print("\n加入±5%浓度测量误差后，各组80%分位达标时间变化：")
print(noise_impact)

print(f"\n🎉 问题2完成！所有结果已保存至 '{output_dir}' 目录")
