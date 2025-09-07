# -*- coding: utf-8 -*-
"""
C题 问题1：Y染色体浓度与孕周、BMI的相关性建模
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
import os
from pathlib import Path

# 设置中文字体和图形样式
import matplotlib

sns.set(style="whitegrid")
fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")
fm.addfont("./times.ttf")
# 设置中文字体和负号正常显示
plt.rcParams["font.sans-serif"] = ["FangSong_GB2312", "times"]
plt.rcParams["axes.unicode_minus"] = False

# 创建保存图表的目录
output_dir = Path("C1_Output")
output_dir.mkdir(exist_ok=True)

print("✅ 正在读取数据...")

# 读取两个Excel文件
file_male = "附件 - 男胎检测数据.xlsx"
file_female = "附件 - 女胎检测数据.xlsx"

# 分别读取所有sheet或跳过标题行
sheets_male = pd.read_excel(file_male, sheet_name=None, header=None)
sheets_female = pd.read_excel(file_female, sheet_name=None, header=None)

# 合并所有sheet的数据（去除重复标题行）
data_frames = []

for name, df in sheets_male.items():
    # 找到标题行（包含“序号”的行）
    header_row = df[
        df.apply(lambda x: x.astype(str).str.contains("序号", na=False).any(), axis=1)
    ]
    if not header_row.empty:
        idx = header_row.index[0]
        df.columns = df.iloc[idx]
        df = df.iloc[idx + 1 :].reset_index(drop=True)
        data_frames.append(df)

for name, df in sheets_female.items():
    header_row = df[
        df.apply(lambda x: x.astype(str).str.contains("序号", na=False).any(), axis=1)
    ]
    if not header_row.empty:
        idx = header_row.index[0]
        df.columns = df.iloc[idx]
        df = df.iloc[idx + 1 :].reset_index(drop=True)
        data_frames.append(df)

# 合并所有数据
df = pd.concat(data_frames, ignore_index=True)
print(f"📊 合并后总样本数：{len(df)}")

# 保存原始列名以供参考
original_columns = df.columns.tolist()
print("📋 原始列名：", original_columns)

# 转换数据类型
numeric_cols = [
    "年龄",
    "身高",
    "体重",
    "孕妇BMI",
    "原始读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "唯一比对的读段数",
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
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")


# 提取孕周列（格式如 "12w+4"）
def parse_gestational_week(gw_str):
    try:
        if pd.isna(gw_str):
            return np.nan
        gw_str = str(gw_str).strip()
        if "w" in gw_str:
            parts = gw_str.split("w+")
            week = float(parts[0])
            day = float(parts[1]) if len(parts) > 1 else 0
            return week + day / 7
        elif "W" in gw_str:
            parts = gw_str.split("W+")
            week = float(parts[0])
            day = float(parts[1]) if len(parts) > 1 else 0
            return week + day / 7
        else:
            return float(gw_str)
    except:
        return np.nan


df["孕周"] = df["检测孕周"].apply(parse_gestational_week)

# 验证BMI
df["计算BMI"] = df["体重"] / (df["身高"] / 100) ** 2
bmi_diff = np.abs(df["计算BMI"] - df["孕妇BMI"])
print(f"🔍 BMI验证：最大差异 = {bmi_diff.max():.4f}")
if bmi_diff.max() > 0.1:
    print("⚠️ 存在BMI计算不一致，请检查")
df["BMI"] = df["孕妇BMI"]  # 使用原始列

# ==================== 第一步：筛选男胎样本 ====================

print("\n🔍 正在筛选男胎样本（Y染色体浓度非空且 > 0）...")
male_df = df[df["Y染色体浓度"].notna() & (df["Y染色体浓度"] > 0)].copy()
print(f"✅ 男胎样本数量：{len(male_df)}")

# 剔除明显异常数据
print("\n🧹 数据清洗：剔除GC含量异常、测序深度过低样本...")
clean_df = male_df[
    (male_df["GC含量"].between(0.35, 0.65))  # 接近40%-60%
    & (male_df["原始读段数"] > 3_000_000)  # 足够测序深度
    & (male_df["被过滤掉读段数的比例"] < 0.1)  # 过滤比例合理
    & (male_df["孕周"].between(8, 28))  # 孕期合理
    & (male_df["Y染色体浓度"] <= 15)  # 极端高值可能是错误
].copy()

print(f"✅ 清洗后男胎样本数：{len(clean_df)}")

# 保存中间数据
clean_df.to_csv(output_dir / "clean_male_data.csv", index=False, encoding="utf-8-sig")
print(f"💾 清洗后数据已保存至：{output_dir / 'clean_male_data.csv'}")

# ==================== 第二步：探索性数据分析（EDA） ====================

print("\n📈 开始探索性数据分析...")

# 1. 相关性热图
plt.figure(figsize=(10, 8))
corr = clean_df[["Y染色体浓度", "孕周", "BMI", "年龄"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True)
plt.title("Y染色体浓度与各变量相关性热图")
plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)

print("📊 相关性热图已保存")

# 2. 散点图：Y浓度 ~ 孕周，按BMI分组着色
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=clean_df, x="孕周", y="Y染色体浓度", hue="BMI", palette="viridis", alpha=0.7
)
plt.title("Y染色体浓度 vs 孕周（颜色表示BMI）")
plt.xlabel("孕周（周）")
plt.ylabel("Y染色体浓度 (%)")
plt.legend(title="BMI", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(output_dir / "scatter_y_vs_gw_by_bmi.png", dpi=300)

print("📊 散点图（Y vs 孕周）已保存")

# 3. Y浓度分布
plt.figure(figsize=(8, 5))
sns.histplot(clean_df["Y染色体浓度"], kde=True)
plt.title("Y染色体浓度分布")
plt.xlabel("Y染色体浓度 (%)")
plt.tight_layout()
plt.savefig(output_dir / "hist_y_concentration.png", dpi=300)

print("📊 Y浓度分布图已保存")

# 4. 孕周分布
plt.figure(figsize=(8, 5))
sns.histplot(clean_df["孕周"], kde=True, color="skyblue")
plt.title("孕周分布")
plt.xlabel("孕周（周）")
plt.tight_layout()
plt.savefig(output_dir / "hist_gestational_week.png", dpi=300)

print("📊 孕周分布图已保存")

# ==================== 第三步：建立回归模型 ====================

print("\n🧮 建立回归模型...")

# 准备建模数据
model_data = clean_df[["Y染色体浓度", "孕周", "BMI"]].dropna().copy()
print(f"📊 建模样本数：{len(model_data)}")

# 添加交互项和二次项
model_data["孕周_2"] = model_data["孕周"] ** 2
model_data["孕周_BMI"] = model_data["孕周"] * model_data["BMI"]

# 模型1：线性模型
print("\n➡️ 模型1：线性回归 Y ~ 孕周 + BMI")
X1 = sm.add_constant(model_data[["孕周", "BMI"]])
y = model_data["Y染色体浓度"]
model1 = sm.OLS(y, X1).fit()
print(model1.summary())
with open(output_dir / "model1_summary.txt", "w", encoding="utf-8") as f:
    f.write(model1.summary().as_text())
print("📄 线性模型结果已保存")

# 模型2：含交互项和二次项
print("\n➡️ 模型2：多项式 + 交互项 Y ~ 孕周 + 孕周² + BMI + 孕周:BMI")
X2 = sm.add_constant(model_data[["孕周", "孕周_2", "BMI", "孕周_BMI"]])
model2 = sm.OLS(y, X2).fit()
print(model2.summary())
with open(output_dir / "model2_summary.txt", "w", encoding="utf-8") as f:
    f.write(model2.summary().as_text())
print("📄 多项式模型结果已保存")

# ==================== 残差诊断图（手动绘制）====================
print("\n📊 正在绘制残差诊断图...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 残差 vs 拟合值
axes[0, 0].scatter(model2.fittedvalues, model2.resid, alpha=0.6)
axes[0, 0].hlines(
    0,
    model2.fittedvalues.min(),
    model2.fittedvalues.max(),
    colors="r",
    linestyles="dashed",
)
axes[0, 0].set_xlabel("拟合值")
axes[0, 0].set_ylabel("残差")
axes[0, 0].set_title("残差 vs 拟合值（检验异方差）")

# 2. Q-Q 图（检验正态性）
residuals_norm = (model2.resid - model2.resid.mean()) / model2.resid.std()
stats.probplot(residuals_norm, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title("Q-Q 图（检验残差正态性）")

# 3. Y浓度 vs 孕周（实际 vs 拟合）
axes[1, 0].scatter(
    model_data["孕周"],
    model_data["Y染色体浓度"],
    alpha=0.6,
    label="实际值",
    color="blue",
)
sorted_idx = np.argsort(model_data["孕周"])
axes[1, 0].plot(
    model_data["孕周"].iloc[sorted_idx],
    model2.fittedvalues.iloc[sorted_idx],
    color="red",
    label="拟合曲线",
)
axes[1, 0].set_xlabel("孕周")
axes[1, 0].set_ylabel("Y染色体浓度 (%)")
axes[1, 0].set_title("Y浓度 vs 孕周（拟合效果）")
axes[1, 0].legend()

# 4. Y浓度 vs BMI
axes[1, 1].scatter(
    model_data["BMI"], model_data["Y染色体浓度"], alpha=0.6, color="green"
)
axes[1, 1].set_xlabel("BMI")
axes[1, 1].set_ylabel("Y染色体浓度 (%)")
axes[1, 1].set_title("Y浓度 vs BMI")

plt.tight_layout()
plt.savefig(output_dir / "residual_diagnostics.png", dpi=300)

print("✅ 残差诊断图已保存")

# ==================== 第四步：混合效应模型（考虑孕妇个体差异）====================

print("\n🔁 建立混合效应模型（随机截距）...")

# 添加孕妇ID
model_data_with_id = (
    clean_df[["Y染色体浓度", "孕周", "BMI", "孕妇代码"]].dropna().copy()
)
model_data_with_id["孕周_2"] = model_data_with_id["孕周"] ** 2
model_data_with_id["孕周_BMI"] = model_data_with_id["孕周"] * model_data_with_id["BMI"]

# 使用 statsmodels 的 MixedLM（仅支持随机截距）
try:
    mixed_model = mixedlm(
        "Y染色体浓度 ~ 孕周 + np.power(孕周, 2) + BMI + 孕周:BMI",
        model_data_with_id,
        groups=model_data_with_id["孕妇代码"],
    )
    mixed_result = mixed_model.fit()
    print(mixed_result.summary())
    with open(output_dir / "mixed_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(mixed_result.summary().as_text())
    print("📄 混合效应模型结果已保存")
except Exception as e:
    print(f"⚠️ 混合模型拟合失败：{e}")

# ==================== 第五步：显著性检验汇总 ====================

print("\n" + "=" * 50)
print("✅ 问题1 结果汇总")
print("=" * 50)

print(f"🔹 样本总数：{len(df)}")
print(f"🔹 男胎有效样本数：{len(clean_df)}")
print(f"🔹 最终建模样本数：{len(model_data)}")

print("\n🔹 线性模型关键结果：")
print(f"   R² = {model1.rsquared:.4f}, F-statistic p-value = {model1.f_pvalue:.2e}")
print(f"   孕周系数 = {model1.params['孕周']:.4f} (p = {model1.pvalues['孕周']:.2e})")
print(f"   BMI系数 = {model1.params['BMI']:.4f} (p = {model1.pvalues['BMI']:.2e})")

print("\n🔹 多项式+交互模型关键结果：")
print(f"   R² = {model2.rsquared:.4f}, F-statistic p-value = {model2.f_pvalue:.2e}")
print(f"   孕周系数 = {model2.params['孕周']:.4f} (p = {model2.pvalues['孕周']:.2e})")
print(f"   BMI系数 = {model2.params['BMI']:.4f} (p = {model2.pvalues['BMI']:.2e})")
print(
    f"   交互项系数 = {model2.params['孕周_BMI']:.4f} (p = {model2.pvalues['孕周_BMI']:.2e})"
)

# 显著性结论
alpha = 0.05
print(f"\n🔹 显著性结论（α = {alpha}）：")
if model2.pvalues["孕周"] < alpha:
    print("   ✅ 孕周对Y浓度有显著正向影响")
else:
    print("   ❌ 孕周无显著影响")

if model2.pvalues["BMI"] < alpha:
    print("   ✅ BMI对Y浓度有显著负向影响")
else:
    print("   ❌ BMI无显著影响")

if model2.pvalues["孕周_BMI"] < alpha:
    print("   ✅ 孕周与BMI存在显著交互作用")
else:
    print("   ❌ 无显著交互作用")

print(f"\n🎉 所有中间步骤已完成，图表与结果已保存至 '{output_dir}' 目录")
