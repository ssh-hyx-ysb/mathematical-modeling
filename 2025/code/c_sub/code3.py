# -*- coding: utf-8 -*-
"""
2025高教社杯C题 - 问题3：综合多因素建模
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report
import warnings

warnings.filterwarnings("ignore")


sns.set_style("whitegrid")
import matplotlib

fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")
fm.addfont("./times.ttf")
# 设置中文字体和负号正常显示
plt.rcParams["font.sans-serif"] = ["FangSong_GB2312", "times"]
plt.rcParams["axes.unicode_minus"] = False

output_dir = Path("C3_Output")
output_dir.mkdir(exist_ok=True)

print("✅ 正在读取数据...")
df = pd.read_csv("C1_Output/clean_male_data.csv")


# 解析孕周
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
df = df.dropna(subset=["孕周", "Y染色体浓度", "孕妇BMI", "年龄", "身高", "体重"])
df["Y达标"] = (df["Y染色体浓度"] >= 0.04).astype(int)

# 定义BMI组
bmi_bins = [20, 28, 32, 36, 40, 50]
bmi_labels = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "≥40"]
df["BMI组"] = pd.cut(
    df["孕妇BMI"], bins=bmi_bins, labels=bmi_labels, include_lowest=True
)

print(f"📊 建模样本数：{len(df)}")

# 构建逻辑回归模型
print(
    "\n🧮 正在构建逻辑回归模型 P(Y≥4%) ~ 孕周 + BMI + 年龄 + 身高 + 体重 + GC含量 ..."
)

# 特征工程
X = df[["孕周", "孕妇BMI", "年龄", "身高", "体重", "GC含量"]]
X = sm.add_constant(X)  # 添加截距
y = df["Y达标"]

# 拟合模型
model = sm.Logit(y, X).fit(disp=False)
print(model.summary())

# 预测概率
df["P_Y达标"] = model.predict(X)

# ROC AUC
auc = roc_auc_score(y, df["P_Y达标"])
print(f"\n✅ 模型AUC = {auc:.3f}")

# 按BMI组分析
print("\n📈 各BMI组逻辑回归预测效果：")
group_auc = df.groupby("BMI组").apply(lambda g: roc_auc_score(g["Y达标"], g["P_Y达标"]))
print(group_auc)

# 绘制：预测概率 vs 实际达标
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="P_Y达标", y="Y达标", alpha=0.6)
plt.xlabel("预测达标概率")
plt.ylabel("实际是否达标 (0/1)")
plt.title("逻辑回归模型预测效果")
plt.tight_layout()
plt.savefig(output_dir / "logit_prediction_scatter.png", dpi=300)
plt.show()

# 为每组生成“达标比例-孕周”曲线
plt.figure(figsize=(12, 8))
weeks = np.arange(10, 26, 0.5)
for label in bmi_labels:
    group_data = df[df["BMI组"] == label]
    if len(group_data) == 0:
        continue

    # 用模型预测该组在不同孕周的达标概率
    sample_row = group_data.iloc[0]
    X_pred = pd.DataFrame(
        {
            "孕周": weeks,
            "孕妇BMI": sample_row["孕妇BMI"],
            "年龄": sample_row["年龄"],
            "身高": sample_row["身高"],
            "体重": sample_row["体重"],
            "GC含量": sample_row["GC含量"],
        }
    )
    X_pred = sm.add_constant(X_pred, has_constant="add")
    prob = model.predict(X_pred)

    # 找到P≥0.9的最小孕周
    try:
        best_week = weeks[np.argmax(prob >= 0.9)]
    except:
        best_week = np.nan

    plt.plot(weeks, prob, label=f"{label} (建议: {best_week:.1f}周)", linewidth=2.5)

plt.axhline(y=0.9, color="r", linestyle="--", label="90%达标概率线")
plt.xlabel("孕周（周）")
plt.ylabel("预测Y浓度≥4%的概率")
plt.title("不同BMI组Y染色体浓度达标概率预测曲线")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "predicted_attainment_curve.png", dpi=300)
plt.show()

# 输出最佳时点建议
best_timing_q3 = {}
for label in bmi_labels:
    group_data = df[df["BMI组"] == label]
    if len(group_data) == 0:
        best_timing_q3[label] = [np.nan, np.nan]
        continue
    sample_row = group_data.iloc[0]
    X_pred = pd.DataFrame(
        {
            "孕周": weeks,
            "孕妇BMI": sample_row["孕妇BMI"],
            "年龄": sample_row["年龄"],
            "身高": sample_row["身高"],
            "体重": sample_row["体重"],
            "GC含量": sample_row["GC含量"],
        }
    )
    X_pred = sm.add_constant(X_pred, has_constant="add")
    prob = model.predict(X_pred)
    try:
        best_week = weeks[np.argmax(prob >= 0.9)]
    except:
        best_week = np.nan
    best_timing_q3[label] = [
        best_week,
        f"{int(best_week)}周{int((best_week-int(best_week))*7)}天",
    ]

best_timing_df = pd.DataFrame(best_timing_q3, index=["最佳时点(周)", "建议时间"]).T
print("\n🎯 问题3：综合模型最佳NIPT时点建议（P≥90%）")
print(best_timing_df)

best_timing_df.to_csv(output_dir / "best_nipt_timing_q3.csv")

# 检测误差影响：加入±5%浓度噪声后重新建模
print("\n🔍 检测误差影响分析...")
np.random.seed(42)
df["Y染色体浓度_噪声"] = df["Y染色体浓度"] * np.random.uniform(0.95, 1.05, len(df))
df["Y达标_噪声"] = (df["Y染色体浓度_噪声"] >= 0.04).astype(int)

model_noisy = sm.Logit(df["Y达标_噪声"], X).fit(disp=False)
df["P_Y达标_噪声"] = model_noisy.predict(X)

noise_auc = roc_auc_score(df["Y达标_噪声"], df["P_Y达标_噪声"])
print(f"加入±5%浓度误差后模型AUC = {noise_auc:.3f}（原为 {auc:.3f}）")

print(f"\n🎉 问题3完成！所有结果已保存至 '{output_dir}' 目录")
