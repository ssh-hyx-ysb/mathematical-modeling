# -*- coding: utf-8 -*-
"""
NIPT数据分析：基于附件.xlsx完成C题问题1-4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)
from pygam import LinearGAM, s, f
from xgbse import XGBSEKaplanNeighbors
import warnings

warnings.filterwarnings("ignore")

# =================== Step 1: 数据读取与预处理 ===================

# 读取Excel文件（请确保路径正确）
file_path = "附件.xlsx"  # 修改为你的实际路径
try:
    df = pd.read_excel(file_path, sheet_name=0)  # 默认第一个sheet
except FileNotFoundError:
    raise FileNotFoundError("请将 '附件.xlsx' 放入当前目录，或修改 file_path")

print("原始数据形状:", df.shape)
print("\n前5行数据:")
print(df.head())


# 提取孕周（J列）为小数形式：如 "12+3" -> 12.43周
def parse_gestation(gest_str):
    if pd.isna(gest_str):
        return np.nan
    try:
        if "+" in str(gest_str):
            weeks, days = gest_str.split("+")
            return int(weeks) + int(days) / 7
        else:
            return float(gest_str)
    except:
        return np.nan


df["J_float"] = df["J"].apply(parse_gestation)

# 区分男胎与女胎（根据V列是否为空）
df["gender"] = df["V"].apply(lambda x: "Male" if pd.notna(x) and x > 0 else "Female")

# 创建BMI四分类变量（用于问题二初步分组）
df["BMI_group"] = pd.cut(
    df["K"],
    bins=[0, 28, 32, 36, 100],
    labels=["G1: [20,28)", "G2: [28,32)", "G3: [32,36)", "G4: [36,∞)"],
)

# 清洗Z值数据（U列为空是正常的女胎）
df["U"] = pd.to_numeric(df["U"], errors="coerce")  # 转换为空值

# 仅保留AE列有明确健康状态的样本用于验证
df_valid = df.dropna(subset=["AE"]).copy()
df_valid["AE_binary"] = (df_valid["AE"] == "不健康").astype(int)

# 分离男胎与女胎数据集
df_male = df_valid[df_valid["gender"] == "Male"].copy()
df_female = df_valid[df_valid["gender"] == "Female"].copy()

print(f"\n男胎样本数: {len(df_male)}, 女胎样本数: {len(df_female)}")


# =================== 问题一：Y浓度与孕周、BMI关系建模 ===================

print("\n" + "=" * 60)
print("问题一：Y染色体浓度与孕周、BMI关系分析")
print("=" * 60)

# 特征选择
X1 = df_male[["J_float", "K"]]
y1 = df_male["V"]  # Y染色体浓度

# 1. 多元线性回归
model_lr = LinearRegression()
model_lr.fit(X1, y1)
y1_pred_lr = model_lr.predict(X1)
r2_lr = r2_score(y1, y1_pred_lr)
mse_lr = mean_squared_error(y1, y1_pred_lr)

print(f"多元线性回归 R² = {r2_lr:.3f}, MSE = {mse_lr:.4f}")
print(
    f"系数: 截距={model_lr.intercept_:.3f}, 孕周={model_lr.coef_[0]:.3f}, BMI={model_lr.coef_[1]:.3f}"
)

# 2. 广义可加模型 (GAM)
gam = LinearGAM(s(0) + s(1))  # 光滑项：孕周 + BMI
gam.fit(X1.values, y1)
y1_pred_gam = gam.predict(X1.values)
r2_gam = r2_score(y1, y1_pred_gam)

print(f"GAM模型 R² = {r2_gam:.3f}")

# 显著性检验（F检验）
f_test = gam.statistics_["f_stat"]
p_values = gam.statistics_["p_values"]
print(f"孕周 p值: {p_values[0]:.4f}, BMI p值: {p_values[1]:.4f}")

# 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(df_male["J_float"], df_male["V"], alpha=0.6)
plt.plot(
    np.sort(df_male["J_float"]),
    gam.predict(
        pd.DataFrame({"J_float": np.sort(df_male["J_float"]), "K": df_male["K"].mean()})
    ),
    color="red",
)
plt.xlabel("孕周")
plt.ylabel("Y染色体浓度 (%)")
plt.title("Y浓度 vs 孕周 (GAM拟合)")

plt.subplot(1, 2, 2)
plt.scatter(df_male["K"], df_male["V"], alpha=0.6)
plt.plot(
    np.sort(df_male["K"]),
    gam.predict(
        pd.DataFrame({"K": np.sort(df_male["K"]), "J_float": df_male["J_float"].mean()})
    ),
    color="red",
)
plt.xlabel("BMI")
plt.ylabel("Y染色体浓度 (%)")
plt.title("Y浓度 vs BMI (GAM拟合)")
plt.tight_layout()
plt.show()


# =================== 问题二：BMI分组与最佳检测时点 ===================

print("\n" + "=" * 60)
print("问题二：基于BMI分组确定最佳检测时点")
print("=" * 60)

# 计算每组在不同孕周达到Y≥4%的比例
threshold = 4.0
weeks_range = np.arange(12, 18, 0.5)

group_results = {}
for name, group in df_male.groupby("BMI_group"):
    prop_list = []
    for w in weeks_range:
        subset = group[group["J_float"] <= w]
        if len(subset) == 0:
            prop_list.append(0)
            continue
        达标_ratio = (subset["V"] >= threshold).mean()
        prop_list.append(达标_ratio)
    group_results[name] = prop_list

# 找到每组达标率首次≥90%的最早孕周
optimal_times = {}
for name, props in group_results.items():
    for i, p in enumerate(props):
        if p >= 0.9:
            optimal_times[name] = weeks_range[i]
            break
    else:
        optimal_times[name] = weeks_range[-1]

print("各BMI组最佳检测时点（Y≥4%概率≥90%）:")
for g, t in optimal_times.items():
    print(f"{g}: {t:.1f} 周")

# 可视化
plt.figure(figsize=(10, 6))
for name, props in group_results.items():
    plt.plot(weeks_range, props, label=f"{name} (推荐: {optimal_times[name]:.1f}周)")
plt.axhline(0.9, color="red", linestyle="--", label="90% 达标线")
plt.xlabel("孕周")
plt.ylabel("Y浓度≥4%比例")
plt.title("不同BMI组Y浓度达标比例随孕周变化")
plt.legend()
plt.grid(True)
plt.show()


# =================== 问题三：多因素达标比例预测与优化 ===================

print("\n" + "=" * 60)
print("问题三：综合多因素预测Y浓度达标时间")
print("=" * 60)

# 构建生存分析数据：事件为 Y≥4%，时间是孕周
df_male["event"] = (df_male["V"] >= threshold).astype(int)
df_male["time"] = df_male["J_float"]

# 特征工程
features = ["J_float", "K", "C", "D", "E"]
X3 = df_male[features]
y3 = pd.DataFrame({"duration": df_male["time"], "event": df_male["event"]})

# 划分训练集测试集
X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y3, test_size=0.2, random_state=42
)

# 使用XGBoost Survival Embedding
xgbse = XGBSEKaplanNeighbors()
xgbse.fit(X3_train, y3_train)

# 预测风险曲线
y_pred_survival = xgbse.predict(X3_test)
y_pred_time = [np.argmax(s) for s in y_pred_survival.values]

# 计算测试集达标比例
threshold_week = 14
predicted_risk = xgbse.predict_risk(X3_test, at=threshold_week)
achieved = (y3_test["duration"] <= threshold_week) & (y3_test["event"] == 1)
actual_ratio = achieved.mean()
predicted_ratio = (predicted_risk < 0.5).mean()  # 风险低则达标高

print(f"在{threshold_week}周时，实际达标比例: {actual_ratio:.3f}")
print(f"模型预测达标比例: {predicted_ratio:.3f}")

# 特征重要性
importance = xgbse.model.feature_importances_
feat_importance = pd.Series(importance, index=features).sort_values(ascending=False)
print("\n特征重要性:")
print(feat_importance)


# =================== 问题四：女胎异常判定模型 ===================

print("\n" + "=" * 60)
print("问题四：女胎染色体异常分类模型")
print("=" * 60)

# 提取女胎特征
features_female = ["Q", "R", "S", "T", "P", "X", "Y", "Z", "K", "AA", "M", "N", "O"]
df_female_clean = df_female.dropna(subset=features_female + ["AB"]).copy()

# 构建标签：AB列非空即为异常
df_female_clean["abnormal"] = (df_female_clean["AB"].notna()).astype(int)

X4 = df_female_clean[features_female]
y4 = df_female_clean["abnormal"]

print(f"女胎异常样本数: {y4.sum()}, 正常: {len(y4) - y4.sum()}")

# 划分数据集
X4_train, X4_test, y4_train, y4_test = train_test_split(
    X4, y4, test_size=0.2, random_state=42, stratify=y4
)

# 模型1：随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X4_train, y4_train)
y4_pred_rf = rf.predict(X4_test)
y4_pred_proba_rf = rf.predict_proba(X4_test)[:, 1]
acc_rf = accuracy_score(y4_test, y4_pred_rf)
auc_rf = roc_auc_score(y4_test, y4_pred_proba_rf)

print(f"随机森林准确率: {acc_rf:.3f}, AUC: {auc_rf:.3f}")

# 模型2：SVM
svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X4_train, y4_train)
y4_pred_svm = svm.predict(X4_test)
y4_pred_proba_svm = svm.predict_proba(X4_test)[:, 1]
acc_svm = accuracy_score(y4_test, y4_pred_svm)
auc_svm = roc_auc_score(y4_test, y4_pred_proba_svm)

print(f"SVM准确率: {acc_svm:.3f}, AUC: {auc_svm:.3f}")

# 特征重要性（RF）
feat_imp = pd.Series(rf.feature_importances_, index=features_female).sort_values(
    ascending=False
)
print("\n随机森林特征重要性 TOP 5:")
print(feat_imp.head())

# ROC曲线
plt.figure()
fpr_rf, tpr_rf, _ = roc_curve(y4_test, y4_pred_proba_rf)
fpr_svm, tpr_svm, _ = roc_curve(y4_test, y4_pred_proba_svm)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc_svm:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Fetal Abnormality Detection")
plt.legend()
plt.grid(True)
plt.show()


# =================== 输出总结 ===================

print("\n" + "=" * 60)
print("✅ 所有分析完成。")
print("请检查图表与结果。")
print("如需保存模型或结果，可扩展代码。")
