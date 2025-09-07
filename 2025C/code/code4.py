# -*coding: utf-8 -*-
"""
问题4：女胎异常判定（修正版）——以AB列为判定标准
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ========================
# 1. 加载并清洗女胎数据
# ========================
# 假设 female_df 已从前面步骤加载
# female_df 来自 '附件 女胎检测数据.xlsx'

# 清理 AB 列：非空即为异常
female_data = pd.read_excel("附件.xlsx", sheet_name="女胎检测数据")


# 提取异常标签：AB列包含T21/T18/T13即为异常
def is_abnormal(ab):
    if pd.isna(ab) or ab.strip() == "":
        return 0
    ab = str(ab).upper()
    if "T13" in ab or "T18" in ab or "T21" in ab:
        return 1
    return 0


female_data["label_abnormal"] = female_data["染色体的非整倍体"].apply(is_abnormal)
print(f"女胎数据总量: {len(female_data)}")
print(f"报告异常数量: {female_data['label_abnormal'].sum()}")

# ========================
# 2. 特征工程
# ========================
# Z值作为核心特征
z_features = ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值"]

# 质量控制特征
qc_features = [
    "GC含量",
    "原始读段数",
    "唯一比对的读段数",
    "在参考基因组上比对的比例",
    "被过滤掉读段数的比例",
    "13号染色体的GC含量",
    "18号染色体的GC含量",
    "21号染色体的GC含量",
]

# 人口统计学
demo_features = ["孕妇BMI", "年龄"]

all_features = z_features + qc_features + demo_features

# 去除缺失值
female_data = female_data.dropna(subset=all_features + ["label_abnormal"])
X = female_data[all_features]
y = female_data["label_abnormal"]

print(f"有效样本量: {len(X)}，其中异常: {y.sum()}")

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
feature_names = X.columns.tolist()

# ========================
# 3. 模型训练与评估
# ========================
# 使用随机森林（可解释性强）
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)

# 交叉验证
cv_f1_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring="f1").mean()
cv_auc_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring="roc_auc").mean()

cv_f1_lr = cross_val_score(lr, X_scaled, y, cv=5, scoring="f1").mean()
cv_auc_lr = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc").mean()

print("\n=== 模型交叉验证性能 ===")
print(f"随机森林  F1: {cv_f1_rf:.3f}, AUC: {cv_auc_rf:.3f}")
print(f"逻辑回归  F1: {cv_f1_lr:.3f}, AUC: {cv_auc_lr:.3f}")

# 训练最终模型
rf.fit(X_scaled, y)
importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": rf.feature_importances_}
).sort_values("importance", ascending=False)

print("\n=== 特征重要性 ===")
print(importance_df.head(10))

# ROC曲线
y_proba = rf.predict_proba(X_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y, y_proba)
auc = roc_auc_score(y, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Female Fetal Aneuploidy Detection")
plt.legend()
plt.grid(True)
plt.show()

# 分类报告
y_pred = rf.predict(X_scaled)
print("\n=== 分类报告 ===")
print(classification_report(y, y_pred, target_names=["正常", "异常"]))


# ========================
# 4. 构建可解释判定规则
# ========================
# 规则1：基于Z值阈值（经典方法）
def simple_z_rule(row):
    z21 = row["21号染色体的Z值"]
    z18 = row["18号染色体的Z值"]
    z13 = row["13号染色体的Z值"]

    # 经典阈值 |Z| > 3
    if abs(z21) > 3 or abs(z18) > 3 or abs(z13) > 3:
        return 1
    return 0


female_data["rule_z3"] = female_data[z_features].apply(simple_z_rule, axis=1)
simple_acc = (female_data["rule_z3"] == y).mean()
print(f"\n经典Z>3规则准确率: {simple_acc:.3f}")


# 规则2：加权Z值（考虑质量）
def quality_weight(row):
    # 质量评分：读段数越多、GC正常、过滤率低 → 分数高
    reads = row["原始读段数"]
    gc = row["GC含量"]
    filter_rate = row["被过滤掉读段数的比例"]

    score = 1.0
    if reads < 4e6:
        score *= 0.8
    if gc < 0.38 or gc > 0.42:
        score *= 0.7
    if filter_rate > 0.03:
        score *= 0.9
    return score


female_data["quality_weight"] = female_data.apply(quality_weight, axis=1)

# 加权Z值：Z_adj = Z * quality_weight
for z_col in z_features:
    w_col = z_col.replace("Z值", "Z值_加权")
    female_data[w_col] = female_data[z_col] * female_data["quality_weight"]


# 新规则：加权Z > 2.8
def weighted_rule(row):
    if (
        abs(row["21号染色体的Z值_加权"]) > 2.8
        or abs(row["18号染色体的Z值_加权"]) > 2.8
        or abs(row["13号染色体的Z值_加权"]) > 2.8
    ):
        return 1
    return 0


female_data["rule_weighted"] = female_data.apply(weighted_rule, axis=1)
weighted_acc = (female_data["rule_weighted"] == y).mean()
print(f"加权Z>2.8规则准确率: {weighted_acc:.3f}")

# ========================
# 5. 输出判定方法（可用于报告）
# ========================
print("\n" + "=" * 50)
print("           女胎异常判定方法")
print("=" * 50)
print("1. 推荐使用随机森林模型预测异常风险，输入以下特征：")
print("   21、18、13、X染色体Z值")
print("   GC含量、读段数、过滤率、BMI等质量指标")
print(f"   模型AUC: {auc:.3f}, F1: {cv_f1_rf:.3f}")
print()
print("2. 简化规则（适用于快速判断）：")
print("   若任一染色体 |Z| > 3 → 判定为异常")
print("   建议结合质量评分调整阈值：")
print("     质量高（读段>4M, GC正常）→ |Z|>3")
print("     质量中 → |Z|>2.8")
print("     质量低 → 建议复测")
print()
print("3. 当检测质量较差时（如低读段数、GC偏移），即使Z值未超标，")
print("   也应标记为‘结果不可靠’，建议重新采样检测。")
