import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# 1. 数据加载与预处理
# 假设 female_data 是包含所有数据的DataFrame
fe

# 选择需要的特征列和标签列
feature_cols = [
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值",
    "X染色体的Z值",
    "孕妇BMI",
]
label_col = "染色体的非整倍体"

# 处理缺失值
female_data_q4 = female_data.dropna(subset=feature_cols + [label_col])

# 2. 解析多标签
# 假设'染色体的非整倍体'是三位字符串，如"101"表示13号和21号异常，18号正常
# 提取每个染色体的标签（0或1）
labels = female_data_q4[label_col].astype(str).str.strip()  # 确保是字符串并去除空格

# 检查并处理可能的无效数据（确保字符串长度为3且只包含0和1）
valid_labels = labels.str.match("^[01]{3}$")
female_data_q4 = female_data_q4[valid_labels]
labels = labels[valid_labels]

# 提取每个染色体的标签
y_13 = labels.str[0].astype(int)  # 13号染色体标签 (0:正常, 1:异常)
y_18 = labels.str[1].astype(int)  # 18号染色体标签
y_21 = labels.str[2].astype(int)  # 21号染色体标签

# 组合成多标签y矩阵
y_multi = np.column_stack((y_13, y_18, y_21))

# 3. 准备特征数据
X = female_data_q4[feature_cols]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 检查类分布
print("染色体异常类分布:")
print(
    f"13号染色体: {np.bincount(y_13)} (正常: {np.bincount(y_13)[0]}, 异常: {np.bincount(y_13)[1]})"
)
print(
    f"18号染色体: {np.bincount(y_18)} (正常: {np.bincount(y_18)[0]}, 异常: {np.bincount(y_18)[1]})"
)
print(
    f"21号染色体: {np.bincount(y_21)} (正常: {np.bincount(y_21)[0]}, 异常: {np.bincount(y_21)[1]})"
)

# 5. 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_multi, test_size=0.3, random_state=42, stratify=y_multi
)

# 6. 构建多标签分类模型
# 使用Binary Relevance方法（为每个标签训练一个独立分类器）[6](@ref)
base_clf = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
)
model = MultiOutputClassifier(base_clf, n_jobs=-1)

# 训练模型
model.fit(X_train, y_train)

# 7. 预测与评估
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # 获取每个标签的概率预测

# 多标签评估指标
print("\n===== 多标签评估结果 =====")
print(f"Hamming Loss: {hamming_loss(y_test, y_pred):.4f}")
print(f"子集准确率(Subset Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1-score (micro): {f1_score(y_test, y_pred, average='micro'):.4f}")

# 每个标签的单独评估
chromosomes = ["13号", "18号", "21号"]
print("\n===== 各染色体单独评估 =====")
for i, chr_name in enumerate(chromosomes):
    print(f"\n--- {chr_name}染色体 ---")
    print(f"准确率: {accuracy_score(y_test[:, i], y_pred[:, i]):.4f}")
    print(f"AUC: {roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1]):.4f}")
    print(
        classification_report(y_test[:, i], y_pred[:, i], target_names=["正常", "异常"])
    )

# 8. 可视化结果
# 设置可视化风格
sns.set(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用于显示中文
plt.rcParams["axes.unicode_minus"] = False

# 8.1 绘制每个染色体的ROC曲线
plt.figure(figsize=(10, 8))
for i, chr_name in enumerate(chromosomes):
    fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[i][:, 1])
    auc_score = roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1])
    plt.plot(fpr, tpr, label=f"{chr_name}染色体 (AUC = {auc_score:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="随机分类器")
plt.xlabel("假正率")
plt.ylabel("真正率")
plt.title("各染色体异常判定的ROC曲线")
plt.legend(loc="lower right")
plt.savefig("chromosome_roc_curves.png", dpi=300, bbox_inches="tight")
plt.close()

# 8.2 绘制每个染色体的混淆矩阵
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, chr_name in enumerate(chromosomes):
    cm = confusion_matrix(y_test[:, i], y_pred[:, i])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[i],
        xticklabels=["正常", "异常"],
        yticklabels=["正常", "异常"],
    )
    axes[i].set_title(f"{chr_name}染色体混淆矩阵")
    axes[i].set_xlabel("预测标签")
    axes[i].set_ylabel("真实标签")

plt.tight_layout()
plt.savefig("chromosome_confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.close()

# 8.3 特征重要性分析（使用RandomForest）
plt.figure(figsize=(10, 6))
feature_importances = []
for i, chr_name in enumerate(chromosomes):
    importance = model.estimators_[i].feature_importances_
    feature_importances.append(importance)

feature_importances = np.mean(feature_importances, axis=0)  # 平均各分类器的重要性
features = feature_cols
importance_df = pd.DataFrame({"特征": features, "重要性": feature_importances})
importance_df = importance_df.sort_values("重要性", ascending=False)

sns.barplot(x="重要性", y="特征", data=importance_df, palette="viridis")
plt.title("特征重要性排序（平均 across all chromosomes）")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n===== 特征重要性排序 =====")
print(importance_df)

# 9. 保存预测结果
results_df = female_data_q4.iloc[y_test.index].copy()
results_df["13号预测"] = y_pred[:, 0]
results_df["18号预测"] = y_pred[:, 1]
results_df["21号预测"] = y_pred[:, 2]
results_df["13号概率"] = y_pred_proba[0][:, 1]
results_df["18号概率"] = y_pred_proba[1][:, 1]
results_df["21号概率"] = y_pred_proba[2][:, 1]

# 保存到CSV文件
results_df.to_csv(
    "chromosome_abnormality_predictions.csv", index=False, encoding="utf-8-sig"
)

print("\n预测结果已保存到 'chromosome_abnormality_predictions.csv'")
