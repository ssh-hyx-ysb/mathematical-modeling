import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
sns.set_style(
    "whitegrid", {"axes.facecolor": "#F0F0F0", "axes.grid": True, "axes.linewidth": 0.5}
)
fm = matplotlib.font_manager.fontManager
fm.addfont("2025C/code/仿宋_GB2312.TTF")
fm.addfont("2025C/code/times.ttf")
print(fm)
plt.rcParams["font.sans-serif"] = ["FangSong_GB2312", "times"]
plt.rcParams["axes.unicode_minus"] = False


# 读取女胎数据
female_data = pd.read_excel("2025C/code/附件.xlsx", sheet_name="女胎检测数据")


# 解析'染色体的非整倍体'为三元组标签 [13,18,21]
def parse_abnormalities(label):
    if pd.isna(label) or label == "":
        return [0, 0, 0]  # 正常
    labels = [0, 0, 0]
    if "T13" in label or "13" in label:  # 兼容'T13'或数字
        labels[0] = 1
    if "T18" in label or "18" in label:
        labels[1] = 1
    if "T21" in label or "21" in label:
        labels[2] = 1
    return labels


female_data["abnormalities"] = female_data["染色体的非整倍体"].apply(
    parse_abnormalities
)

# 数据清洗
female_data_q4 = female_data.dropna(
    subset=[
        "13号染色体的Z值",
        "18号染色体的Z值",
        "21号染色体的Z值",
        "X染色体的Z值",
        "13号染色体的GC含量",
        "18号染色体的GC含量",
        "21号染色体的GC含量",
        "孕妇BMI",
        "唯一比对的读段数",
        "被过滤掉读段数的比例",
        "染色体的非整倍体",
    ]
)

# 特征列表
features = [
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值",
    "X染色体的Z值",
    "13号染色体的GC含量",
    "18号染色体的GC含量",
    "21号染色体的GC含量",
    "孕妇BMI",
    "唯一比对的读段数",
    "被过滤掉读段数的比例",
]

X = female_data_q4[features]
y = np.array(female_data_q4["abnormalities"].tolist())  # 形状: (n_samples, 3)

# 检查类分布
class_dist = [np.bincount(y[:, i]) for i in range(3)]
print(f"13号类分布: {class_dist[0]}")
print(f"18号类分布: {class_dist[1]}")
print(f"21号类分布: {class_dist[2]}")

# 训练三个独立随机森林分类器
models = []
accuracies = []
aucs = []
conf_matrices = []
feature_importances_list = []

for i in range(3):  # 13,18,21
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y[:, i],
        test_size=0.2,
        random_state=42,
        stratify=y[:, i] if len(np.unique(y[:, i])) > 1 else None,
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced" if len(np.unique(y_train)) > 1 else None,
    )
    rf.fit(X_train, y_train)
    models.append(rf)

    y_pred = rf.predict(X_test)
    y_prob = (
        rf.predict_proba(X_test)[:, 1]
        if len(np.unique(y_train)) > 1
        else np.full(len(y_test), 0.5)
    )  # 单类处理

    accuracies.append(accuracy_score(y_test, y_pred))
    aucs.append(roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan)
    conf_matrices.append(confusion_matrix(y_test, y_pred))
    feature_importances_list.append(rf.feature_importances_)

# 平均特征重要性
avg_importance = np.mean(feature_importances_list, axis=0)

# 可视化特征重要性
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_importance, y=features, palette="viridis")
plt.title("平均特征重要性 (13/18/21号染色体)")
plt.xlabel("重要性")
plt.ylabel("特征")
plt.savefig("feature_importance.png")
plt.close()

# 可视化混淆矩阵 (示例13号)
for i, cm in enumerate(conf_matrices):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'混淆矩阵 - {["13号", "18号", "21号"][i]}染色体')
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.savefig(f'confusion_matrix_{["13", "18", "21"][i]}.png')
    plt.close()

# 输出结果
print(f"准确率 (13/18/21): {accuracies}")
print(f"AUC (13/18/21): {aucs}")


# 判定方法示例：输入新样本，返回异常概率
def predict_abnormal(new_sample):
    probs = [rf.predict_proba(pd.DataFrame([new_sample]))[:, 1][0] for rf in models]
    return {
        "13号异常概率": probs[0],
        "18号异常概率": probs[1],
        "21号异常概率": probs[2],
    }


# 测试示例
new_sample = {f: np.random.rand() for f in features}
print(predict_abnormal(new_sample))
