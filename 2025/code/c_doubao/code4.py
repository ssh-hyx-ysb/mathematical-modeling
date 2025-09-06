import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")

# 设置中文显示
import matplotlib

fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")
fm.addfont("./times.ttf")
print(fm)
# 设置中文字体和负号正常显示
plt.rcParams["font.family"] = ["FangSong_GB2312"]
plt.rcParams["axes.unicode_minus"] = False


def process_chromosome_anomaly(triplet):
    """
    解析染色体非整倍体三元组字段
    输入: 三元组字符串(如"100"、"010"、"001"、"110"等)
    输出:
        - 异常标记(1:存在异常, 0:正常)
        - 各染色体异常情况(13号, 18号, 21号)
    """
    if pd.isna(triplet) or len(str(triplet)) != 3:
        return 0, 0, 0, 0  # 无效值视为正常

    triplet_str = str(triplet).strip()
    # 提取每位的异常情况(0:正常, 1:异常)
    chrom13 = int(triplet_str[0]) if triplet_str[0] in ["0", "1"] else 0
    chrom18 = int(triplet_str[1]) if triplet_str[1] in ["0", "1"] else 0
    chrom21 = int(triplet_str[2]) if triplet_str[2] in ["0", "1"] else 0
    # 整体异常标记(只要有一个异常即为1)
    overall_anomaly = 1 if (chrom13 == 1 or chrom18 == 1 or chrom21 == 1) else 0
    return overall_anomaly, chrom13, chrom18, chrom21


def analyze_female_anomaly(female_data):
    """
    女胎染色体非整倍体异常判定完整分析流程
    """
    print("===== 女胎染色体非整倍体异常判定分析 =====")

    # 1. 数据预处理与特征工程
    # 筛选所需字段(确保列名与实际数据一致)
    required_cols = [
        "13号染色体的Z值",
        "18号染色体的Z值",
        "21号染色体的Z值",
        "X染色体的Z值",
        "孕妇BMI",
        "染色体的非整倍体",
    ]
    female_data_q4 = female_data[required_cols].copy()

    # 移除关键特征缺失的样本
    female_data_q4 = female_data_q4.dropna(subset=required_cols)
    print(f"有效分析样本量: {female_data_q4.shape[0]}")

    # 2. 解析染色体异常标记
    # 应用解析函数,生成目标变量
    anomaly_results = female_data_q4["染色体的非整倍体"].apply(
        process_chromosome_anomaly
    )
    female_data_q4[["整体异常标记", "13号异常", "18号异常", "21号异常"]] = pd.DataFrame(
        anomaly_results.tolist(), index=female_data_q4.index
    )

    # 3. 异常分布分析
    print("\n=== 异常分布统计 ===")
    overall_counts = female_data_q4["整体异常标记"].value_counts()
    print(f"正常样本(0): {overall_counts.get(0, 0)}")
    print(f"异常样本(1): {overall_counts.get(1, 0)}")
    print(f"异常比例: {overall_counts.get(1, 0)/female_data_q4.shape[0]:.2%}")

    # 各染色体异常分布
    chrom_counts = {
        "13号染色体": female_data_q4["13号异常"].sum(),
        "18号染色体": female_data_q4["18号异常"].sum(),
        "21号染色体": female_data_q4["21号异常"].sum(),
    }
    print("\n各染色体异常样本数:")
    for chrom, count in chrom_counts.items():
        print(f"{chrom}: {count}")

    # 可视化异常分布
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x="整体异常标记", data=female_data_q4)
    plt.title("整体异常分布")
    plt.xlabel("异常标记(0=正常, 1=异常)")

    plt.subplot(1, 2, 2)
    sns.barplot(x=list(chrom_counts.keys()), y=list(chrom_counts.values()))
    plt.title("各染色体异常样本数")
    plt.ylabel("样本数")
    plt.tight_layout()
    plt.savefig("女胎异常分布统计.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 4. 特征与目标变量定义
    # 特征: 各染色体Z值 + BMI
    X = female_data_q4[
        [
            "13号染色体的Z值",
            "18号染色体的Z值",
            "21号染色体的Z值",
            "X染色体的Z值",
            "孕妇BMI",
        ]
    ]
    # 目标变量: 整体异常标记
    y = female_data_q4["整体异常标记"]

    # 5. 数据拆分(训练集70%, 测试集30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y  # stratify确保分层抽样
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. 模型选择与训练
    # 检查是否为单类问题
    if len(np.unique(y)) < 2:
        print("\n检测到单类别数据, 使用孤立森林(Isolation Forest)进行异常检测")
        # 孤立森林适用于单类异常检测
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,  # 异常比例(可根据实际数据调整)
            random_state=42,
        )
        model.fit(X_train_scaled)

        # 预测异常分数(负值表示异常倾向)
        train_scores = -model.decision_function(X_train_scaled)  # 转为正分数越大越异常
        test_scores = -model.decision_function(X_test_scaled)

        # 确定阈值(使用训练集90%分位数)
        threshold = np.percentile(train_scores, 90)
        y_pred_train = (train_scores > threshold).astype(int)
        y_pred_test = (test_scores > threshold).astype(int)

        # 评估指标(针对异常检测的近似指标)
        metrics = {
            "准确率": accuracy_score(y_train, y_pred_train),
            "测试集准确率": accuracy_score(y_test, y_pred_test),
            "AUC": np.nan,  # 单类问题无法计算AUC
        }

    else:
        print("\n使用随机森林模型进行二分类预测")
        # 随机森林对不平衡数据有较好表现
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",  # 处理类别不平衡
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        y_prob_test = model.predict_proba(X_test_scaled)[:, 1]  # 异常概率

        # 计算多维度评估指标
        metrics = {
            "训练集准确率": accuracy_score(y_train, y_pred_train),
            "测试集准确率": accuracy_score(y_test, y_pred_test),
            "精确率": precision_score(y_test, y_pred_test),
            "召回率": recall_score(y_test, y_pred_test),
            "F1分数": f1_score(y_test, y_pred_test),
            "AUC": roc_auc_score(y_test, y_prob_test),
        }

        # 输出特征重要性
        feature_importance = pd.DataFrame(
            {"特征": X.columns, "重要性": model.feature_importances_}
        ).sort_values("重要性", ascending=False)
        print("\n=== 特征重要性排序 ===")
        print(feature_importance)

        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        sns.barplot(x="重要性", y="特征", data=feature_importance)
        plt.title("特征对异常判定的重要性")
        plt.tight_layout()
        plt.savefig("女胎异常判定特征重要性.png", dpi=300, bbox_inches="tight")
        plt.show()

    # 7. 模型评估结果展示
    print("\n=== 模型评估指标 ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 8. 可视化评估结果
    # 混淆矩阵
    plt.figure(figsize=(10, 6))
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["正常", "异常"],
        yticklabels=["正常", "异常"],
    )
    plt.xlabel("预测结果")
    plt.ylabel("实际结果")
    plt.title("测试集混淆矩阵")
    plt.savefig("女胎异常判定混淆矩阵.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ROC曲线(仅二分类情况)
    if len(np.unique(y)) >= 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {metrics["AUC"]:.4f})')
        plt.plot([0, 1], [0, 1], "k--", label="随机猜测")
        plt.xlabel("假阳性率")
        plt.ylabel("真阳性率")
        plt.title("ROC曲线")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("女胎异常判定ROC曲线.png", dpi=300, bbox_inches="tight")
        plt.show()

    # 异常得分分布
    plt.figure(figsize=(10, 6))
    if len(np.unique(y)) < 2:
        sns.histplot(test_scores, kde=True)
        plt.axvline(
            x=threshold, color="r", linestyle="--", label=f"判定阈值: {threshold:.4f}"
        )
        plt.xlabel("异常得分")
    else:
        sns.histplot(
            data=pd.DataFrame({"异常得分": y_prob_test, "实际标签": y_test}),
            x="异常得分",
            hue="实际标签",
            multiple="stack",
            kde=True,
        )
        plt.axvline(x=0.5, color="r", linestyle="--", label="判定阈值: 0.5")
        plt.xlabel("异常概率")
    plt.title("测试集异常得分分布")
    plt.legend()
    plt.savefig("女胎异常得分分布.png", dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
    }


# 示例调用(需替换为实际数据)
if __name__ == "__main__":
    # 假设female_data是包含所需字段的DataFrame
    # 实际使用时需从Excel读取数据
    # female_data = pd.read_excel("附件.xlsx")

    # 生成示例数据(实际使用时删除)
    np.random.seed(42)
    n_samples = 500
    female_data = pd.DataFrame(
        {
            "13号染色体的Z值": np.random.normal(0, 1, n_samples),
            "18号染色体的Z值": np.random.normal(0, 1, n_samples),
            "21号染色体的Z值": np.random.normal(0, 1, n_samples),
            "X染色体的Z值": np.random.normal(0, 1, n_samples),
            "孕妇BMI": np.random.uniform(18, 30, n_samples),
            "染色体的非整倍体": np.random.choice(
                ["000", "100", "010", "001", "110"],
                size=n_samples,
                p=[0.85, 0.05, 0.05, 0.04, 0.01],
            ),
        }
    )

    # 执行分析
    results = analyze_female_anomaly(female_data)
    print("\n分析完成! 结果已保存为图片文件。")
