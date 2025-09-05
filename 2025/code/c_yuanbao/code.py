# -*- coding: utf-8 -*-
"""
NIPT检测数据建模分析：时点选择与异常判定
针对2025年高教社杯数学建模C题
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_squared_error,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体（可选，便于图表显示中文）
import matplotlib

fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")
fm.addfont("./times.ttf")
print(fm)
# 设置中文字体和负号正常显示
plt.rcParams["font.sans-serif"] = ["FangSong_GB2312", "times"]
plt.rcParams["axes.unicode_minus"] = False


# 1. 数据加载与初步探索
def load_and_explore_data(file_path):
    """
    加载数据并进行初步探索
    """
    print("正在加载数据...")
    df = pd.read_excel(file_path)
    print(f"数据形状: {df.shape}")
    print("\n数据列名及类型:")
    print(df.dtypes)
    print("\n缺失值统计:")
    print(df.isnull().sum())
    print("\n数据前5行:")
    print(df.head())
    return df


# 2. 数据预处理函数
def preprocess_data(df):
    """
    数据清洗和预处理
    """
    df_clean = df.copy()

    # 解析孕周（示例：将'12+3'解析为12.4286周）
    def parse_gestational_weeks(gest_str):
        try:
            if isinstance(gest_str, str) and "+" in gest_str:
                weeks, days = gest_str.split("+")
                return float(weeks) + float(days) / 7.0
            else:
                return float(gest_str)
        except:
            return np.nan

    df_clean["孕周_数值"] = df_clean["检测孕周"].apply(parse_gestational_weeks)

    # 分离男胎和女胎数据
    # 假设Y染色体浓度>0为男胎，否则为女胎（根据数据说明调整）
    male_data = df_clean[df_clean["Y染色体浓度"] > 0].copy()
    female_data = df_clean[df_clean["Y染色体浓度"] <= 0].copy()  # 或根据实际情况判断

    print(f"男胎样本数: {len(male_data)}")
    print(f"女胎样本数: {len(female_data)}")

    # 处理其他可能的缺失值或异常值（根据实际数据情况调整）
    # 例如，对关键数值列的异常值进行过滤
    numeric_cols = [
        "孕妇BMI",
        "孕周_数值",
        "Y染色体浓度",
        "13号染色体的Z值",
        "18号染色体的Z值",
        "21号染色体的Z值",
        "GC含量",
    ]
    for col in numeric_cols:
        if col in df_clean.columns:
            q_low = df_clean[col].quantile(0.01)
            q_high = df_clean[col].quantile(0.99)
            df_clean = df_clean[(df_clean[col] >= q_low) & (df_clean[col] <= q_high)]

    return df_clean, male_data, female_data


# 3. 探索性数据分析 (EDA)
def perform_eda(df, male_data, female_data):
    """
    执行探索性数据分析并生成可视化图表
    """
    # 绘制关键变量的分布
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # 示例：孕妇BMI分布
    sns.histplot(df["孕妇BMI"].dropna(), kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("孕妇BMI分布")
    # 示例：孕周分布
    sns.histplot(df["孕周_数值"].dropna(), kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("孕周分布")
    # 示例：Y染色体浓度分布 (男胎)
    sns.histplot(male_data["Y染色体浓度"].dropna(), kde=True, ax=axes[0, 2])
    axes[0, 2].set_title("男胎Y染色体浓度分布")
    # ... 可以添加更多图表，如Z值分布、GC含量分布等
    plt.tight_layout()
    plt.savefig("eda_distributions.png")
    plt.show()

    # 计算相关系数矩阵（数值列）
    numeric_cols_for_corr = [
        "年龄",
        "孕妇BMI",
        "孕周_数值",
        "Y染色体浓度",
        "13号染色体的Z值",
        "18号染色体的Z值",
        "21号染色体的Z值",
        "GC含量",
    ]
    corr_matrix = df[numeric_cols_for_corr].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("变量间相关系数矩阵")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.show()


# 4. 问题1: Y染色体浓度与孕周、BMI的关系
def analyze_y_chromosome_relationship(male_data):
    """
    分析Y染色体浓度与孕周、BMI的关系
    """
    # 确保数据无缺失
    data_for_analysis = male_data[["孕周_数值", "孕妇BMI", "Y染色体浓度"]].dropna()

    # 计算Pearson相关系数及显著性
    corr_weeks, p_weeks = stats.pearsonr(
        data_for_analysis["孕周_数值"], data_for_analysis["Y染色体浓度"]
    )
    corr_bmi, p_bmi = stats.pearsonr(
        data_for_analysis["孕妇BMI"], data_for_analysis["Y染色体浓度"]
    )

    print("=== 问题1: Y染色体浓度与孕周、BMI的相关性分析 ===")
    print(f"Y染色体浓度与孕周的相关系数: {corr_weeks:.4f}, p值: {p_weeks:.4e}")
    print(f"Y染色体浓度与BMI的相关系数: {corr_bmi:.4f}, p值: {p_bmi:.4e}")

    # 可视化关系
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x="孕周_数值", y="Y染色体浓度", data=data_for_analysis, ax=axes[0])
    axes[0].set_title("Y染色体浓度 vs. 孕周")
    sns.scatterplot(x="孕妇BMI", y="Y染色体浓度", data=data_for_analysis, ax=axes[1])
    axes[1].set_title("Y染色体浓度 vs. 孕妇BMI")
    plt.tight_layout()
    plt.savefig("y_chromosome_relationships.png")
    plt.show()

    # 建立多元线性回归模型
    X = data_for_analysis[["孕周_数值", "孕妇BMI"]]
    X = sm.add_constant(X)  # 添加截距项
    y = data_for_analysis["Y染色体浓度"]
    model = sm.OLS(y, X).fit()
    print("\n多元线性回归模型摘要:")
    print(model.summary())

    # 返回模型结果以备后用
    return model, data_for_analysis


# 5. 问题2 & 3: 基于BMI分组及最佳检测时点优化
def optimize_test_timing(male_data, y_chr_model):
    """
    基于BMI分组并为每组寻找最佳检测时点
    """
    # 假设：风险函数与Y染色体浓度达标（>4%）的概率和孕周有关
    # 这是一个简化的示例，实际风险函数需根据题目要求精确定义
    # 例如：风险 = f(检测时孕周, BMI分组, 预测的Y染色体浓度)

    # 方法：使用聚类（如K-Means）根据BMI进行分组
    bmi_data = male_data[["孕妇BMI"]].dropna()
    scaler = StandardScaler()
    bmi_scaled = scaler.fit_transform(bmi_data)

    # 使用轮廓系数寻找最佳聚类数（示例，这里假设分3组）
    best_k = 3
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    male_data["BMI分组"] = kmeans.fit_predict(bmi_scaled)

    # 可视化分组结果
    plt.figure(figsize=(8, 6))
    for group in range(best_k):
        group_data = male_data[male_data["BMI分组"] == group]
        plt.hist(group_data["孕妇BMI"], alpha=0.7, label=f"Group {group}")
    plt.xlabel("孕妇BMI")
    plt.ylabel("频数")
    plt.title("基于BMI的孕妇分组")
    plt.legend()
    plt.savefig("bmi_clustering.png")
    plt.show()

    # 为每个分组计算最佳检测时点（需结合问题1的模型预测Y染色体浓度）
    # 这里需要定义“最佳”的准则，例如：最小化风险（发现异常过晚的风险）
    # 以下是一个概念性的框架，具体优化过程需根据题目要求细化
    def risk_function(gestational_week, bmi_group, target_y_concentration=0.04):
        """
        定义风险函数（示例）
        """
        # 使用问题1的模型预测该孕周、该BMI组典型BMI值下的Y染色体浓度
        typical_bmi = male_data[male_data["BMI分组"] == bmi_group]["孕妇BMI"].median()
        X_pred = pd.DataFrame(
            {"const": [1], "孕周_数值": [gestational_week], "孕妇BMI": [typical_bmi]}
        )
        predicted_y = y_chr_model.predict(X_pred)[0]

        # 风险计算：假设风险与（预测浓度低于目标浓度的程度）和（孕周过大）有关
        risk = 0.0
        if predicted_y < target_y_concentration:
            risk += (target_y_concentration - predicted_y) * 10  # 浓度不足的风险
        # 可以添加其他风险项，如孕周过大导致干预过晚的风险
        return risk

    optimal_weeks = {}
    for group in range(best_k):
        # 使用优化算法（如scipy.optimize.minimize）寻找使风险最小的孕周
        result = minimize(
            lambda x: risk_function(x[0], group),
            x0=[12.0],  # 初始猜测：12周
            bounds=[(10, 20)],
        )  # 孕周合理范围
        optimal_weeks[group] = result.x[0]
        print(f"BMI分组 {group} 的最佳检测孕周: {result.x[0]:.2f} 周")

    return male_data, optimal_weeks


# 6. 问题4: 女胎染色体异常判定模型
def build_female_abnormality_model(female_data):
    """
    构建女胎染色体异常判定模型
    """
    if len(female_data) == 0:
        print("警告：无女胎数据，无法构建异常判定模型。")
        return None, None, None

    # 准备特征和目标变量
    # 特征：可选择Z值、GC含量、孕妇年龄、BMI、孕周等
    features = [
        "13号染色体的Z值",
        "18号染色体的Z值",
        "21号染色体的Z值",
        "GC含量",
        "年龄",
        "孕妇BMI",
        "孕周_数值",
    ]
    target = "胎儿是否健康"  # 假设‘否’表示异常，‘是’表示健康（需根据数据调整）

    # 数据预处理：确保目标变量为二值（0/1）
    female_data_clean = female_data.dropna(subset=features + [target])
    female_data_clean["target_binary"] = female_data_clean[target].apply(
        lambda x: 0 if x == "否" else 1
    )

    X = female_data_clean[features]
    y = female_data_clean["target_binary"]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 构建分类模型（以随机森林为例）
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 模型评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== 问题4: 女胎染色体异常判定模型 ===")
    print(f"模型准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    # 特征重要性
    feat_importances = pd.Series(model.feature_importances_, index=features)
    feat_importances.sort_values(ascending=False, inplace=True)
    print("\n特征重要性排序:")
    print(feat_importances)

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    feat_importances.plot(kind="barh")
    plt.title("特征重要性 (随机森林模型)")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

    return model, X_test, y_test


# 主函数：执行完整流程
def main():
    # 文件路径（请根据实际文件位置修改）
    file_path = "附件.xlsx"

    # 1. 加载数据
    df = load_and_explore_data(file_path)

    # 2. 数据预处理
    df_clean, male_data, female_data = preprocess_data(df)

    # 3. 探索性数据分析 (EDA)
    perform_eda(df_clean, male_data, female_data)

    # 4. 解决问题1
    y_chr_model, analysis_data = analyze_y_chromosome_relationship(male_data)

    # 5. 解决问题2 & 3 (基于问题1的模型)
    male_data_with_group, optimal_weeks_dict = optimize_test_timing(
        male_data, y_chr_model
    )

    # 6. 解决问题4
    abnormality_model, X_test, y_test = build_female_abnormality_model(female_data)

    print("\n*** 分析完成！***")


if __name__ == "__main__":
    main()
