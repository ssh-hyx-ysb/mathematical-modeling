import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 1. 数据读取与预处理
def load_and_preprocess_data(file_path):
    """读取并预处理NIPT数据"""
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 查看数据基本信息
    print("数据基本信息：")
    print(f"数据集形状: {df.shape}")
    print("\n前5行数据:")
    print(df.head())

    # 分离男胎和女胎数据
    # 男胎数据有Y染色体相关信息，女胎没有
    male_df = df[df["U"].notna() & df["V"].notna()].copy()  # U和V列非空的是男胎
    female_df = df[df["U"].isna() & df["V"].isna()].copy()  # U和V列空的是女胎

    print(f"\n男胎样本数: {male_df.shape[0]}")
    print(f"女胎样本数: {female_df.shape[0]}")

    # 处理孕周数据：将"11w+6"格式转换为小数
    def convert_week(week_str):
        if pd.isna(week_str) or "w" not in str(week_str):
            return np.nan
        week_part = str(week_str).split("w")[0]
        day_part = str(week_str).split("w+")[-1] if "+" in str(week_str) else "0"
        try:
            return float(week_part) + float(day_part) / 7  # 转换为周
        except:
            return np.nan

    # 对男胎和女胎数据都处理孕周
    male_df["孕周_小数"] = male_df["J"].apply(convert_week)
    female_df["孕周_小数"] = female_df["J"].apply(convert_week)

    # 过滤无效数据
    male_df = male_df.dropna(subset=["孕周_小数", "V", "K"])  # V是Y染色体浓度，K是BMI
    female_df = female_df.dropna(
        subset=["孕周_小数", "K", "Q", "R", "S", "T", "X", "Y", "Z", "AA"]
    )

    # 过滤Y染色体浓度异常值（合理范围0-0.2）
    male_df = male_df[(male_df["V"] >= 0) & (male_df["V"] <= 0.2)]

    # 过滤GC含量异常值（正常范围40%-60%）
    valid_gc = (male_df["P"] >= 0.4) & (male_df["P"] <= 0.6)
    male_df = male_df[valid_gc]

    valid_gc_female = (female_df["P"] >= 0.4) & (female_df["P"] <= 0.6)
    female_df = female_df[valid_gc_female]

    # 为女胎数据创建异常标签（1表示异常，0表示正常）
    female_df["异常标签"] = female_df["AB"].apply(lambda x: 0 if pd.isna(x) else 1)

    print(f"清洗后男胎样本数: {male_df.shape[0]}")
    print(f"清洗后女胎样本数: {female_df.shape[0]}")

    return male_df, female_df


# 2. 问题1：分析胎儿Y染色体浓度与孕妇孕周数、BMI等指标的相关特性
def solve_problem1(male_df):
    """解决问题1：分析Y染色体浓度与孕周、BMI的关系"""
    print("\n===== 解决问题1 =====")

    # 提取需要的特征
    X = male_df[["孕周_小数", "K"]]  # K是BMI
    y = male_df["V"]  # Y染色体浓度

    # 添加交互项：孕周×BMI
    X["孕周×BMI"] = X["孕周_小数"] * X["K"]

    # 构建多元线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 输出模型参数
    print(
        f"回归方程: Y染色体浓度 = {model.intercept_:.4f} + {model.coef_[0]:.4f}×孕周 + {model.coef_[1]:.4f}×BMI + {model.coef_[2]:.4f}×孕周×BMI"
    )
    print(f"模型R²值: {model.score(X, y):.4f}")

    # 相关性分析
    corr_matrix = male_df[["孕周_小数", "K", "V"]].corr()
    print("\n相关性矩阵:")
    print(corr_matrix)

    # 可视化相关性
    plt.figure(figsize=(12, 10))

    # 散点图：孕周与Y染色体浓度
    plt.subplot(2, 2, 1)
    plt.scatter(male_df["孕周_小数"], male_df["V"], alpha=0.5)
    plt.xlabel("孕周")
    plt.ylabel("Y染色体浓度")
    plt.title("孕周与Y染色体浓度的关系")

    # 散点图：BMI与Y染色体浓度
    plt.subplot(2, 2, 2)
    plt.scatter(male_df["K"], male_df["V"], alpha=0.5, color="orange")
    plt.xlabel("BMI")
    plt.ylabel("Y染色体浓度")
    plt.title("BMI与Y染色体浓度的关系")

    # 相关性热力图
    plt.subplot(2, 2, 3)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("特征相关性热力图")

    # 部分依赖图：展示不同BMI下，孕周与Y染色体浓度的关系
    plt.subplot(2, 2, 4)
    bmi_values = [male_df["K"].min(), male_df["K"].mean(), male_df["K"].max()]
    week_range = np.linspace(
        male_df["孕周_小数"].min(), male_df["孕周_小数"].max(), 100
    )

    for bmi in bmi_values:
        X_pred = pd.DataFrame(
            {
                "孕周_小数": week_range,
                "K": [bmi] * len(week_range),
                "孕周×BMI": week_range * bmi,
            }
        )
        y_pred = model.predict(X_pred)
        plt.plot(week_range, y_pred, label=f"BMI={bmi:.1f}")

    plt.xlabel("孕周")
    plt.ylabel("预测的Y染色体浓度")
    plt.title("不同BMI下孕周与Y染色体浓度的关系")
    plt.legend()

    plt.tight_layout()
    plt.savefig("问题1_相关性分析.png", dpi=300, bbox_inches="tight")
    plt.show()

    return model


# 3. 问题2：以男胎孕妇BMI为核心因素进行分组，确定每组最佳NIPT时点
def solve_problem2(male_df, regression_model):
    """解决问题2：BMI分组与最佳NIPT时点"""
    print("\n===== 解决问题2 =====")

    # 确定最佳聚类数（肘部法则）
    bmi_data = male_df["K"].values.reshape(-1, 1)
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(bmi_data)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), sse, marker="o")
    plt.xlabel("聚类数")
    plt.ylabel("SSE")
    plt.title("肘部法则确定最佳BMI分组数")
    plt.grid(True, alpha=0.3)
    plt.savefig("问题2_肘部法则.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 选择最佳聚类数（根据肘部法则，这里选择4）
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    male_df["BMI分组"] = kmeans.fit_predict(bmi_data)

    # 获取每个分组的BMI范围
    groups = []
    for i in range(n_clusters):
        group_data = male_df[male_df["BMI分组"] == i]["K"]
        groups.append(
            {
                "分组": i,
                "最小值": group_data.min(),
                "最大值": group_data.max(),
                "均值": group_data.mean(),
                "样本数": len(group_data),
            }
        )

    # 按BMI范围排序分组
    groups.sort(key=lambda x: x["均值"])
    for i, group in enumerate(groups):
        group["分组名称"] = f'分组{i+1} ({group["最小值"]:.1f}-{group["最大值"]:.1f})'
        male_df.loc[male_df["BMI分组"] == group["分组"], "BMI分组名称"] = group[
            "分组名称"
        ]

    print("BMI分组结果:")
    for group in groups:
        print(
            f"{group['分组名称']}: 样本数={group['样本数']}, 均值={group['均值']:.2f}"
        )

    # 计算每个分组的最佳检测时点
    # 定义风险函数
    def calculate_risk(week):
        if week <= 12:
            return 0.2  # 早期风险系数
        elif week <= 27:
            return 0.6  # 中期风险系数
        else:
            return 0.9  # 晚期风险系数

    # 计算每个孕周的达标率和风险值
    optimal_weeks = []
    week_range = np.arange(10, 26, 0.1)  # NIPT检测通常在10-25周

    for group in groups:
        group_data = male_df[male_df["BMI分组名称"] == group["分组名称"]]
        min_risk = float("inf")
        optimal_week = 0

        for week in week_range:
            # 预测该孕周的Y染色体浓度
            X_pred = pd.DataFrame(
                {
                    "孕周_小数": [week],
                    "K": [group["均值"]],
                    "孕周×BMI": [week * group["均值"]],
                }
            )
            y_pred = regression_model.predict(X_pred)[0]

            # 计算达标率（浓度≥4%）
            success_rate = np.mean(
                group_data.apply(
                    lambda row: (
                        1
                        if regression_model.predict(
                            pd.DataFrame(
                                {
                                    "孕周_小数": [week],
                                    "K": [row["K"]],
                                    "孕周×BMI": [week * row["K"]],
                                }
                            )
                        )[0]
                        >= 0.04
                        else 0
                    ),
                    axis=1,
                )
            )

            # 计算风险值
            risk = calculate_risk(week) * (1 - success_rate)

            if risk < min_risk:
                min_risk = risk
                optimal_week = week

        optimal_weeks.append(
            {
                "分组名称": group["分组名称"],
                "最佳孕周": optimal_week,
                "最小风险值": min_risk,
            }
        )
        print(
            f"{group['分组名称']}的最佳检测时点: {optimal_week:.2f}周, 最小风险值: {min_risk:.4f}"
        )

    # 可视化不同BMI分组的Y染色体浓度随孕周变化
    plt.figure(figsize=(12, 8))
    colors = ["blue", "green", "orange", "red"]

    for i, group in enumerate(groups):
        group_data = male_df[male_df["BMI分组名称"] == group["分组名称"]]
        week_means = group_data.groupby("孕周_小数")["V"].mean().reset_index()

        if len(week_means) > 1:
            plt.plot(
                week_means["孕周_小数"],
                week_means["V"],
                label=group["分组名称"],
                color=colors[i],
                linewidth=2.5,
                marker="o",
                markersize=4,
            )

    # 标记最佳检测时点
    for i, ow in enumerate(optimal_weeks):
        plt.axvline(x=ow["最佳孕周"], color=colors[i], linestyle="--", alpha=0.7)

    # 标记达标阈值(4%)
    plt.axhline(y=0.04, color="gray", linestyle="--", alpha=0.7, label="达标阈值(4%)")

    plt.xlabel("孕周")
    plt.ylabel("Y染色体浓度均值")
    plt.title("不同BMI分组的Y染色体浓度随孕周变化趋势")
    plt.legend(title="BMI分组")
    plt.grid(True, alpha=0.3)
    plt.xlim(10, 25)
    plt.savefig("问题2_BMI分组趋势图.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 检测误差分析
    error_impacts = []
    error_levels = [0.005, 0.01, 0.015]  # ±0.5%, ±1%, ±1.5%

    for error in error_levels:
        for ow in optimal_weeks:
            group = next(g for g in groups if g["分组名称"] == ow["分组名称"])
            # 计算误差影响下的最佳孕周变化
            week_with_error = ow["最佳孕周"]
            # 简单模拟：误差越大，最佳孕周可能需要延后
            week_with_error += error * 10  # 经验系数
            error_impacts.append(
                {
                    "分组名称": ow["分组名称"],
                    "误差水平": error * 100,
                    "原始最佳孕周": ow["最佳孕周"],
                    "误差后最佳孕周": week_with_error,
                    "变化量": week_with_error - ow["最佳孕周"],
                }
            )

    print("\n检测误差对最佳时点的影响:")
    for impact in error_impacts:
        if impact["分组名称"] == error_impacts[0]["分组名称"]:
            print(f"\n误差水平: ±{impact['误差水平']}%")
        print(f"  {impact['分组名称']}: 变化 {impact['变化量']:.2f}周")

    return male_df, groups, optimal_weeks


# 4. 问题3：综合多因素对男胎孕妇分组并确定最佳NIPT时点
def solve_problem3(male_df, regression_model, original_groups):
    """解决问题3：多因素分组与最佳NIPT时点"""
    print("\n===== 解决问题3 =====")

    # 筛选关键变量（年龄、身高、怀孕次数等）
    features = ["孕周_小数", "K", "C", "D", "AC"]  # K:BMI, C:年龄, D:身高, AC:怀孕次数

    # 特征标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(male_df[features])

    # 使用K-means进行多因素聚类
    n_clusters = 5  # 相比问题2增加一个分组
    kmeans_multi = KMeans(n_clusters=n_clusters, random_state=42)
    male_df["多因素分组"] = kmeans_multi.fit_predict(X)

    # 分析每个分组的特征
    multi_groups = []
    for i in range(n_clusters):
        group_data = male_df[male_df["多因素分组"] == i]
        group_stats = {
            "分组": i,
            "样本数": len(group_data),
            "BMI均值": group_data["K"].mean(),
            "年龄均值": group_data["C"].mean(),
            "身高均值": group_data["D"].mean(),
            "怀孕次数均值": group_data["AC"].mean(),
        }
        multi_groups.append(group_stats)

    # 按BMI均值排序并命名
    multi_groups.sort(key=lambda x: x["BMI均值"])
    for i, group in enumerate(multi_groups):
        group["分组名称"] = f"多因素分组{i+1}"
        male_df.loc[male_df["多因素分组"] == group["分组"], "多因素分组名称"] = group[
            "分组名称"
        ]

    print("多因素分组统计:")
    for group in multi_groups:
        print(
            f"{group['分组名称']}: 样本数={group['样本数']}, BMI均值={group['BMI均值']:.2f}, 年龄均值={group['年龄均值']:.1f}"
        )

    # 计算每个多因素分组的最佳检测时点，考虑达标比例
    def calculate_risk(week):
        if week <= 12:
            return 0.2
        elif week <= 27:
            return 0.6
        else:
            return 0.9

    optimal_weeks_multi = []
    week_range = np.arange(10, 26, 0.1)

    for group in multi_groups:
        group_data = male_df[male_df["多因素分组名称"] == group["分组名称"]]
        min_risk = float("inf")
        optimal_week = 0
        best_success_rate = 0

        for week in week_range:
            # 计算该孕周的达标率
            success_rate = np.mean(
                group_data.apply(
                    lambda row: (
                        1
                        if regression_model.predict(
                            pd.DataFrame(
                                {
                                    "孕周_小数": [week],
                                    "K": [row["K"]],
                                    "孕周×BMI": [week * row["K"]],
                                }
                            )
                        )[0]
                        >= 0.04
                        else 0
                    ),
                    axis=1,
                )
            )

            # 过滤达标率低于80%的情况
            if success_rate < 0.8:
                continue

            # 计算风险值，考虑达标比例
            risk = (
                calculate_risk(week) * (1 - success_rate) * (1 - success_rate)
            )  # 增加达标比例的权重

            if risk < min_risk:
                min_risk = risk
                optimal_week = week
                best_success_rate = success_rate

        optimal_weeks_multi.append(
            {
                "分组名称": group["分组名称"],
                "最佳孕周": optimal_week,
                "最小风险值": min_risk,
                "达标率": best_success_rate,
            }
        )
        print(
            f"{group['分组名称']}的最佳检测时点: {optimal_week:.2f}周, 最小风险值: {min_risk:.4f}, 达标率: {best_success_rate:.2%}"
        )

    # 可视化多因素分组与原始BMI分组的最佳时点对比
    plt.figure(figsize=(12, 8))

    # 提取原始分组和多因素分组的最佳孕周
    original_weeks = [ow["最佳孕周"] for ow in original_groups]
    original_labels = [ow["分组名称"].split(" ")[0] for ow in original_groups]

    multi_weeks = [ow["最佳孕周"] for ow in optimal_weeks_multi]
    multi_labels = [ow["分组名称"] for ow in optimal_weeks_multi]

    x1 = np.arange(len(original_weeks))
    x2 = np.arange(len(multi_weeks)) + len(original_weeks) + 1  # 留出间隔

    plt.bar(x1, original_weeks, width=0.8, label="原始BMI分组")
    plt.bar(x2, multi_weeks, width=0.8, label="多因素分组")

    plt.xlabel("分组")
    plt.ylabel("最佳检测孕周")
    plt.title("原始BMI分组与多因素分组的最佳检测时点对比")
    plt.xticks(np.concatenate([x1, x2]), original_labels + multi_labels, rotation=45)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("问题3_分组对比.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 蒙特卡洛模拟分析检测误差影响
    np.random.seed(42)
    n_simulations = 1000
    error_results = []

    for group in optimal_weeks_multi:
        base_week = group["最佳孕周"]
        week_changes = []

        for _ in range(n_simulations):
            # 模拟误差：浓度测量误差和孕周计算误差
            conc_error = np.random.normal(0, 0.01)  # 浓度测量误差±1%
            week_error = np.random.normal(0, 0.5)  # 孕周计算误差±0.5周

            # 误差导致的最佳孕周变化
            new_week = base_week + abs(conc_error) * 5 + abs(week_error)
            week_changes.append(new_week - base_week)

        error_results.append(
            {
                "分组名称": group["分组名称"],
                "平均变化": np.mean(week_changes),
                "95%置信区间": (
                    np.percentile(week_changes, 2.5),
                    np.percentile(week_changes, 97.5),
                ),
            }
        )

    print("\n蒙特卡洛模拟检测误差影响:")
    for res in error_results:
        print(
            f"{res['分组名称']}: 平均变化 {res['平均变化']:.2f}周, 95%置信区间 [{res['95%置信区间'][0]:.2f}, {res['95%置信区间'][1]:.2f}]周"
        )

    return male_df, optimal_weeks_multi


# 5. 问题4：构建女胎异常判定方法
def solve_problem4(female_df):
    """解决问题4：女胎异常判定模型"""
    print("\n===== 解决问题4 =====")

    # 查看异常样本比例
    abnormal_ratio = female_df["异常标签"].mean()
    print(f"女胎异常样本比例: {abnormal_ratio:.2%}")

    # 选择特征：染色体Z值、GC含量、过滤读段比例、BMI等
    features = [
        "Q",
        "R",
        "S",
        "T",
        "X",
        "Y",
        "Z",
        "AA",
        "K",
    ]  # Q-R-S-T是染色体Z值，X-Y-Z是GC含量，AA是过滤读段比例，K是BMI

    X = female_df[features]
    y = female_df["异常标签"]

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 构建随机森林模型
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # 模型评估
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"随机森林模型性能:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
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
    plt.title("女胎异常判定混淆矩阵")
    plt.savefig("问题4_混淆矩阵.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 特征重要性
    feature_importance = pd.DataFrame(
        {"特征": features, "重要性": rf_model.feature_importances_}
    ).sort_values("重要性", ascending=False)

    print("\n特征重要性排序:")
    print(feature_importance)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="重要性", y="特征", data=feature_importance)
    plt.xlabel("重要性得分")
    plt.ylabel("特征")
    plt.title("女胎异常判定的特征重要性")
    plt.grid(True, axis="x", alpha=0.3)
    plt.savefig("问题4_特征重要性.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 关键特征与异常的关系可视化
    plt.figure(figsize=(15, 10))
    top_features = feature_importance["特征"].head(4).tolist()

    for i, feature in enumerate(top_features):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x="异常标签", y=feature, data=female_df)
        plt.xlabel("是否异常")
        plt.ylabel(feature)
        plt.title(f"{feature}与异常的关系")

    plt.tight_layout()
    plt.savefig("问题4_关键特征与异常关系.png", dpi=300, bbox_inches="tight")
    plt.show()

    return rf_model, feature_importance


# 主函数
def main(file_path):
    # 加载和预处理数据
    male_df, female_df = load_and_preprocess_data(file_path)

    # 解决问题1
    regression_model = solve_problem1(male_df)

    # 解决问题2
    male_df, groups, optimal_weeks = solve_problem2(male_df, regression_model)

    # 解决问题3
    male_df, optimal_weeks_multi = solve_problem3(
        male_df, regression_model, optimal_weeks
    )

    # 解决问题4
    rf_model, feature_importance = solve_problem4(female_df)

    print("\n所有问题分析完成！")
    return {
        "male_data": male_df,
        "female_data": female_df,
        "regression_model": regression_model,
        "rf_model": rf_model,
        "feature_importance": feature_importance,
    }


if __name__ == "__main__":
    # 替换为实际的Excel文件路径
    file_path = "附件.xlsx"
    results = main(file_path)
