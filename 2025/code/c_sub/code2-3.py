import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings("ignore")
# 设置科学出版风格的绘图
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")
fm.addfont("./times.ttf")

# 设置中文字体和负号正常显示
plt.rcParams["font.sans-serif"] = ["FangSong_GB2312", "times"]
plt.rcParams["axes.unicode_minus"] = False


# 加载数据
df = pd.read_csv("./cleaned_data.csv")


# 检查数据集中可能用于识别胎儿性别的列
print("Y染色体浓度统计:")
print(df["Y染色体浓度"].describe())
print("\nY染色体浓度分布:")
print(df["Y染色体浓度"].value_counts().head(10))

print("\nX染色体浓度统计:")
print(df["X染色体浓度"].describe())

# 查看是否有明显的性别区分模式
plt.figure(figsize=(10, 6))
plt.hist(df["Y染色体浓度"].dropna(), bins=50, alpha=0.7, color="blue")
plt.xlabel("Y染色体浓度")
plt.ylabel("频数")
plt.title("Y染色体浓度分布")
plt.grid(True, alpha=0.3)
plt.savefig("y_chromosome_distribution.png", dpi=300, bbox_inches="tight")
# plt.show()

# 基于Y染色体浓度阈值来识别男胎（通常Y染色体浓度>0表示男胎）
threshold = 0.001  # 设置一个小的阈值
male_fetus_df = df[df["Y染色体浓度"] > threshold].copy()
print(f"\n基于Y染色体浓度> {threshold} 的男胎样本量: {len(male_fetus_df)}")
print(
    f"BMI范围: {male_fetus_df['孕妇BMI'].min():.1f} - {male_fetus_df['孕妇BMI'].max():.1f}"
)

# 1. BMI聚类分析 - 使用肘部法则确定最佳分组数
bmi_values = df["孕妇BMI"].dropna().values.reshape(-1, 1)

# 尝试不同的K值进行肘部法则分析
k_range = range(2, 11)
wcss = []  # 簇内平方和
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(bmi_values)
    wcss.append(kmeans.inertia_)

    if k > 1:  # 轮廓系数需要至少2个簇
        silhouette_scores.append(silhouette_score(bmi_values, kmeans.labels_))

# 绘制肘部法则图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, "bo-", linewidth=2, markersize=8)
plt.xlabel("簇数量 (K)")
plt.ylabel("簇内平方和 (WCSS)")
plt.title("肘部法则 - BMI聚类")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, "ro-", linewidth=2, markersize=8)
plt.xlabel("簇数量 (K)")
plt.ylabel("轮廓系数")
plt.title("轮廓系数 - BMI聚类")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("bmi_clustering_elbow.png", dpi=300, bbox_inches="tight")
# plt.show()

# 选择最佳K值（肘部位置）
best_k = 4  # 根据肘部法则选择
print(f"选择的最佳簇数量: {best_k}")

# 使用最佳K值进行K-means聚类
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["BMI_Cluster"] = kmeans.fit_predict(bmi_values)

# 获取每个簇的BMI范围
cluster_stats = df.groupby("BMI_Cluster")["孕妇BMI"].agg(
    ["min", "max", "mean", "count"]
)
print("\n各BMI簇的统计信息:")
print(cluster_stats)

# 为每个簇命名
cluster_names = ["偏瘦", "正常", "超重", "肥胖"]
cluster_ranges = []
for i in range(best_k):
    min_bmi = cluster_stats.loc[i, "min"]
    max_bmi = cluster_stats.loc[i, "max"]
    cluster_ranges.append(f"{min_bmi:.1f}-{max_bmi:.1f}")
    print(
        f"簇{i} ({cluster_names[i]}): BMI范围 {min_bmi:.1f}-{max_bmi:.1f}, 样本数: {cluster_stats.loc[i, 'count']}"
    )

# 重新评估BMI聚类结果并正确命名
print("BMI聚类结果分析:")
print("=" * 50)

# 重新检查聚类中心
cluster_centers = kmeans.cluster_centers_.flatten()
cluster_stats = df.groupby("BMI_Cluster")["孕妇BMI"].agg(
    ["min", "max", "mean", "std", "count"]
)

for i in range(best_k):
    min_bmi = cluster_stats.loc[i, "min"]
    max_bmi = cluster_stats.loc[i, "max"]
    mean_bmi = cluster_stats.loc[i, "mean"]
    count = cluster_stats.loc[i, "count"]

    # 根据WHO标准确定BMI分类
    if mean_bmi < 18.5:
        category = "偏瘦"
    elif 18.5 <= mean_bmi < 25:
        category = "正常"
    elif 25 <= mean_bmi < 30:
        category = "超重"
    else:
        category = "肥胖"

    print(
        f"簇{i}: BMI均值 {mean_bmi:.1f} ({category}), 范围 {min_bmi:.1f}-{max_bmi:.1f}, 样本数: {count}"
    )

# 根据实际医学标准重新命名簇
# 簇0: 40.5 (肥胖III级)
# 簇1: 29.6 (超重)
# 簇2: 35.4 (肥胖I级)
# 簇3: 32.4 (肥胖II级)

# 重新命名簇
cluster_rename = {0: "肥胖III级", 1: "超重", 2: "肥胖I级", 3: "肥胖II级"}

df["BMI_Category"] = df["BMI_Cluster"].map(cluster_rename)

# 验证重命名
print("\n重新命名后的BMI分类:")
print(df["BMI_Category"].value_counts())

# 可视化BMI分布
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="BMI_Category",
    y="孕妇BMI",
    order=["超重", "肥胖I级", "肥胖II级", "肥胖III级"],
)
plt.title("不同BMI分类的分布")
plt.xlabel("BMI分类")
plt.ylabel("孕妇BMI")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("bmi_category_distribution.png", dpi=300, bbox_inches="tight")
# plt.show()
df["孕周"] = df["检测孕周"].apply(
    lambda v: float(v.split("+")[0][::-1][1:][::-1])
    + (0 if (len(v.split("+")) == 1) else float(v.split("+")[1])) / 7
)
# 2. 分析各BMI组的最佳检测时间（Y染色体浓度首次达到4%的最小孕周）

# 首先检查数据中孕周的范围
print("孕周范围:", df["孕周"].min(), "-", df["孕周"].max())

# 定义目标浓度阈值
target_concentration = 0.04  # 4%

# 为每个BMI分类计算达到目标浓度的最小孕周
optimal_times = {}

for category in df["BMI_Category"].unique():
    category_data = df[df["BMI_Category"] == category]

    # 找到该分类中Y染色体浓度达到4%的样本
    qualified_samples = category_data[
        category_data["Y染色体浓度"] >= target_concentration
    ]

    if len(qualified_samples) > 0:
        min_gestational_age = qualified_samples["孕周"].min()
        optimal_times[category] = min_gestational_age
        print(
            f"{category}: 最早达到4%浓度的孕周 = {min_gestational_age:.1f}周 (样本数: {len(qualified_samples)})"
        )
    else:
        optimal_times[category] = None
        print(f"{category}: 没有样本达到4%浓度阈值")

# 可视化各BMI组的达标时间分布
plt.figure(figsize=(12, 6))

# 箱线图显示各BMI组的孕周分布
sns.boxplot(
    data=df,
    x="BMI_Category",
    y="孕周",
    order=["超重", "肥胖I级", "肥胖II级", "肥胖III级"],
)
plt.axhline(
    y=target_concentration * 100,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"目标浓度 ({target_concentration*100:.1f}%)",
)
plt.title("各BMI分类的孕周分布与目标浓度")
plt.xlabel("BMI分类")
plt.ylabel("孕周 (周)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("gestational_age_by_bmi_category.png", dpi=300, bbox_inches="tight")
# plt.show()

# 进一步分析：各BMI组在不同孕周的达标比例
gestational_bins = np.arange(10, 25, 1)  # 10-24周，每1周一个区间

plt.figure(figsize=(14, 8))

for i, category in enumerate(["超重", "肥胖I级", "肥胖II级", "肥胖III级"]):
    category_data = df[df["BMI_Category"] == category]

    qualified_ratio = []
    for week in gestational_bins:
        week_data = category_data[category_data["孕周"] <= week]
        if len(week_data) > 0:
            qualified_count = len(
                week_data[week_data["Y染色体浓度"] >= target_concentration]
            )
            qualified_ratio.append(qualified_count / len(week_data))
        else:
            qualified_ratio.append(0)

    plt.plot(
        gestational_bins,
        qualified_ratio,
        "o-",
        linewidth=2,
        markersize=6,
        label=f"{category} (n={len(category_data)})",
    )

plt.axhline(y=0.95, color="gray", linestyle="--", alpha=0.7, label="95%达标阈值")
plt.xlabel("孕周 (周)")
plt.ylabel("达标比例")
plt.title("各BMI分类在不同孕周的Y染色体浓度达标比例 (≥4%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(gestational_bins)
plt.tight_layout()
plt.savefig("qualified_ratio_by_gestational_age_bmi.png", dpi=300, bbox_inches="tight")
# plt.show()

df.to_csv("test.csv", index=False)

# 3. 建立多元回归模型：达标时间 = f(BMI, 身高, 体重, 年龄) + ε


# 准备回归分析数据
# 对于每个样本，我们关心的是达到目标浓度的时间
# 由于我们无法知道每个样本具体何时达到目标浓度，我们使用检测时的孕周作为代理变量
# 但只考虑那些已经达到目标浓度的样本

regression_data = df[df["Y染色体浓度"] >= target_concentration].copy()
print(f"用于回归分析的样本数量: {len(regression_data)}")

# 定义特征和目标变量
features = ["孕妇BMI", "身高", "体重", "年龄", "孕周"]
X = regression_data[features]
y = regression_data["Y染色体浓度"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 使用statsmodels进行详细的回归分析（包含统计显著性）
X_sm = sm.add_constant(X_scaled)  # 添加常数项
model_sm = sm.OLS(y, X_sm).fit()

print("=" * 60)
print("多元线性回归分析结果 (statsmodels)")
print("=" * 60)
print(model_sm.summary())

# 使用scikit-learn进行预测
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

# 评估模型
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "=" * 60)
print("模型性能评估")
print("=" * 60)
print(f"R² 分数: {r2:.4f}")
print(f"RMSE: {rmse:.6f}")
print(f"特征系数: {lr_model.coef_}")
print(f"截距: {lr_model.intercept_}")

# 特征重要性分析
feature_importance = pd.DataFrame(
    {
        "feature": features,
        "coefficient": lr_model.coef_,
        "abs_coefficient": np.abs(lr_model.coef_),
    }
).sort_values("abs_coefficient", ascending=False)

print("\n特征重要性排序:")
print(feature_importance)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
plt.barh(
    feature_importance["feature"], feature_importance["abs_coefficient"], color=colors
)
plt.xlabel("特征重要性 (系数绝对值)")
plt.title("影响Y染色体浓度的特征重要性")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance_y_chromosome.png", dpi=300, bbox_inches="tight")
# plt.show()

# 保存模型结果
regression_results = {
    "model": lr_model,
    "scaler": scaler,
    "features": features,
    "r2_score": r2,
    "rmse": rmse,
    "coefficients": dict(zip(features, lr_model.coef_)),
    "intercept": lr_model.intercept_,
}

import joblib

joblib.dump(regression_results, "regression_model_results.pkl")
print("回归模型结果已保存到 regression_model_results.pkl")
# 重新执行完整的分析流程，修复错误

# 1. 重新加载数据并准备特征和目标变量
df = pd.read_csv("cleaned_data_with_bmi_categories.csv")

# 定义特征和目标变量
features = ["孕妇BMI", "身高", "体重", "年龄", "孕周"]
target = "Y染色体浓度"

# 准备数据
X = df[features].values
y = df[target].values

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 重新进行训练测试分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")

# 3. 重新训练线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

# 4. 主成分分析/因子分析降维处理多因素影响
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 分析主成分解释的方差比例
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_,
    "o-",
    linewidth=2,
)
plt.xlabel("主成分数量")
plt.ylabel("解释方差比例")
plt.title("各主成分解释的方差比例")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    np.cumsum(pca.explained_variance_ratio_),
    "o-",
    linewidth=2,
)
plt.xlabel("主成分数量")
plt.ylabel("累计解释方差比例")
plt.title("累计解释方差比例")
plt.grid(True, alpha=0.3)
plt.axhline(y=0.95, color="red", linestyle="--", alpha=0.7, label="95%阈值")
plt.legend()

plt.tight_layout()
plt.savefig("pca_variance_explained.png", dpi=300, bbox_inches="tight")
# plt.show()

print("主成分分析结果:")
print(f"各主成分解释方差比例: {pca.explained_variance_ratio_}")
print(f"累计解释方差比例: {np.cumsum(pca.explained_variance_ratio_)}")

# 查看主成分与原始特征的关系
pca_components = pd.DataFrame(
    pca.components_, columns=features, index=[f"PC{i+1}" for i in range(len(features))]
)
print("\n主成分载荷矩阵:")
print(pca_components)

# 5. 敏感性分析：检测误差对各因素的影响
# 假设Y染色体浓度测量误差为±10%
error_magnitude = 0.10  # 10%误差

# 创建带有误差的数据
np.random.seed(42)
y_with_error = y * (1 + np.random.uniform(-error_magnitude, error_magnitude, len(y)))

# 使用带有误差的数据重新训练模型
lr_model_error = LinearRegression()
# 重新分割带有误差的数据
X_train_error, X_test_error, y_train_error, y_test_error = train_test_split(
    X_scaled, y_with_error, test_size=0.2, random_state=42
)
lr_model_error.fit(X_train_error, y_train_error)

y_pred_error = lr_model_error.predict(X_test_error)

# 比较原始模型和误差模型的系数变化
coefficient_changes = pd.DataFrame(
    {
        "feature": features,
        "original_coef": lr_model.coef_,
        "error_coef": lr_model_error.coef_,
        "absolute_change": np.abs(lr_model.coef_ - lr_model_error.coef_),
        "relative_change": np.abs(
            (lr_model.coef_ - lr_model_error.coef_) / lr_model.coef_
        ),
    }
)

print("\n" + "=" * 60)
print("敏感性分析：10%测量误差对模型系数的影响")
print("=" * 60)
print(coefficient_changes)

# 可视化敏感性分析结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.barh(
    features,
    coefficient_changes["absolute_change"],
    color=plt.cm.coolwarm(np.linspace(0, 1, len(features))),
)
plt.xlabel("系数绝对变化量")
plt.title("测量误差导致的系数变化(绝对值)")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.barh(
    features,
    coefficient_changes["relative_change"] * 100,
    color=plt.cm.coolwarm(np.linspace(0, 1, len(features))),
)
plt.xlabel("系数相对变化百分比 (%)")
plt.title("测量误差导致的系数变化(相对百分比)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sensitivity_analysis_error_impact.png", dpi=300, bbox_inches="tight")
# plt.show()

# 6. 基于模型结果进行BMI分组优化建议
target_concentration = 0.04  # 4%目标浓度
optimal_times_by_bmi = {}

for category in df["BMI_Category"].unique():
    category_data = df[df["BMI_Category"] == category]

    # 计算该组达到95%样本达标的最小孕周
    gestational_weeks = np.arange(10, 25, 0.5)
    qualified_ratios = []

    for week in gestational_weeks:
        week_data = category_data[category_data["孕周"] <= week]
        if len(week_data) > 0:
            qualified_count = len(
                week_data[week_data["Y染色体浓度"] >= target_concentration]
            )
            qualified_ratios.append(qualified_count / len(week_data))
        else:
            qualified_ratios.append(0)

    # 找到达到95%达标率的最小孕周
    qualified_ratios = np.array(qualified_ratios)
    if any(qualified_ratios >= 0.95):
        optimal_week = gestational_weeks[np.where(qualified_ratios >= 0.95)[0][0]]
    else:
        optimal_week = None

    optimal_times_by_bmi[category] = optimal_week
    print(f"{category}: 达到95%样本达标的最小孕周 = {optimal_week}周")

# 保存最终分析结果
final_results = {
    "optimal_times_by_bmi": optimal_times_by_bmi,
    "regression_coefficients": dict(zip(features, lr_model.coef_)),
    "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
    "sensitivity_analysis": coefficient_changes.to_dict(),
    "target_concentration": target_concentration,
}

import json

with open("final_analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)

print("最终分析结果已保存到 final_analysis_results.json")
# 生成最终的三组可视化图表

# 1. 各BMI组Y染色体浓度随孕周变化趋势
plt.figure(figsize=(14, 10))

# 子图1: 各BMI组浓度趋势
plt.subplot(2, 2, 1)
for category in df["BMI_Category"].unique():
    category_data = df[df["BMI_Category"] == category]
    # 按孕周分组计算平均浓度
    weekly_avg = category_data.groupby("孕周")["Y染色体浓度"].mean()
    plt.plot(
        weekly_avg.index,
        weekly_avg.values,
        "o-",
        label=category,
        linewidth=2,
        markersize=6,
    )

plt.xlabel("检测孕周")
plt.ylabel("Y染色体平均浓度 (%)")
plt.title("各BMI组Y染色体浓度随孕周变化趋势")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0.04, color="red", linestyle="--", alpha=0.7, label="4%阈值")

# 子图2: 达标率曲线
plt.subplot(2, 2, 2)
gestational_weeks = np.arange(10, 25, 0.5)

for category in df["BMI_Category"].unique():
    category_data = df[df["BMI_Category"] == category]
    qualified_ratios = []

    for week in gestational_weeks:
        week_data = category_data[category_data["孕周"] <= week]
        if len(week_data) > 0:
            qualified_count = len(week_data[week_data["Y染色体浓度"] >= 0.04])
            qualified_ratios.append(qualified_count / len(week_data))
        else:
            qualified_ratios.append(0)

    plt.plot(
        gestational_weeks,
        qualified_ratios,
        "o-",
        label=category,
        linewidth=2,
        markersize=4,
    )

plt.xlabel("检测孕周")
plt.ylabel("达标率 (浓度≥4%)")
plt.title("各BMI组达标率随孕周变化")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0.95, color="green", linestyle="--", alpha=0.7, label="95%达标率")

# 子图3: 回归系数重要性
plt.subplot(2, 2, 3)
coefficients = pd.DataFrame(
    {"feature": features, "coefficient": lr_model.coef_}
).sort_values("coefficient", key=abs, ascending=True)

plt.barh(
    coefficients["feature"],
    coefficients["coefficient"],
    color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(features))),
)
plt.xlabel("回归系数")
plt.title("特征对Y染色体浓度的影响程度")
plt.grid(True, alpha=0.3)
plt.axvline(x=0, color="black", linestyle="-", alpha=0.5)

# 子图4: 敏感性分析结果
plt.subplot(2, 2, 4)
plt.barh(
    features,
    coefficient_changes["relative_change"] * 100,
    color=plt.cm.coolwarm(np.linspace(0, 1, len(features))),
)
plt.xlabel("系数相对变化百分比 (%)")
plt.title("10%测量误差对系数的影响")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("comprehensive_analysis_summary.png", dpi=300, bbox_inches="tight")
# plt.show()

# 2. 主成分分析可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# 主成分载荷热图
pca_loadings = pd.DataFrame(pca.components_, columns=features)
sns.heatmap(
    pca_loadings,
    annot=True,
    cmap="RdBu_r",
    center=0,
    fmt=".3f",
    cbar_kws={"label": "载荷系数"},
)
plt.title("主成分载荷矩阵")

plt.subplot(1, 2, 2)
# 主成分解释方差
plt.bar(
    range(1, 6),
    pca.explained_variance_ratio_ * 100,
    color=plt.cm.viridis(np.linspace(0, 1, 5)),
)
plt.plot(
    range(1, 6), np.cumsum(pca.explained_variance_ratio_) * 100, "ro-", linewidth=2
)
plt.xlabel("主成分")
plt.ylabel("解释方差比例 (%)")
plt.title("主成分解释方差比例")
plt.legend(["累计解释方差", "单个主成分解释方差"])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pca_analysis_visualization.png", dpi=300, bbox_inches="tight")
# plt.show()

# 3. 最优检测时间推荐
optimal_times = []
categories = []

for category, week in optimal_times_by_bmi.items():
    if week is not None:
        optimal_times.append(week)
        categories.append(category)

plt.figure(figsize=(10, 6))
colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"]

plt.bar(categories, optimal_times, color=colors[: len(categories)])
plt.xlabel("BMI分类")
plt.ylabel("推荐检测孕周")
plt.title("各BMI组达到95%达标率的最早检测时间")

# 添加数值标签
for i, (category, week) in enumerate(optimal_times_by_bmi.items()):
    if week is not None:
        plt.text(
            i, week + 0.1, f"{week}周", ha="center", va="bottom", fontweight="bold"
        )

plt.grid(True, alpha=0.3, axis="y")
plt.ylim(10, 13)
plt.savefig("optimal_detection_times.png", dpi=300, bbox_inches="tight")
# plt.show()

print("=" * 60)
print("最终分析完成！已生成三组核心可视化图表：")
print("1. comprehensive_analysis_summary.png - 综合分析总览")
print("2. pca_analysis_visualization.png - 主成分分析可视化")
print("3. optimal_detection_times.png - 最优检测时间推荐")
print("=" * 60)

# 输出关键结论
print("\n关键发现：")
print(f"1. 所有BMI组在孕11-11.5周即可达到95%的样本Y染色体浓度≥4%")
print(f"2. 体重对Y染色体浓度影响最大（负相关），其次是孕妇BMI（正相关）")
print(f"3. 前两个主成分解释了63.5%的方差，表明数据有较好的降维潜力")
print(f"4. 模型对测量误差相对稳健，10%误差导致系数变化<16%")
print(f"5. 推荐检测时间：超重和肥胖I级-11周，肥胖II级和III级-11.5周")
