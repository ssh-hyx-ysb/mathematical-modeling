import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import IsolationForest
import os

# 设置Seaborn风格
sns.set_style("whitegrid")
import matplotlib

fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")
fm.addfont("./times.ttf")
print(fm)
# 设置中文字体和负号正常显示
plt.rcParams["font.sans-serif"] = ["FangSong_GB2312", "times"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据
male_data = pd.read_excel("附件.xlsx", sheet_name="男胎检测数据")
female_data = pd.read_excel("附件.xlsx", sheet_name="女胎检测数据")


# 数据预处理
def clean_week(week_str):
    if isinstance(week_str, str):
        parts = week_str.replace("w", "").replace("W", "").split("+")
        return int(parts[0]) + (int(parts[1]) / 7 if len(parts) > 1 else 0)
    return week_str


male_data["孕周"] = male_data["检测孕周"].apply(clean_week)
female_data["孕周"] = female_data["检测孕周"].apply(clean_week)

# 问题1: 分析Y染色体浓度与孕周数和BMI的相关性
male_data_q1 = male_data.dropna(subset=["Y染色体浓度", "孕周", "孕妇BMI"])
X_q1 = male_data_q1[["孕周", "孕妇BMI", "孕周**2"]].assign(
    **{"孕周**2": male_data_q1["孕周"] ** 2}
)
y_q1 = male_data_q1["Y染色体浓度"]

# 单变量回归
model_q1_gw = stats.linregress(X_q1["孕周"], y_q1)
slope_gw, intercept_gw, r_value_gw, p_value_gw, _ = model_q1_gw
model_q1_bmi = stats.linregress(X_q1["孕妇BMI"], y_q1)
slope_bmi, intercept_bmi, r_value_bmi, p_value_bmi, _ = model_q1_bmi

# 多变量回归
model_q1_multi = LinearRegression()
model_q1_multi.fit(X_q1, y_q1)
r_squared = model_q1_multi.score(X_q1, y_q1)

# 可视化
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=male_data_q1,
    x="孕周",
    y="Y染色体浓度",
    hue="孕周",
    palette="Blues",
    legend=False,
)
sns.lineplot(x=X_q1["孕周"], y=intercept_gw + slope_gw * X_q1["孕周"], color="red")
plt.title("Y染色体浓度与孕周的相关性")
plt.xlabel("孕周")
plt.ylabel("Y染色体浓度 (%)")
plt.savefig("y_concentration_vs_week.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=male_data_q1,
    x="孕妇BMI",
    y="Y染色体浓度",
    hue="孕妇BMI",
    palette="Greens",
    legend=False,
)
sns.lineplot(
    x=X_q1["孕妇BMI"], y=intercept_bmi + slope_bmi * X_q1["孕妇BMI"], color="red"
)
plt.title("Y染色体浓度与BMI的相关性")
plt.xlabel("BMI")
plt.ylabel("Y染色体浓度 (%)")
plt.savefig("y_concentration_vs_bmi.png")
plt.close()

print(f"孕周相关系数: {r_value_gw:.3f}, p值: {p_value_gw:.3f}")
print(f"BMI相关系数: {r_value_bmi:.3f}, p值: {p_value_bmi:.3f}")
print(f"多变量R-squared: {r_squared:.3f}")

# 问题2: BMI分组和最优NIPT时点
male_data_q2 = male_data[male_data["Y染色体浓度"] >= 0.04].dropna(
    subset=["孕妇BMI", "孕周"]
)
# 使用分位数分组
male_data_q2["BMI_Group"] = pd.qcut(
    male_data_q2["孕妇BMI"], q=4, labels=["G1", "G2", "G3", "G4"]
)
optimal_weeks = male_data_q2.groupby("BMI_Group")["孕周"].median().sort_index()

# 误差敏感性分析
np.random.seed(42)
error_range = 0.05
perturbed_weeks = []
for _ in range(100):
    perturbed_data = male_data_q2.copy()
    perturbed_data["Y染色体浓度"] *= 1 + np.random.uniform(-error_range, error_range)
    perturbed_data = perturbed_data[perturbed_data["Y染色体浓度"] >= 0.04]
    perturbed_weeks.append(perturbed_data.groupby("BMI_Group")["孕周"].median())

perturbed_weeks = pd.DataFrame(perturbed_weeks)
week_variation = perturbed_weeks.std()

# 可视化
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=male_data_q2,
    x="BMI_Group",
    y="孕周",
    hue="BMI_Group",
    palette="Set2",
    legend=False,
)
plt.title("BMI分组与最优NIPT时点")
plt.xlabel("BMI分组")
plt.ylabel("最优孕周")
plt.savefig("bmi_grouping_vs_week.png")
plt.close()

print(f"分组与时点: {optimal_weeks.to_dict()}")
print(f"时点变动范围: {week_variation.to_dict()}")

# 问题3: 综合因素优化
male_data_q3 = male_data.dropna(
    subset=["孕妇BMI", "孕周", "身高", "体重", "年龄", "Y染色体浓度"]
)
X_q3 = male_data_q3[["孕妇BMI", "身高", "体重", "年龄"]]
y_q3 = (male_data_q3["Y染色体浓度"] >= 0.04).astype(int)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_q3, y_q3)
importance = rf_model.feature_importances_
reach_prob = rf_model.predict_proba(X_q3)[:, 1]

male_data_q3["Reach_Prob"] = reach_prob
male_data_q3["BMI_Group"] = pd.qcut(
    male_data_q3["孕妇BMI"], q=4, labels=["G1", "G2", "G3", "G4"]
)
optimal_weeks_q3 = male_data_q3.groupby("BMI_Group")["孕周"].median().sort_index()

# 误差分析
y_pred_q3 = rf_model.predict(X_q3)
mse = np.mean((y_q3 - y_pred_q3) ** 2)

# 可视化
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=male_data_q3, x="孕妇BMI", y="Reach_Prob", hue="BMI_Group", palette="deep"
)
plt.title("达标概率与BMI分布")
plt.xlabel("BMI")
plt.ylabel("达标概率")
plt.savefig("reach_prob_vs_bmi.png")
plt.close()

print(f"优化分组与时点: {optimal_weeks_q3.to_dict()}")
print(f"误差MSE: {mse:.4f}")

# 问题4: 女胎异常判定
female_data_q4 = female_data.dropna(
    subset=[
        "13号染色体的Z值",
        "18号染色体的Z值",
        "21号染色体的Z值",
        "X染色体的Z值",
        "孕妇BMI",
        "染色体的非整倍体",
    ]
)
y_q4 = (female_data_q4["染色体的非整倍体"] != "").astype(int)
X_q4 = female_data_q4[
    ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值", "孕妇BMI"]
]

# 检查类分布
print(f"女胎数据类分布: {np.bincount(y_q4)}")
if len(np.unique(y_q4)) < 2:
    # 若仅一类，使用Isolation Forest进行异常检测
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_q4)
    y_pred_q4 = iso_forest.predict(X_q4)
    anomaly_score = -iso_forest.score_samples(X_q4)
    threshold = np.percentile(anomaly_score, 10)  # 10%为异常
    y_pred_q4 = (anomaly_score > threshold).astype(int)
else:
    # 若有两类，使用逻辑回归
    from sklearn.linear_model import LogisticRegression

    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_q4, y_q4)
    y_pred_q4 = lr_model.predict(X_q4)

accuracy = accuracy_score(y_q4, y_pred_q4)
auc = (
    roc_auc_score(y_q4, iso_forest.score_samples(X_q4))
    if len(np.unique(y_q4)) < 2
    else roc_auc_score(y_q4, lr_model.predict_proba(X_q4)[:, 1])
)

# 可视化
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=anomaly_score if len(np.unique(y_q4)) < 2 else lr_model.predict_proba(X_q4)[:, 1],
    y=y_q4,
    hue=y_pred_q4,
    palette="viridis",
)
plt.title("女胎异常判定得分分布")
plt.xlabel("异常得分" if len(np.unique(y_q4)) < 2 else "概率")
plt.ylabel("真实标签")
plt.savefig("anomaly_score_female.png")
plt.close()

print(f"准确率: {accuracy:.2f}, AUC: {auc:.2f}")
