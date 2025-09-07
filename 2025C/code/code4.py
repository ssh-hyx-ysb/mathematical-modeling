import pandas as shuju_gongju
import matplotlib
import matplotlib.pyplot as huatude
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
female_data = shuju_gongju.read_excel("附件.xlsx", sheet_name="女胎检测数据")
output_dir = Path("C4_Output")
output_dir.mkdir(exist_ok=True)
fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")
fm.addfont("./times.ttf")
huatude.rcParams["font.sans-serif"] = ["FangSong_GB2312", "times"]
huatude.rcParams["axes.unicode_minus"] = False


def is_abnormal(ab):
    if shuju_gongju.isna(ab) or ab.strip() == "":
        return 0
    ab = str(ab).upper()
    if "T13" in ab or "T18" in ab or "T21" in ab:
        return 1
    return 0


def weighted_rule(row):
    if (
        abs(row["21号染色体的Z值_加权"]) > 2.8
        or abs(row["18号染色体的Z值_加权"]) > 2.8
        or abs(row["13号染色体的Z值_加权"]) > 2.8
    ):
        return 1
    return 0


def simple_z_rule(row):
    z21 = row["21号染色体的Z值"]
    z18 = row["18号染色体的Z值"]
    z13 = row["13号染色体的Z值"]
    if abs(z21) > 3 or abs(z18) > 3 or abs(z13) > 3:
        return 1
    return 0


def quality_weight(row):
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


female_data["label_abnormal"] = female_data["染色体的非整倍体"].apply(is_abnormal)
print(f"女胎数据总量: {len(female_data)}")
print(f"报告异常数量: {female_data['label_abnormal'].sum()}")
z_features = ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值"]
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
demo_features = ["孕妇BMI", "年龄"]
all_features = z_features + qc_features + demo_features
female_data = female_data.dropna(subset=all_features + ["label_abnormal"])
X = female_data[all_features]
y = female_data["label_abnormal"]
print(f"有效样本量: {len(X)}，其中异常: {y.sum()}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
feature_names = X.columns.tolist()
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
cv_f1_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring="f1").mean()
cv_auc_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring="roc_auc").mean()
cv_f1_lr = cross_val_score(lr, X_scaled, y, cv=5, scoring="f1").mean()
cv_auc_lr = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc").mean()
print("\n 模型交叉验证性能 ")
print(f"随机森林  F1: {cv_f1_rf:.3f}, AUC: {cv_auc_rf:.3f}")
print(f"逻辑回归  F1: {cv_f1_lr:.3f}, AUC: {cv_auc_lr:.3f}")
rf.fit(X_scaled, y)
importance_df = shuju_gongju.DataFrame(
    {"feature": feature_names, "importance": rf.feature_importances_}
).sort_values("importance", ascending=False)
print("\n特征重要性")
print(importance_df.head(10))
y_proba = rf.predict_proba(X_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y, y_proba)
auc = roc_auc_score(y, y_proba)
huatude.figure(figsize=(8, 6))
huatude.plot(fpr, tpr, label=f"随机森林(AUC = {auc:.3f})")
huatude.plot([0, 1], [0, 1], "k--", label="随机猜测")
huatude.xlabel("假阳性率")
huatude.xlabel("真阳性率")
huatude.title("女胎染色体非整倍体检测的ROC曲线")
huatude.legend()
huatude.grid(True)
huatude.savefig(
    output_dir / "roc_curve_for_female_fetal_aneuploidy_detection.png", dpi=300
)
y_pred = rf.predict(X_scaled)
print("分类报告")
print(classification_report(y, y_pred, target_names=["正常", "异常"]))
female_data["rule_z3"] = female_data[z_features].apply(simple_z_rule, axis=1)
simple_acc = (female_data["rule_z3"] == y).mean()
print(f"\n经典Z>3规则准确率: {simple_acc:.3f}")
female_data["quality_weight"] = female_data.apply(quality_weight, axis=1)
for z_col in z_features:
    w_col = z_col.replace("Z值", "Z值_加权")
    female_data[w_col] = female_data[z_col] * female_data["quality_weight"]
female_data["rule_weighted"] = female_data.apply(weighted_rule, axis=1)
weighted_acc = (female_data["rule_weighted"] == y).mean()
print(f"加权Z>2.8规则准确率: {weighted_acc:.3f}")
