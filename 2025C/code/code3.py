import pandas as shuju_gongju
import numpy as suanshu_pack
import matplotlib.pyplot as huatude
import seaborn as huabiaode
import statsmodels.api as tongjigongju
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report
import warnings
import matplotlib

warnings.filterwarnings("ignore")
huabiaode.set_style("whitegrid")
fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")
fm.addfont("./times.ttf")
huatude.rcParams["font.sans-serif"] = ["FangSong_GB2312", "times"]
huatude.rcParams["axes.unicode_minus"] = False
output_dir = Path("C3_Output")
output_dir.mkdir(exist_ok=True)
df = shuju_gongju.read_csv("C1_Output/clean_male_data.csv")


def parse_gestational_week(gw_str):
    if shuju_gongju.isna(gw_str):
        return suanshu_pack.nan
    try:
        if "w+" in gw_str:
            week, day = gw_str.split("w+")
            return float(week) + float(day) / 7
        elif "w" in gw_str:
            return float(gw_str.replace("w", ""))
        else:
            return float(gw_str)
    except:
        return suanshu_pack.nan


df["孕周"] = df["检测孕周"].apply(parse_gestational_week)
df = df.dropna(subset=["孕周", "Y染色体浓度", "孕妇BMI", "年龄", "身高", "体重"])
df["Y达标"] = (df["Y染色体浓度"] >= 0.04).astype(int)
bmi_bins = [20, 28, 32, 36, 40, 50]
bmi_labels = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "≥40"]
df["BMI组"] = shuju_gongju.cut(
    df["孕妇BMI"], bins=bmi_bins, labels=bmi_labels, include_lowest=True
)
print(f"建模样本数：{len(df)}")
X = df[["孕周", "孕妇BMI", "年龄", "身高", "体重", "GC含量"]]
X = tongjigongju.add_constant(X)
y = df["Y达标"]
model = tongjigongju.Logit(y, X).fit(disp=False)
print(model.summary())
df["P_Y达标"] = model.predict(X)
auc = roc_auc_score(y, df["P_Y达标"])
print(f"模型AUC = {auc:.3f}")
print("各BMI组逻辑回归预测效果：")
group_auc = df.groupby("BMI组").apply(lambda g: roc_auc_score(g["Y达标"], g["P_Y达标"]))
print(group_auc)
huatude.figure(figsize=(10, 6))
huabiaode.scatterplot(data=df, x="P_Y达标", y="Y达标", alpha=0.6)
huatude.xlabel("预测达标概率")
huatude.ylabel("实际是否达标 (0/1)")
huatude.title("逻辑回归模型预测效果")
huatude.tight_layout()
huatude.savefig(output_dir / "logit_prediction_scatter.png", dpi=300)
huatude.figure(figsize=(12, 8))
weeks = suanshu_pack.arange(10, 26, 0.5)
for label in bmi_labels:
    group_data = df[df["BMI组"] == label]
    if len(group_data) == 0:
        continue
    sample_row = group_data.iloc[0]
    X_pred = shuju_gongju.DataFrame(
        {
            "孕周": weeks,
            "孕妇BMI": sample_row["孕妇BMI"],
            "年龄": sample_row["年龄"],
            "身高": sample_row["身高"],
            "体重": sample_row["体重"],
            "GC含量": sample_row["GC含量"],
        }
    )
    X_pred = tongjigongju.add_constant(X_pred, has_constant="add")
    prob = model.predict(X_pred)
    try:
        best_week = weeks[suanshu_pack.argmax(prob >= 0.9)]
    except:
        best_week = suanshu_pack.nan

    huatude.plot(weeks, prob, label=f"{label} (建议: {best_week:.1f}周)", linewidth=2.5)

huatude.axhline(y=0.9, color="r", linestyle="--", label="90%达标概率线")
huatude.xlabel("孕周（周）")
huatude.ylabel("预测Y浓度≥4%的概率")
huatude.title("不同BMI组Y染色体浓度达标概率预测曲线")
huatude.legend()
huatude.grid(True, alpha=0.3)
huatude.tight_layout()
huatude.savefig(output_dir / "predicted_attainment_curve.png", dpi=300)
best_timing_q3 = {}
for label in bmi_labels:
    group_data = df[df["BMI组"] == label]
    if len(group_data) == 0:
        best_timing_q3[label] = [suanshu_pack.nan, suanshu_pack.nan]
        continue
    sample_row = group_data.iloc[0]
    X_pred = shuju_gongju.DataFrame(
        {
            "孕周": weeks,
            "孕妇BMI": sample_row["孕妇BMI"],
            "年龄": sample_row["年龄"],
            "身高": sample_row["身高"],
            "体重": sample_row["体重"],
            "GC含量": sample_row["GC含量"],
        }
    )
    X_pred = tongjigongju.add_constant(X_pred, has_constant="add")
    prob = model.predict(X_pred)
    try:
        best_week = weeks[suanshu_pack.argmax(prob >= 0.9)]
    except:
        best_week = suanshu_pack.nan
    best_timing_q3[label] = [
        best_week,
        f"{int(best_week)}周{int((best_week-int(best_week))*7)}天",
    ]

best_timing_df = shuju_gongju.DataFrame(
    best_timing_q3, index=["最佳时点(周)", "建议时间"]
).T
print("\n问题3：综合模型最佳NIPT时点建议（P≥90%）")
print(best_timing_df)
best_timing_df.to_csv(output_dir / "best_nipt_timing_q3.csv")
print("\n检测误差影响分析...")
suanshu_pack.random.seed(42)
df["Y染色体浓度_噪声"] = df["Y染色体浓度"] * suanshu_pack.random.uniform(
    0.95, 1.05, len(df)
)
df["Y达标_噪声"] = (df["Y染色体浓度_噪声"] >= 0.04).astype(int)
model_noisy = tongjigongju.Logit(df["Y达标_噪声"], X).fit(disp=False)
df["P_Y达标_噪声"] = model_noisy.predict(X)

noise_auc = roc_auc_score(df["Y达标_噪声"], df["P_Y达标_噪声"])
print(f"加入±5%浓度误差后模型AUC = {noise_auc:.3f}（原为 {auc:.3f}）")
