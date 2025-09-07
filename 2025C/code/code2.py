import pandas as shuju_gongju
import numpy as suanshu_pack
import matplotlib.pyplot as huatude
import seaborn as huabiaode
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
huabiaode.set_style("whitegrid")
import matplotlib

fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")
fm.addfont("./times.ttf")
huatude.rcParams["font.sans-serif"] = ["FangSong_GB2312", "times"]
huatude.rcParams["axes.unicode_minus"] = False
output_dir = Path("C2_Output")
output_dir.mkdir(exist_ok=True)
df = shuju_gongju.read_csv("C1_Output/clean_male_data.csv")
assert "Y染色体浓度" in df.columns, "缺少Y染色体浓度列"
assert "检测孕周" in df.columns, "缺少检测孕周列"
assert "孕妇BMI" in df.columns, "缺少BMI列"
print(f"当前男胎样本数：{len(df)}")


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
df = df.dropna(subset=["孕周", "Y染色体浓度", "孕妇BMI"])
df = df[(df["孕周"] >= 8) & (df["孕周"] <= 28)]
print(f"清洗后有效样本数：{len(df)}")
THRESHOLD = 0.04
df["Y达标"] = df["Y染色体浓度"] >= THRESHOLD
first达标 = (
    df[df["Y达标"]]
    .groupby("孕妇代码")
    .agg({"孕周": "min"})
    .rename(columns={"孕周": "首次达标孕周"})
)
df = df.merge(first达标, on="孕妇代码", how="left")
df = df.sort_values("孕周").groupby("孕妇代码").first().reset_index()
print(f"可用于分组的孕妇数：{len(df)}")
bmi_bins = [20, 28, 32, 36, 40, 100]  # 扩展范围
bmi_labels = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "≥40"]
df["BMI组"] = shuju_gongju.cut(
    df["孕妇BMI"], bins=bmi_bins, labels=bmi_labels, include_lowest=True
)
group_stats = (
    df.groupby("BMI组")
    .agg(
        n_samples=("孕妇代码", "size"),
        mean_attainment=("首次达标孕周", "mean"),
        median_attainment=("首次达标孕周", "median"),
        q80_attainment=("首次达标孕周", lambda x: x.quantile(0.8)),
        q90_attainment=("首次达标孕周", lambda x: x.quantile(0.9)),
    )
    .round(2)
)
group_stats.columns = [
    "孕妇数量",
    "平均首次达标孕周",
    "中位首次达标孕周",
    "80分位达标孕周",
    "90分位达标孕周",
]
group_stats.to_csv(output_dir / "bmi_group_first_attainment.csv")
df.to_csv(output_dir / "bmi_grouped_data.csv", index=False)
huatude.figure(figsize=(12, 6))
huabiaode.boxplot(data=df, x="BMI组", y="首次达标孕周")
huatude.title("各BMI组Y染色体浓度首次≥4%的孕周分布")
huatude.ylabel("首次达标孕周（周）")
huatude.xlabel("BMI分组")
huatude.xticks(rotation=0)
huatude.tight_layout()
huatude.savefig(output_dir / "first_attainment_by_bmi.png", dpi=300)
huatude.figure(figsize=(12, 7))
for label in bmi_labels:
    data = df[df["BMI组"] == label]
    if len(data) == 0:
        continue
    weeks = suanshu_pack.arange(10, 26, 0.5)
    attainment_ratio = []
    for w in weeks:
        ratio = (data[data["孕周"] <= w]["Y达标"].mean()) * 100
        attainment_ratio.append(ratio)
    huatude.plot(weeks, attainment_ratio, label=f"{label}", linewidth=2.5)

huatude.axhline(y=90, color="r", linestyle="--", label="90%达标线")
huatude.axhline(y=80, color="orange", linestyle="--", label="80%达标线")
huatude.xlabel("孕周（周）")
huatude.ylabel("Y浓度≥4%的比例（%）")
huatude.title("不同BMI组Y染色体浓度达标比例随孕周变化")
huatude.legend()
huatude.grid(True, alpha=0.3)
huatude.tight_layout()
huatude.savefig(output_dir / "attainment_ratio_by_bmi.png", dpi=300)
best_timing = group_stats[["80分位达标孕周"]].copy()
print("80分位达标孕周（含NaN）：")
print(group_stats["80分位达标孕周"])
best_timing.columns = ["最佳NIPT时点（周）"]
best_timing["建议检测时间"] = best_timing["最佳NIPT时点（周）"].apply(
    lambda x: f"{int(x)}周{int((x-int(x))*7)}天"
)
print(best_timing)
best_timing.to_csv(output_dir / "best_nipt_timing_q2.csv")
best_timing["时点下限"] = best_timing["最佳NIPT时点（周）"] - 0.5
best_timing["时点上限"] = best_timing["最佳NIPT时点（周）"] + 0.5
suanshu_pack.random.seed(42)
df_noisy = df.copy()
df_noisy["Y染色体浓度_噪声"] = df_noisy["Y染色体浓度"] * suanshu_pack.random.uniform(
    0.95, 1.05, len(df_noisy)
)
df_noisy["Y达标_噪声"] = df_noisy["Y染色体浓度_噪声"] >= THRESHOLD

first达标_噪声 = df_noisy[df_noisy["Y达标_噪声"]].groupby("孕妇代码")["孕周"].min()
df = df.merge(first达标_噪声, on="孕妇代码", how="left", suffixes=("", "_噪声"))
noise_impact = df.groupby("BMI组")["首次达标孕周"].quantile(0.8).round(2)
print("\n加入±5%浓度测量误差后，各组80%分位达标时间变化：")
print(noise_impact)
