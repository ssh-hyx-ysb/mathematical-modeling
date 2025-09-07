import pandas as shuju_gongju
import numpy as suanshu_pack
import matplotlib.pyplot as huatude
import seaborn as huabiaode
from scipy import stats as tongjide
import statsmodels.api as tongjigongju
from statsmodels.formula.api import ols, mixedlm
import os
from pathlib import Path
import matplotlib

huabiaode.set(style="whitegrid")
fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")
fm.addfont("./times.ttf")
huatude.rcParams["font.sans-serif"] = ["FangSong_GB2312", "times"]
huatude.rcParams["axes.unicode_minus"] = False
output_dir = Path("C1_Output")
output_dir.mkdir(exist_ok=True)
nantai_wenjian = "附件 - 男胎检测数据.xlsx"
nuitai_wenjian = "附件 - 女胎检测数据.xlsx"
nantai_biao = shuju_gongju.read_excel(nantai_wenjian, sheet_name=None, header=None)
nuitai_biao = shuju_gongju.read_excel(nuitai_wenjian, sheet_name=None, header=None)
data_frames = []
for name, df in nantai_biao.items():

    header_row = df[
        df.apply(lambda x: x.astype(str).str.contains("序号", na=False).any(), axis=1)
    ]
    if not header_row.empty:
        suoyin = header_row.index[0]
        df.columns = df.iloc[suoyin]
        df = df.iloc[suoyin + 1 :].reset_index(drop=True)
        data_frames.append(df)

for name, df in nuitai_biao.items():
    header_row = df[
        df.apply(lambda x: x.astype(str).str.contains("序号", na=False).any(), axis=1)
    ]
    if not header_row.empty:
        suoyin = header_row.index[0]
        df.columns = df.iloc[suoyin]
        df = df.iloc[suoyin + 1 :].reset_index(drop=True)
        data_frames.append(df)
df = shuju_gongju.concat(data_frames, ignore_index=True)
print(f"合并后总样本数：{len(df)}")
original_columns = df.columns.tolist()
numeric_cols = [
    "年龄",
    "身高",
    "体重",
    "孕妇BMI",
    "原始读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "唯一比对的读段数",
    "GC含量",
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值",
    "X染色体的Z值",
    "Y染色体的Z值",
    "Y染色体浓度",
    "X染色体浓度",
    "13号染色体的GC含量",
    "18号染色体的GC含量",
    "21号染色体的GC含量",
    "被过滤掉读段数的比例",
]
df[numeric_cols] = df[numeric_cols].apply(shuju_gongju.to_numeric, errors="coerce")


def parse_gestational_week(gw_str):
    try:
        if shuju_gongju.isna(gw_str):
            return suanshu_pack.nan
        gw_str = str(gw_str).strip()
        if "w" in gw_str:
            parts = gw_str.split("w+")
            week = float(parts[0])
            day = float(parts[1]) if len(parts) > 1 else 0
            return week + day / 7
        elif "W" in gw_str:
            parts = gw_str.split("W+")
            week = float(parts[0])
            day = float(parts[1]) if len(parts) > 1 else 0
            return week + day / 7
        else:
            return float(ga_str)
    except:
        return suanshu_pack.nan


df["孕周"] = df["检测孕周"].apply(parse_gestational_week)
df["计算BMI"] = df["体重"] / (df["身高"] / 100) ** 2
bmi_diff = suanshu_pack.abs(df["计算BMI"] - df["孕妇BMI"])
print(f"BMI验证：最大差异 = {bmi_diff.max():.4f}")
if bmi_diff.max() > 0.1:
    print("存在BMI计算不一致，请检查")
df["BMI"] = df["孕妇BMI"]
nanbiao = df[df["Y染色体浓度"].notna() & (df["Y染色体浓度"] > 0)].copy()
print(f"男胎样本数量：{len(nanbiao)}")
print("\n数据清洗：剔除GC含量异常、测序深度过低样本...")
clean_df = nanbiao[
    (nanbiao["GC含量"].between(0.35, 0.65))
    & (nanbiao["原始读段数"] > 3_000_000)
    & (nanbiao["被过滤掉读段数的比例"] < 0.1)
    & (nanbiao["孕周"].between(8, 28))
    & (nanbiao["Y染色体浓度"] <= 15)
].copy()
print(f"清洗后男胎样本数：{len(clean_df)}")
clean_df.to_csv(output_dir / "clean_male_data.csv", index=False, encoding="utf-8-sig")
huatude.figure(figsize=(10, 8))
corr = clean_df[["Y染色体浓度", "孕周", "BMI", "年龄"]].corr()
huabiaode.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True)
huatude.title("Y染色体浓度与各变量相关性热图")
huatude.tight_layout()
huatude.savefig(output_dir / "correlation_heatmap.png", dpi=300)
huatude.figure(figsize=(10, 6))
huabiaode.scatterplot(
    data=clean_df, x="孕周", y="Y染色体浓度", hue="BMI", palette="viridis", alpha=0.7
)
huatude.title("Y染色体浓度 vs 孕周（颜色表示BMI）")
huatude.xlabel("孕周（周）")
huatude.ylabel("Y染色体浓度 (%)")
huatude.legend(title="BMI", bbox_to_anchor=(1.05, 1), loc="upper left")
huatude.tight_layout()
huatude.savefig(output_dir / "scatter_y_vs_gw_by_bmi.png", dpi=300)
huatude.figure(figsize=(8, 5))
huabiaode.histplot(clean_df["Y染色体浓度"], kde=True)
huatude.title("Y染色体浓度分布")
huatude.xlabel("Y染色体浓度 (%)")
huatude.tight_layout()
huatude.savefig(output_dir / "hist_y_concentration.png", dpi=300)
huatude.figure(figsize=(8, 5))
huabiaode.histplot(clean_df["孕周"], kde=True, color="skyblue")
huatude.title("孕周分布")
huatude.xlabel("孕周（周）")
huatude.tight_layout()
huatude.savefig(output_dir / "hist_gestational_week.png", dpi=300)
moxingde_shuju = clean_df[["Y染色体浓度", "孕周", "BMI"]].dropna().copy()
print(f"建模样本数：{len(moxingde_shuju)}")
moxingde_shuju["孕周_2"] = moxingde_shuju["孕周"] ** 2
moxingde_shuju["孕周_BMI"] = moxingde_shuju["孕周"] * moxingde_shuju["BMI"]
print("\n模型1：线性回归 Y ~ 孕周 + BMI")
X1 = tongjigongju.add_constant(moxingde_shuju[["孕周", "BMI"]])
y = moxingde_shuju["Y染色体浓度"]
model1 = tongjigongju.OLS(y, X1).fit()
print(model1.summary())
with open(output_dir / "model1_summary.txt", "w", encoding="utf-8") as f:
    f.write(model1.summary().as_text())
print("\n模型2：多项式 + 交互项 Y ~ 孕周 + 孕周² + BMI + 孕周:BMI")
X2 = tongjigongju.add_constant(moxingde_shuju[["孕周", "孕周_2", "BMI", "孕周_BMI"]])
model2 = tongjigongju.OLS(y, X2).fit()
print(model2.summary())
with open(output_dir / "model2_summary.txt", "w", encoding="utf-8") as f:
    f.write(model2.summary().as_text())
fig, axes = huatude.subplots(2, 2, figsize=(12, 10))
axes[0, 0].scatter(model2.fittedvalues, model2.resid, alpha=0.6)
axes[0, 0].hlines(
    0,
    poly_model.fittedvalues.min(),
    poly_model.fittedvalues.max(),
    colors="r",
    linestyles="dashed",
)
axes[0, 0].set_xlabel("拟合值")
axes[0, 0].set_ylabel("残差")
axes[0, 0].set_title("残差 vs 拟合值（检验异方差）")
residuals_norm = (model2.resid - model2.resid.mean()) / model2.resid.std()
tongjide.probplot(residuals_norm, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title("Q-Q 图（检验残差正态性）")
axes[1, 0].scatter(
    moxingde_shuju["孕周"],
    moxingde_shuju["Y染色体浓度"],
    alpha=0.6,
    label="实际值",
    color="blue",
)
sorted_suoyin = suanshu_pack.argsort(moxingde_shuju["孕周"])
axes[1, 0].plot(
    moxingde_shuju["孕周"].iloc[sorted_suoyin],
    model2.fittedvalues.iloc[sorted_suoyin],
    color="red",
    label="拟合曲线",
)
axes[1, 0].set_xlabel("孕周")
axes[1, 0].set_ylabel("Y染色体浓度 (%)")
axes[1, 0].set_title("Y染色体浓度与孕周关系（拟合效果）")
axes[1, 0].legend()
axes[1, 1].scatter(
    moxingde_shuju["BMI"], moxingde_shuju["Y染色体浓度"], alpha=0.6, color="green"
)
axes[1, 1].set_xlabel("BMI")
axes[1, 1].set_ylabel("Y染色体浓度 (%)")
axes[1, 1].set_title("Y浓度 vs BMI")
huatude.tight_layout()
huatude.savefig(output_dir / "residual_diagnostics.png", dpi=300)
moxingde_shuju_with_id = (
    clean_df[["Y染色体浓度", "孕周", "BMI", "孕妇代码"]].dropna().copy()
)
moxingde_shuju_with_id["孕周_2"] = moxingde_shuju_with_id["孕周"] ** 2
moxingde_shuju_with_id["孕周_BMI"] = (
    moxingde_shuju_with_id["孕周"] * moxingde_shuju_with_id["BMI"]
)
try:
    mixed_model = mixedlm(
        "Y染色体浓度 ~ 孕周 + np.power(孕周, 2) + BMI + 孕周:BMI",
        moxingde_shuju_with_id,
        groups=moxingde_shuju_with_id["孕妇代码"],
    )
    mixed_result = mixed_model.fit()
    print(mixed_result.summary())
    with open(output_dir / "mixed_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(mixed_result.summary().as_text())
    print("混合效应模型结果已保存")
except Exception as e:
    print(f"混合模型拟合失败：{e}")
