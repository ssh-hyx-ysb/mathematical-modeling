import os  # 导入操作系统模块

# 使用别名导入关键库
import pandas as c_tongjitubiaogongju  # pandas - 数据处理
import numpy as c_shuxuejisuanqi  # numpy - 数值计算
import matplotlib.pyplot as c_huitu  # matplotlib - 数据可视化
from scipy import stats as c_tongji  # scipy.stats - 统计分析（含各类检验）
from scipy.stats import f_oneway  # ANOVA检验（多组差异）
import warnings  # 警告信息处理

warnings.filterwarnings("ignore")  # 忽略警告信息

import matplotlib  # 导入matplotlib

# 添加中文字体文件
fm = matplotlib.font_manager.fontManager
fm.addfont("./仿宋_GB2312.TTF")  # 添加仿宋字体
fm.addfont("./times.ttf")  # 添加Times New Roman字体

# 设置中文字体显示
c_huitu.rcParams["font.sans-serif"] = ["FangSong_GB2312"]  # 指定默认字体
c_huitu.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# -------------------------- 1. 数据读取与清洗（保留原有逻辑） --------------------------
# 读取Excel数据（使用别名c_tongjitubiaogongju）
c_dxygds_df = c_tongjitubiaogongju.read_excel("./附件.xlsx")

# 获取数值类型列
c_cololo_lie_ = c_dxygds_df.select_dtypes(include=[c_shuxuejisuanqi.number]).columns

# 计算Z-score检测异常值（仅标记，暂不删除，避免影响检验效力）
for col in c_cololo_lie_:
    # 计算绝对Z-score
    c_z_zhi = c_shuxuejisuanqi.abs(c_tongji.zscore(c_dxygds_df[col].dropna()))
    outliers = c_z_zhi > 3  # 标记异常值（Z值>3为极端异常）
    print(f"{col} 极端异常值数量: {outliers.sum()}")

# 开始数据清洗流程
print("\n=== 数据清洗 ===")
print("1. 缺失值处理")

# 处理缺失值
for col in c_dxygds_df.columns:
    if c_dxygds_df[col].isnull().sum() > 0:  # 检查是否存在缺失值
        missing_ratio = c_dxygds_df[col].isnull().sum() / len(c_dxygds_df)
        print(f"  {col}: 缺失比例 {missing_ratio:.2%}")
        if c_dxygds_df[col].dtype in ["int64", "float64"]:  # 数值型列
            # 用中位数填充（抗异常值）
            median_val = c_dxygds_df[col].median()
            c_dxygds_df[col].fillna(median_val, inplace=True)
            print(f"    用中位数 {median_val:.4f} 填充")
        else:  # 类别型列
            # 用众数填充
            mode_val = c_dxygds_df[col].mode()[0]
            c_dxygds_df[col].fillna(mode_val, inplace=True)
            print(f"    用众数 {mode_val} 填充")

# 删除重复行
print("\n2. 删除重复行")
duplicate_count = c_dxygds_df.duplicated().sum()
c_dxygds_df_cleaned = c_dxygds_df.drop_duplicates()
print(
    f"  原始数据行数: {len(c_dxygds_df)}, 去重后行数: {len(c_dxygds_df_cleaned)}, 删除重复行: {duplicate_count}"
)

# 保存清洗后的数据
c_dxygds_df_cleaned.to_csv("cleaned_shujubiaoao.csv", index=False, encoding="utf-8-sig")
print("\n数据清洗完成，清洗后数据已保存")


# -------------------------- 2. 日期转换与孕周计算（保留原有逻辑） --------------------------
# 转换日期格式（处理可能的格式异常）
def safe_to_datetime(series, format=None):
    """安全转换日期格式，无法转换的返回NaT"""
    try:
        return c_tongjitubiaogongju.to_datetime(series, format=format, errors="coerce")
    except:
        return c_tongjitubiaogongju.Series([c_tongjitubiaogongju.NaT] * len(series))


c_dxygds_df_cleaned["检测日期"] = safe_to_datetime(
    c_dxygds_df_cleaned["检测日期"], format="%Y%m%d"
)
c_dxygds_df_cleaned["末次月经"] = safe_to_datetime(c_dxygds_df_cleaned["末次月经"])

# 计算孕周（天数差除以7，保留2位小数）
c_dxygds_df_cleaned["孕周"] = (
    c_dxygds_df_cleaned["检测日期"] - c_dxygds_df_cleaned["末次月经"]
).dt.days / 7
c_dxygds_df_cleaned["孕周"] = c_dxygds_df_cleaned["孕周"].round(2)

# 过滤孕周合理范围（10-40周，排除极端值）
c_dxygds_df_cleaned = c_dxygds_df_cleaned[
    (c_dxygds_df_cleaned["孕周"] >= 10) & (c_dxygds_df_cleaned["孕周"] <= 40)
]
print(f"\n过滤后有效孕周样本数: {len(c_dxygds_df_cleaned)}")


# -------------------------- 3. 相关性分析 + P检验（验证相关显著性） --------------------------
print("\n=== 3. 相关性分析与P检验（验证相关显著性） ===")
# 定义分析变量（聚焦Y染色体浓度与关键指标的相关性）
corr_vars = ["Y染色体浓度", "孕周", "孕妇BMI", "GC含量", "原始读段数"]
# 创建无缺失值的分析数据集
corr_df = c_dxygds_df_cleaned[corr_vars].dropna()
print(f"相关性分析有效样本数: {len(corr_df)}")

# 1. 计算Pearson相关系数及P值（双尾检验）
corr_results = {}
for var1 in corr_vars:
    if var1 == "Y染色体浓度":  # 重点分析Y染色体浓度与其他指标的相关性
        for var2 in corr_vars:
            if var1 != var2:
                # Pearson相关系数 + P值（显著性检验）
                r, p = c_tongji.pearsonr(corr_df[var1], corr_df[var2])
                corr_results[(var1, var2)] = {"r": r, "p": p}
                # 输出结果（标注显著性：p<0.001***, p<0.01**, p<0.05*, p≥0.05ns）
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = "ns"
                print(f"{var1}与{var2}: r={r:.3f}, p={p:.4f} {sig}")

# 2. 绘制相关性热图（标注相关系数和显著性）
c_tutukuang, ffppt = c_huitu.subplots(2, 2, figsize=(12, 10))
c_tutukuang.suptitle(
    "Y染色体浓度与关键指标的相关性分析（含显著性标注）", fontsize=14, fontweight="bold"
)

# 子图1: Y染色体浓度 vs 孕周（标注相关系数和P值）
ffppt[0, 0].scatter(
    corr_df["孕周"], corr_df["Y染色体浓度"], alpha=0.6, s=30, color="#1f77b4"
)
ffppt[0, 0].set_xlabel("孕周")
ffppt[0, 0].set_ylabel("Y染色体浓度")
ffppt[0, 0].grid(True, alpha=0.3)
# 添加相关系数和显著性
r_week, p_week = (
    corr_results[("Y染色体浓度", "孕周")]["r"],
    corr_results[("Y染色体浓度", "孕周")]["p"],
)
sig_week = (
    "***"
    if p_week < 0.001
    else "**" if p_week < 0.01 else "*" if p_week < 0.05 else "ns"
)
ffppt[0, 0].text(
    0.05,
    0.95,
    f"r={r_week:.3f}, p={p_week:.4f} {sig_week}",
    transform=ffppt[0, 0].transAxes,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# 子图2: Y染色体浓度 vs BMI（标注相关系数和P值）
ffppt[0, 1].scatter(
    corr_df["孕妇BMI"], corr_df["Y染色体浓度"], alpha=0.6, s=30, color="#ff7f0e"
)
ffppt[0, 1].set_xlabel("孕妇BMI")
ffppt[0, 1].set_ylabel("Y染色体浓度")
ffppt[0, 1].grid(True, alpha=0.3)
r_bmi, p_bmi = (
    corr_results[("Y染色体浓度", "孕妇BMI")]["r"],
    corr_results[("Y染色体浓度", "孕妇BMI")]["p"],
)
sig_bmi = (
    "***" if p_bmi < 0.001 else "**" if p_bmi < 0.01 else "*" if p_bmi < 0.05 else "ns"
)
ffppt[0, 1].text(
    0.05,
    0.95,
    f"r={r_bmi:.3f}, p={p_bmi:.4f} {sig_bmi}",
    transform=ffppt[0, 1].transAxes,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)
<<<<<<< HEAD
ffppt[1, 0].set_xlabel("孕妇BMI")
ffppt[1, 0].set_ylabel("孕周")
ffppt[1, 0].grid(True, alpha=0.3)
=======
>>>>>>> d144980b28e5a9b567ca675aa931d8107bf2a1ec

# 子图3: Y染色体浓度 vs GC含量（标注相关系数和P值）
ffppt[1, 0].scatter(
    corr_df["GC含量"], corr_df["Y染色体浓度"], alpha=0.6, s=30, color="#2ca02c"
)
ffppt[1, 0].set_xlabel("GC含量")
ffppt[1, 0].set_ylabel("Y染色体浓度")
ffppt[1, 0].grid(True, alpha=0.3)
r_gc, p_gc = (
    corr_results[("Y染色体浓度", "GC含量")]["r"],
    corr_results[("Y染色体浓度", "GC含量")]["p"],
)
sig_gc = (
    "***" if p_gc < 0.001 else "**" if p_gc < 0.01 else "*" if p_gc < 0.05 else "ns"
)
ffppt[1, 0].text(
    0.05,
    0.95,
    f"r={r_gc:.3f}, p={p_gc:.4f} {sig_gc}",
    transform=ffppt[1, 0].transAxes,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# 子图4: 相关性矩阵热图（标注系数和显著性）
# 子图4: 相关性矩阵热图（标注系数和显著性）
corr_matrix = corr_df.corr()
im = ffppt[1, 1].imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
ffppt[1, 1].set_xticks(range(len(corr_matrix.columns)))
ffppt[1, 1].set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
ffppt[1, 1].set_yticks(range(len(corr_matrix.columns)))
ffppt[1, 1].set_yticklabels(corr_matrix.columns)

# 标注相关系数和显著性（核心修复部分）
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        r_val = corr_matrix.iloc[i, j]
        # 获取P值（统一以"Y染色体浓度"为第一个元素构建键）
        if (
            corr_matrix.columns[i] == "Y染色体浓度"
            and corr_matrix.columns[j] != "Y染色体浓度"
        ) or (
            corr_matrix.columns[j] == "Y染色体浓度"
            and corr_matrix.columns[i] != "Y染色体浓度"
        ):
            # 确定正确的键：始终将"Y染色体浓度"放在前面
            if corr_matrix.columns[i] == "Y染色体浓度":
                key = (corr_matrix.columns[i], corr_matrix.columns[j])
            else:
                key = (corr_matrix.columns[j], corr_matrix.columns[i])
            # 读取P值（此时key一定在corr_results中）
            p_val = corr_results[key]["p"] if key in corr_results else 1.0
            # 显著性标注
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "ns"
            label = f"{r_val:.2f}\n{sig}"
        else:
            label = f"{r_val:.2f}"
        ffppt[1, 1].text(
            j, i, label, ha="center", va="center", color="#000000", fontweight="bold"
        )
# 添加颜色条
c_huitu.colorbar(im, ax=ffppt[1, 1])
ffppt[1, 1].set_title("相关系数矩阵（标注显著性）")

# 保存相关性图表
c_huitu.tight_layout()
c_huitu.savefig(
    "fig-1-1-y_chromosome_correlation_with_sig.png", dpi=300, bbox_inches="tight"
)
print("\n相关性分析图表已保存（含显著性标注）")


# -------------------------- 4. BMI分组趋势 + ANOVA+T检验（验证组间差异显著性） --------------------------
print("\n=== 4. BMI分组差异检验（ANOVA + 两两T检验） ===")
# 创建BMI分组（沿用原有分组逻辑）
bmi_bins = [0, 18.5, 24, 28, 50]  # 偏瘦, 正常, 超重, 肥胖
bmi_labels = ["偏瘦", "正常", "超重", "肥胖"]
c_dxygds_df_cleaned["BMI分组"] = c_tongjitubiaogongju.cut(
    c_dxygds_df_cleaned["孕妇BMI"], bins=bmi_bins, labels=bmi_labels
)

# 筛选有Y染色体浓度数据的样本（男胎数据）
bmi_analysis_df = c_dxygds_df_cleaned.dropna(subset=["Y染色体浓度", "BMI分组", "孕周"])
# 按BMI分组统计样本数
bmi_group_counts = bmi_analysis_df["BMI分组"].value_counts().sort_index()
print(f"BMI分组样本分布: {dict(bmi_group_counts)}")

# 1. 按BMI分组和孕周计算Y染色体浓度均值（用于趋势图）
bmi_week_mean = (
    bmi_analysis_df.groupby(["BMI分组", "孕周"])["Y染色体浓度"].mean().reset_index()
)

# 2. ANOVA检验：多组（BMI分组）间Y染色体浓度是否存在显著差异
# 提取各组Y染色体浓度数据（仅保留样本数≥5的组，保证检验效力）
valid_bmi_groups = [
    group for group in bmi_labels if bmi_group_counts.get(group, 0) >= 5
]
group_data = [
    bmi_analysis_df[bmi_analysis_df["BMI分组"] == group]["Y染色体浓度"].values
    for group in valid_bmi_groups
]
# 执行单因素ANOVA
f_stat, p_anova = f_oneway(*group_data)
print(f"ANOVA检验（BMI分组间Y染色体浓度差异）: F={f_stat:.3f}, p={p_anova:.4f}")
if p_anova < 0.001:
    anova_sig = "***（极显著）"
elif p_anova < 0.01:
    anova_sig = "**（显著）"
elif p_anova < 0.05:
    anova_sig = "*（边际显著）"
else:
    anova_sig = "ns（不显著）"
print(f"结论：BMI分组间Y染色体浓度差异 {anova_sig}")

# 3. 两两T检验（事后检验，Bonferroni校正控制I类错误）
print("\n两两T检验（Bonferroni校正）:")
from itertools import combinations

pairwise_results = []
for group1, group2 in combinations(valid_bmi_groups, 2):
    data1 = bmi_analysis_df[bmi_analysis_df["BMI分组"] == group1][
        "Y染色体浓度"
    ].dropna()
    data2 = bmi_analysis_df[bmi_analysis_df["BMI分组"] == group2][
        "Y染色体浓度"
    ].dropna()
    # 独立样本T检验（假设方差不齐，用equal_var=False）
    t_stat, p_t = c_tongji.ttest_ind(data1, data2, equal_var=False)
    # Bonferroni校正（多组比较次数：k*(k-1)/2）
    n_comparisons = len(valid_bmi_groups) * (len(valid_bmi_groups) - 1) // 2
    p_corrected = p_t * n_comparisons
    # 记录结果
    pairwise_results.append(
        {
            "分组": f"{group1} vs {group2}",
            "t": t_stat,
            "原始p": p_t,
            "校正后p": p_corrected,
            "显著性": (
                "***"
                if p_corrected < 0.001
                else "**" if p_corrected < 0.01 else "*" if p_corrected < 0.05 else "ns"
            ),
        }
    )
    print(
        f"  {group1} vs {group2}: t={t_stat:.3f}, 校正后p={p_corrected:.4f} {pairwise_results[-1]['显著性']}"
    )

# 4. 绘制BMI分组趋势图（标注ANOVA显著性）
c_huitu.figure(figsize=(12, 8))
colors = ["#00ffe4", "#3400ff", "#ff0000", "#67ff00"]
for i, group in enumerate(valid_bmi_groups):
    group_data_trend = bmi_week_mean[bmi_week_mean["BMI分组"] == group]
    if len(group_data_trend) > 1:  # 确保有足够数据点绘制趋势
        c_huitu.plot(
            group_data_trend["孕周"],
            group_data_trend["Y染色体浓度"],
            label=f"{group}（n={bmi_group_counts[group]}）",
            color=colors[i],
            linewidth=2.5,
            marker="o",
            markersize=4,
        )

# 设置图表属性（标注ANOVA结果）
c_huitu.xlabel("孕周", fontsize=14)
c_huitu.ylabel("Y染色体浓度均值", fontsize=14)
c_huitu.title(
    f"不同BMI分组的Y染色体浓度随孕周变化趋势\nANOVA: F={f_stat:.3f}, p={p_anova:.4f} {anova_sig}",
    fontsize=16,
    fontweight="bold",
)
c_huitu.legend(title="BMI分组（样本数）", fontsize=12)
c_huitu.grid(True, alpha=0.3)
# 添加达标阈值（4%=0.04）
c_huitu.axhline(
    y=0.04, color="gray", linestyle="--", alpha=0.7, label="Y染色体浓度达标阈值（4%）"
)
c_huitu.legend()

# 保存趋势图
c_huitu.tight_layout()
c_huitu.savefig(
    "fig-1-2-y_chromosome_by_bmi_week_with_anova.png", dpi=300, bbox_inches="tight"
)
print("\nBMI分组趋势图已保存（含ANOVA显著性标注）")


# -------------------------- 5. 染色体Z值分布 + 正态性检验（Shapiro-Wilk） --------------------------
print("\n=== 5. 染色体Z值正态性检验（Shapiro-Wilk检验） ===")
# 定义Z值列（排除Y染色体Z值，女胎数据为空）
z_value_cols = ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值"]

# 创建画布和子图
c_tutukuang, ffppt = c_huitu.subplots(2, 2, figsize=(15, 10))
c_tutukuang.suptitle(
    "染色体Z值分布与正态性检验（Shapiro-Wilk）", fontsize=16, fontweight="bold"
)
ffppt = ffppt.flatten()  # 展平子图数组

# 绘制各Z值分布并进行正态性检验
norm_test_results = {}
for i, col in enumerate(z_value_cols):
    # 移除极端异常值（1%和99%分位数）
    z_data = c_dxygds_df_cleaned[col].dropna()
    q1 = z_data.quantile(0.01)
    q3 = z_data.quantile(0.99)
    filtered_z = z_data[(z_data >= q1) & (z_data <= q3)]
    print(f"\n{col}: 有效样本数={len(filtered_z)}")

    # 1. Shapiro-Wilk正态性检验（适用于n<5000，若n≥5000用Kolmogorov-Smirnov）
    if len(filtered_z) < 5000:
        stat_shapiro, p_shapiro = c_tongji.shapiro(filtered_z)
        test_name = "Shapiro-Wilk"
    else:
        stat_shapiro, p_shapiro = c_tongji.kstest(
            filtered_z, "norm", args=(filtered_z.mean(), filtered_z.std())
        )
        test_name = "Kolmogorov-Smirnov"
    norm_test_results[col] = {"stat": stat_shapiro, "p": p_shapiro, "test": test_name}

    # 输出正态性检验结果
    if p_shapiro < 0.05:
        norm_conclusion = "不符合正态分布"
        norm_sig = "*"
    else:
        norm_conclusion = "符合正态分布"
        norm_sig = "ns"
    print(
        f"  {test_name}检验: stat={stat_shapiro:.3f}, p={p_shapiro:.4f} {norm_sig}, 结论：{norm_conclusion}"
    )

    # 2. 绘制直方图（含正态分布拟合曲线）
    ffppt[i].hist(
        filtered_z, bins=30, alpha=0.7, color=f"C{i}", edgecolor="#000000", density=True
    )
    # 拟合正态分布曲线
    mean_z = filtered_z.mean()
    std_z = filtered_z.std()
    x_norm = c_shuxuejisuanqi.linspace(filtered_z.min(), filtered_z.max(), 100)
    y_norm = c_tongji.norm.pdf(x_norm, loc=mean_z, scale=std_z)
    ffppt[i].plot(
        x_norm, y_norm, color="red", linestyle="--", linewidth=2, label="理论正态分布"
    )

    # 设置子图属性（标注正态性检验结果）
    ffppt[i].set_xlabel("Z值")
    ffppt[i].set_ylabel("概率密度")
    ffppt[i].set_title(f"{col}\n{test_name}: p={p_shapiro:.4f} {norm_sig}", fontsize=10)
    ffppt[i].grid(True, alpha=0.3)
    ffppt[i].legend()
    # 添加均值和标准差线
    ffppt[i].axvline(
        mean_z,
        color="darkred",
        linestyle="-",
        linewidth=1.5,
        label=f"均值: {mean_z:.2f}",
    )
    ffppt[i].axvline(
        mean_z + 2 * std_z,
        color="orange",
        linestyle=":",
        linewidth=1,
        label=f"±2σ: {mean_z+2*std_z:.2f}",
    )
    ffppt[i].axvline(mean_z - 2 * std_z, color="orange", linestyle=":", linewidth=1)
    ffppt[i].legend(fontsize=8)

# 保存Z值分布图
c_huitu.tight_layout()
c_huitu.savefig(
    "fig-1-3-chromosome_zvalue_distribution_with_normtest.png",
    dpi=300,
    bbox_inches="tight",
)
print("\n染色体Z值分布图已保存（含正态性检验结果）")


# -------------------------- 6. 质量控制指标 + 组间差异检验（ANOVA） --------------------------
print("\n=== 6. 质量控制指标组间差异检验（ANOVA） ===")
# 定义质量控制列（清理列名空格）
qc_cols = [
    "GC含量",
    "原始读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "唯一比对的读段数",
    "被过滤掉读段数的比例",
]
# 清理列名（去除多余空格）
c_dxygds_df_cleaned.columns = [col.strip() for col in c_dxygds_df_cleaned.columns]

# 创建画布和子图
c_tutukuang, ffppt = c_huitu.subplots(2, 3, figsize=(15, 10))
c_tutukuang.suptitle(
    "质量控制指标分布与BMI组间差异（ANOVA）", fontsize=16, fontweight="bold"
)
ffppt = ffppt.flatten()  # 展平子图数组

# 分析每个QC指标在不同BMI组间的差异
qc_anova_results = {}
for i, col in enumerate(qc_cols):
    if i < len(ffppt) and col in c_dxygds_df_cleaned.columns:
        # 筛选有效数据
        qc_data = c_dxygds_df_cleaned.dropna(subset=[col, "BMI分组"])
        # 按BMI分组提取数据（仅保留样本数≥3的组）
        qc_group_data = [
            qc_data[qc_data["BMI分组"] == group][col].values
            for group in valid_bmi_groups
            if len(qc_data[qc_data["BMI分组"] == group]) >= 3
        ]

        # ANOVA检验逻辑（保持不变）
        if len(qc_group_data) >= 2:
            f_qc, p_qc = f_oneway(*qc_group_data)
            qc_anova_results[col] = {"F": f_qc, "p": p_qc}
            if p_qc < 0.001:
                qc_sig = "***"
            elif p_qc < 0.01:
                qc_sig = "**"
            elif p_qc < 0.05:
                qc_sig = "*"
            else:
                qc_sig = "ns"
            print(f"{col}: F={f_qc:.3f}, p={p_qc:.4f} {qc_sig}")
        else:
            p_qc = 1.0
            qc_sig = "ns"
            print(f"{col}: 样本量不足，未进行ANOVA")

        # 绘制箱线图（修复部分）
        box_data = [
            qc_data[qc_data["BMI分组"] == group][col].dropna()
            for group in valid_bmi_groups
        ]
        bp = ffppt[i].boxplot(box_data, labels=valid_bmi_groups, patch_artist=True)

        # 1. 设置箱线图颜色（保持不变）
        for patch, color in zip(bp["boxes"], colors[: len(valid_bmi_groups)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 2. 处理比例/百分比数据（修复：从原始数据计算范围）
        if "比例" in col or "含量" in col:
            ffppt[i].set_ylabel(f"{col} (%)")
            # 合并所有组的原始数据，计算整体范围
            all_data = c_shuxuejisuanqi.concatenate(
                [data.values for data in box_data if len(data) > 0]
            )
            if len(all_data) > 0:
                # 扩展5%的范围，避免数据贴边
                y_min = all_data.min() * 0.95
                y_max = all_data.max() * 1.05
                ffppt[i].set_ylim(y_min, y_max)
        else:
            ffppt[i].set_ylabel(col)

        # 3. 标注ANOVA结果（保持不变）
        ffppt[i].set_title(f"{col}\nANOVA: p={p_qc:.4f} {qc_sig}", fontsize=10)
        ffppt[i].grid(True, alpha=0.3, axis="y")
        ffppt[i].tick_params(axis="x", rotation=45)

# 移除多余子图
if len(qc_cols) < len(ffppt):
    for j in range(len(qc_cols), len(ffppt)):
        c_tutukuang.delaxes(ffppt[j])

# 保存QC指标图
c_huitu.tight_layout()
c_huitu.savefig("fig-1-4-quality_control_with_anova.png", dpi=300, bbox_inches="tight")
print("\n质量控制指标图已保存（含BMI组间ANOVA检验结果）")


print("\n=== 所有分析完成 ===")
print("生成文件清单：")
print("1. cleaned_shujubiaoao.csv - 清洗后的数据")
print("2. fig-1-1-y_chromosome_correlation_with_sig.png - 相关性分析（含P检验）")
print(
    "3. fig-1-2-y_chromosome_by_bmi_week_with_anova.png - BMI分组趋势（含ANOVA+T检验）"
)
print(
    "4. fig-1-3-chromosome_zvalue_distribution_with_normtest.png - Z值分布（含正态性检验）"
)
print("5. fig-1-4-quality_control_with_anova.png - 质量指标（含组间ANOVA）")
