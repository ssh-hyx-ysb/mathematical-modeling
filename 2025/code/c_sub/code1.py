import os  # 导入操作系统模块

# 使用别名导入关键库
import pandas as c_tongjitubiaogongju  # pandas - 数据处理
import numpy as c_shuxuejisuanqi  # numpy - 数值计算
import matplotlib.pyplot as c_huitu  # matplotlib - 数据可视化
from scipy import stats as c_tongji  # scipy.stats - 统计分析
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


# 读取Excel数据（使用别名c_tongjitubiaogongju）
c_dxygds_df = c_tongjitubiaogongju.read_excel("./附件.xlsx")

# 获取数值类型列
c_cololo_lie_ = c_dxygds_df.select_dtypes(include=[c_shuxuejisuanqi.number]).columns

# 计算Z-score检测异常值
for col in c_cololo_lie_:
    # 计算绝对Z-score
    c_z_zhi = c_shuxuejisuanqi.abs(c_tongji.zscore(c_dxygds_df[col].dropna()))
    outliers = c_z_zhi > 3  # 标记异常值


# 开始数据清洗流程
print("数据清洗。")
print("缺失值处理。")

# 处理缺失值
for col in c_dxygds_df.columns:
    if c_dxygds_df[col].isnull().sum() > 0:  # 检查是否存在缺失值
        if c_dxygds_df[col].dtype in ["int64", "float64"]:  # 数值型列
            # 用中位数填充
            c_dxygds_df[col].fillna(c_dxygds_df[col].median(), inplace=True)
        else:  # 类别型列
            # 用众数填充
            c_dxygds_df[col].fillna(c_dxygds_df[col].mode()[0], inplace=True)

# 删除重复行
print("删除重复行")
c_dxygds_df_cleauc_tongjitubiaogongjuatened = c_dxygds_df.drop_duplicates()

# 保存清洗后的数据
c_dxygds_df_cleauc_tongjitubiaogongjuatened.to_csv(
    "cleaned_shujubiaoao.csv", index=False, encoding="utf-8-sig"
)
print("数据清洗完成。")

# 转换日期格式
c_dxygds_df_cleauc_tongjitubiaogongjuatened["检测日期"] = (
    c_tongjitubiaogongju.to_datetime(
        c_dxygds_df_cleauc_tongjitubiaogongjuatened["检测日期"], format="%Y%m%d"
    )
)
c_dxygds_df_cleauc_tongjitubiaogongjuatened["末次月经"] = (
    c_tongjitubiaogongju.to_datetime(
        c_dxygds_df_cleauc_tongjitubiaogongjuatened["末次月经"]
    )
)

# 计算孕周（天数差除以7）
c_dxygds_df_cleauc_tongjitubiaogongjuatened["孕周"] = (
    c_dxygds_df_cleauc_tongjitubiaogongjuatened["检测日期"]
    - c_dxygds_df_cleauc_tongjitubiaogongjuatened["末次月经"]
).dt.days / 7

# 定义分析变量
huizhitudiandbianliang = ["Y染色体浓度", "孕周", "孕妇BMI"]

# 创建分析数据集（删除缺失值）
scatter_c_dxygds_df = c_dxygds_df_cleauc_tongjitubiaogongjuatened[
    huizhitudiandbianliang
].dropna()

# 创建画布和子图
c_tutukuang, ffppt = c_huitu.subplots(2, 2, figsize=(12, 10))
c_tutukuang.suptitle("Y染色体浓度与孕周、BMI的相关性分析")

# 绘制各散点图
# Y染色体浓度 vs 孕周
ffppt[0, 0].scatter(
    scatter_c_dxygds_df["孕周"], scatter_c_dxygds_df["Y染色体浓度"], alpha=0.6, s=30
)
ffppt[0, 0].set_xlabel("孕周")
ffppt[0, 0].set_ylabel("Y染色体浓度")
ffppt[0, 0].grid(True, alpha=0.3)

# Y染色体浓度 vs BMI
ffppt[0, 1].scatter(
    scatter_c_dxygds_df["孕妇BMI"],
    scatter_c_dxygds_df["Y染色体浓度"],
    alpha=0.6,
    s=30,
    color="orange",
)
ffppt[0, 1].set_xlabel("孕妇BMI")
ffppt[0, 1].set_ylabel("Y染色体浓度")
ffppt[0, 1].grid(True, alpha=0.3)

# 孕周 vs BMI
ffppt[1, 0].scatter(
    scatter_c_dxygds_df["孕妇BMI"],
    scatter_c_dxygds_df["孕周"],
    alpha=0.6,
    s=30,
    color="green",
)
ffppt[1, 0].set_xlabel("孕妇BMI")
ffppt[1, 0].set_ylabel("孕周")
ffppt[1, 0].grid(True, alpha=0.3)

# 相关系数矩阵热图
corr_matrix = scatter_c_dxygds_df.corr()
imamagiiinat = ffppt[1, 1].imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
ffppt[1, 1].set_xticks(range(len(corr_matrix.columns)))
ffppt[1, 1].set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
ffppt[1, 1].set_yticks(range(len(corr_matrix.columns)))
ffppt[1, 1].set_yticklabels(corr_matrix.columns)

# 添加相关系数值
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        ffppt[1, 1].text(
            j,
            i,
            f"{corr_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

# 添加颜色条并调整布局
c_huitu.colorbar(imamagiiinat, ax=ffppt[1, 1])
ffppt[1, 1].set_title("相关系数矩阵")
c_huitu.tight_layout()
c_huitu.savefig("y_chromosome_correlation_matrix.png", dpi=300, bbox_inches="tight")


# 创建BMI分组
bmi_bins = [0, 18.5, 24, 28, 50]  # 分组阈值
# 使用cut函数进行分组
c_dxygds_df_cleauc_tongjitubiaogongjuatened["BMI分组"] = c_tongjitubiaogongju.cut(
    c_dxygds_df_cleauc_tongjitubiaogongjuatened["孕妇BMI"],
    bins=bmi_bins,
    labels=["偏瘦", "正常", "超重", "肥胖"],
)

# 按分组和孕周计算Y染色体浓度均值
minbia_aaluze_wssk = (
    c_dxygds_df_cleauc_tongjitubiaogongjuatened.groupby(["BMI分组", "孕周"])[
        "Y染色体浓度"
    ]
    .mean()
    .reset_index()
)

# 创建画布
c_huitu.figure(figsize=(12, 8))

# 定义分组和颜色
minbia_progues = minbia_aaluze_wssk["BMI分组"].unique()
colors = ["blue", "green", "orange", "red"]

# 绘制各分组曲线
for i, group in enumerate(minbia_progues):
    group_shujubiaoao = minbia_aaluze_wssk[minbia_aaluze_wssk["BMI分组"] == group]
    if len(group_shujubiaoao) > 1:  # 确保有足够数据点
        c_huitu.plot(
            group_shujubiaoao["孕周"],
            group_shujubiaoao["Y染色体浓度"],
            label=f"{group}",
            color=colors[i],
            linewidth=2.5,
            marker="o",
            markersize=4,
        )

# 设置图表属性
c_huitu.xlabel("孕周", fontsize=14)
c_huitu.ylabel("Y染色体浓度均值", fontsize=14)
c_huitu.title("不同BMI分组的Y染色体浓度随孕周变化趋势", fontsize=16, fontweight="bold")
c_huitu.legend(title="BMI分组", fontsize=12)
c_huitu.grid(True, alpha=0.3)
c_huitu.savefig("y_chromosome_by_bmi_week.png", dpi=300, bbox_inches="tight")
print("BMI分组曲线图已保存为 'y_chromosome_by_bmi_week.png'")

# 定义Z值列
z_value_cols = [
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值",
    "X染色体的Z值",
    "Y染色体的Z值",
]

# 创建画布和子图
c_tutukuang, ffppt = c_huitu.subplots(2, 3, figsize=(15, 10))
c_tutukuang.suptitle("各染色体Z值分布直方图", fontsize=16, fontweight="bold")
ffppt = ffppt.flatten()  # 展平子图数组

# 绘制各Z值分布
for i, col in enumerate(z_value_cols):
    if i < len(ffppt):
        # 移除极端异常值
        shujubiaoao = c_dxygds_df_cleauc_tongjitubiaogongjuatened[col].dropna()
        q1 = shujubiaoao.quantile(0.01)
        q3 = shujubiaoao.quantile(0.99)
        filtered_shujubiaoao = shujubiaoao[(shujubiaoao >= q1) & (shujubiaoao <= q3)]

        # 绘制直方图
        ffppt[i].hist(
            filtered_shujubiaoao, bins=30, alpha=0.7, color=f"C{i}", edgecolor="black"
        )
        ffppt[i].set_xlabel("Z值")
        ffppt[i].set_ylabel("频数")
        ffppt[i].set_title(f"{col}分布")
        ffppt[i].grid(True, alpha=0.3)

        # 添加统计信息
        jun_zhi = filtered_shujubiaoao.mean()
        std_val = filtered_shujubiaoao.std()
        ffppt[i].axvline(
            jun_zhi,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"均值: {jun_zhi:.2f}",
        )
        ffppt[i].axvline(jun_zhi + std_val, color="orange", linestyle=":", linewidth=1)
        ffppt[i].axvline(jun_zhi - std_val, color="orange", linestyle=":", linewidth=1)
        ffppt[i].legend()

# 移除多余子图
if len(z_value_cols) < len(ffppt):
    for j in range(len(z_value_cols), len(ffppt)):
        c_tutukuang.delaxes(ffppt[j])

# 保存图表
c_huitu.tight_layout()
c_huitu.savefig("chromosome_zvalue_distribution.png", dpi=300, bbox_inches="tight")

# print("染色体Z值分布直方图已保存为 'chromosome_zvalue_distribution.png'")

# 定义质量控制列（注意列名中的空格）
qc_cols = [
    "GC含量",
    "原始读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "唯一比对的读段数  ",
    "被过滤掉读段数的比例",
]

# 创建画布和子图
c_tutukuang, ffppt = c_huitu.subplots(2, 3, figsize=(15, 10))
c_tutukuang.suptitle("质量控制指标分布图", fontsize=16, fontweight="bold")
ffppt = ffppt.flatten()  # 展平子图数组

# 绘制各QC指标
for i, col in enumerate(qc_cols):
    if i < len(ffppt):
        shujubiaoao = c_dxygds_df_cleauc_tongjitubiaogongjuatened[col].dropna()

        # 处理比例/百分比数据
        if "比例" in col or "含量" in col:
            shujubiaoao = shujubiaoao * 100  # 转换为百分比
            xlabel = f"{col.strip()} (%)"  # 去除列名空格
        else:
            xlabel = col.strip()  # 去除列名空格

        # 绘制直方图
        ffppt[i].hist(
            shujubiaoao, bins=30, alpha=0.7, color=f"C{i+2}", edgecolor="black"
        )
        ffppt[i].set_xlabel(xlabel)
        ffppt[i].set_ylabel("频数")
        ffppt[i].set_title(f"{col.strip()}分布")  # 去除列名空格
        ffppt[i].grid(True, alpha=0.3)

        # 添加统计信息
        jun_zhi = shujubiaoao.mean()
        zhong_wei_sshu = shujubiaoao.median()
        ffppt[i].axvline(
            jun_zhi,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"均值: {jun_zhi:.2f}",
        )
        ffppt[i].axvline(
            zhong_wei_sshu,
            color="green",
            linestyle="-.",
            linewidth=2,
            label=f"中位数: {zhong_wei_sshu:.2f}",
        )
        ffppt[i].legend()

# 保存图表
c_huitu.tight_layout()
c_huitu.savefig("quality_control_distribution.png", dpi=300, bbox_inches="tight")
