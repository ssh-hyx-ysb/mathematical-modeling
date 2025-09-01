from sympy import symbols, cos, pi, solve, sqrt, simplify, Eq
import numpy as np

# 定义变量
x, y = symbols("x y")
x1, y1, x2, y2, x3, y3 = symbols("x1 y1 x2 y2 x3 y3")
alpha1, alpha2, alpha3 = symbols("alpha1 alpha2 alpha3")


# 定义距离函数（简化）
def dist_sq(xa, ya, xb, yb):
    """返回两点间距离的平方（避免 sqrt）"""
    return (xa - xb) ** 2 + (ya - yb) ** 2


def cos_rule_expr(dAB_sq, dBC_sq, dAC_sq):
    """余弦定理：cos(B) = (AB² + BC² - AC²) / (2 * AB * BC)"""
    # 注意：这里我们保留 sqrt，但可以尝试平方两边来消除
    AB = sqrt(dAB_sq)
    BC = sqrt(dBC_sq)
    return (dAB_sq + dBC_sq - dAC_sq) / (2 * AB * BC)


# 主函数
def optimized_eqs(
    x1_val=0,
    y1_val=0,
    x2_val=100,
    y2_val=0,
    x3_val=50,
    y3_val=86.6,  # 例如等边三角形
    alpha1_deg=30,
    alpha2_deg=30,
    alpha3_deg=60,
):
    """
    求解点 P(x,y) 使得：
        ∠APB = alpha1
        ∠BPC = alpha2
        ∠CPA = alpha3
    """

    # 转换为弧度
    alpha1_rad = alpha1_deg * pi / 180
    alpha2_rad = alpha2_deg * pi / 180
    alpha3_rad = alpha3_deg * pi / 180

    # 定义距离平方（避免 sqrt）
    dPA_sq = dist_sq(x, y, x1, y1)
    dPB_sq = dist_sq(x, y, x2, y2)
    dPC_sq = dist_sq(x, y, x3, y3)

    dAB_sq = dist_sq(x1, y1, x2, y2)
    dBC_sq = dist_sq(x2, y2, x3, y3)
    dCA_sq = dist_sq(x3, y3, x1, y1)

    # 使用余弦定理建立方程（避免分母嵌套 sqrt）
    # 但注意：仍有 sqrt，所以考虑平方两边（需小心增根）

    # 方程1: ∠APB = alpha1
    cos_APB = cos_rule_expr(dPA_sq, dPB_sq, dAB_sq)
    eq1 = Eq(cos_APB, cos(alpha1_rad))

    # 方程2: ∠BPC = alpha2
    cos_BPC = cos_rule_expr(dPB_sq, dPC_sq, dBC_sq)
    eq2 = Eq(cos_BPC, cos(alpha2_rad))

    # 方程3: ∠CPA = alpha3
    cos_CPA = cos_rule_expr(dPC_sq, dPA_sq, dCA_sq)
    eq3 = Eq(cos_CPA, cos(alpha3_rad))

    # 代入具体数值
    eq1_num = eq1.subs(
        {
            x1: x1_val,
            y1: y1_val,
            x2: x2_val,
            y2: y2_val,
            x3: x3_val,
            y3: y3_val,
            alpha1: alpha1_rad,
        }
    )

    eq2_num = eq2.subs(
        {
            x1: x1_val,
            y1: y1_val,
            x2: x2_val,
            y2: y2_val,
            x3: x3_val,
            y3: y3_val,
            alpha2: alpha2_rad,
        }
    )

    eq3_num = eq3.subs(
        {
            x1: x1_val,
            y1: y1_val,
            x2: x2_val,
            y2: y2_val,
            x3: x3_val,
            y3: y3_val,
            alpha3: alpha3_rad,
        }
    )

    # 尝试求解（仍可能失败，但表达式更清晰）
    try:
        result = solve([eq1_num, eq2_num, eq3_num], [x, y], dict=True)
        return result
    except NotImplementedError:
        print("符号求解失败，建议使用数值方法")
        return None
