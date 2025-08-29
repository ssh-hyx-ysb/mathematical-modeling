from enum import Enum


# 分隔标记
class SPLITE_FLAG:
    # 按1分钟分隔流水时段
    MININT_1 = "1min"
    # 按5分钟分隔流水时段
    MININT_5 = "5min"
    # 按10分钟分隔流水时段
    MININT_10 = "10min"
    # 按15分钟分隔流水时段
    MININT_15 = "15min"
    # 按30分钟分隔流水时段
    MININT_30 = "30min"
    # 按小时分隔流水时段
    HOUR = "h"
    # 按天分隔流水时段
    DAY = "D"
    # 按周分隔流水时段
    WEEK = "W"
    # 按月分隔流水时段
    MONTH = "ME"
    # 按季度分隔流水时段
    SEASON = "QE"
    # 按年分隔流水时段
    YEAR = "YE"


# 分组模式
class GROUP_METHOD(Enum):
    # 按类别代码
    CLASS_CODE = 0
    # 按类别名称
    CLASS_NAME = 1
    # 按商品代码
    ITEM_CODE = 2
    # 按商品名（全名）
    ITEM_NAME = 3
    # 按商品名（简名）
    ITEM_NAMES = 4
