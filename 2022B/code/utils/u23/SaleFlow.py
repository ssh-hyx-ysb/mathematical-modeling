import pandas as pd
import numpy as np

from .Enums import SPLITE_FLAG, GROUP_METHOD


class SaleFlow:
    def __init__(
        self,
        info_chart,
        sale_flow,
        cost_flow,
        loss_rate_chart_item,
        loss_rate_chart_avge,
    ):
        self.info_chart = info_chart
        self.loss_rate_chart_item = loss_rate_chart_item
        self.loss_rate_chart_avge = loss_rate_chart_avge

        # 定义映射表 商品编码 -> 商品信息
        self.reflect_dict_ic2in = {
            self.info_chart.loc[i]["单品编码"]: [
                self.info_chart.loc[i]["单品名称"],
                self.info_chart.loc[i]["分类编码"],
                self.info_chart.loc[i]["分类名称"],
            ]
            for i in range(0, len(self.info_chart))
        }

        # 定义映射表 商品名 -> 商品编码
        self.reflect_dict_in2ic = {}
        for i in range(len(self.info_chart)):
            try:
                self.reflect_dict_in2ic[
                    self.__del_brace(self.info_chart.loc[i]["单品名称"])
                ].append(self.info_chart.loc[i]["单品编码"])
            except KeyError:
                self.reflect_dict_in2ic[
                    self.__del_brace(self.info_chart.loc[i]["单品名称"])
                ] = [self.info_chart.loc[i]["单品编码"]]

        # 定义映射表 商品编码 -> 类别编码
        self.reflect_dict_ic2cc = {
            self.info_chart.loc[i]["单品编码"]: self.info_chart.loc[i]["分类编码"]
            for i in range(len(self.info_chart))
        }

        # 定义映射表 类别编码 -> 商品编码
        self.reflect_dict_cc2ic = {}
        for i in range(len(self.info_chart)):
            try:
                self.reflect_dict_cc2ic[self.info_chart.loc[i]["分类编码"]].append(
                    self.info_chart.loc[i]["单品编码"]
                )
            except KeyError:
                self.reflect_dict_cc2ic[self.info_chart.loc[i]["分类编码"]] = [
                    self.info_chart.loc[i]["单品编码"]
                ]

        # 定义映射表 类别编码 -> 类别名
        self.reflect_dict_cc2cn = {
            self.info_chart.loc[i]["分类编码"]: self.info_chart.loc[i]["分类名称"]
            for i in range(0, len(self.info_chart))
        }

        # 定义映射表 类别名 -> 类别编码
        self.reflect_dict_cn2cc = {
            self.info_chart.loc[i]["分类名称"]: self.info_chart.loc[i]["分类编码"]
            for i in range(0, len(self.info_chart))
        }

        # 定义映射表 商品编码 -> 折损率
        self.loss_dict_item = {
            self.loss_rate_chart_item.loc[i]["单品编码"]: {
                "name": self.loss_rate_chart_item.loc[i]["单品名称"],
                "loss_rate": self.loss_rate_chart_item.loc[i]["损耗率(%)"],
            }
            for i in range(0, len(self.loss_rate_chart_item))
        }

        # 定义映射表 商品品类 -> 平均折损率
        self.loss_dict_avge = {
            self.loss_rate_chart_avge.loc[i]["小分类编码"]: {
                "name": self.loss_rate_chart_avge.loc[i]["小分类名称"],
                "loss_rate": self.loss_rate_chart_avge.loc[i][
                    "平均损耗率(%)_小分类编码_不同值"
                ],
            }
            for i in range(0, len(self.loss_rate_chart_avge))
        }

        # 定义常用变量
        self.class_nums = len(self.reflect_dict_cc2cn)
        self.items_nums = len(self.reflect_dict_ic2in)

        self.sale_flow = sale_flow
        self.cost_flow = cost_flow
        self.reflect_dict_dc2c = self.cost_flow.groupby(["日期", "单品编码"])[
            "批发价格(元/千克)"
        ].mean()

        # 处理附件2
        # 添加Name列，商品名
        self.sale_flow["Name"] = self.sale_flow["单品编码"].apply(
            lambda code: self.reflect_dict_ic2in[code][0]
        )
        # 添加name列，商品名（去标签）
        self.sale_flow["name"] = self.sale_flow["单品编码"].apply(
            lambda code: self.__del_brace(self.reflect_dict_ic2in[code][0])
        )
        # 添加Date列，合并日期与时间
        self.sale_flow["Date"] = pd.to_datetime(
            self.sale_flow["销售日期"].dt.strftime("%Y-%m-%d")
            + " "
            + self.sale_flow["扫码销售时间"].str.split(".").str[0],
            format="%Y-%m-%d %H:%M:%S",
        )
        self.sale_flow["date"] = self.sale_flow["Date"].apply(
            lambda date: date.strftime("%Y-%m-%d")
        )
        # 添加Sale列，计算每件商品销售额
        self.sale_flow["Sale"] = pd.to_numeric(
            self.sale_flow["销量(千克)"] * self.sale_flow["销售单价(元/千克)"]
        )
        # 添加Cate列，添加每种单品品类
        self.sale_flow["Cate"] = self.sale_flow["单品编码"].apply(
            self.get_class_code_by_item_code
        )
        # 添加CNme列，添加每种单品品类
        self.sale_flow["CNme"] = self.sale_flow["单品编码"].apply(
            self.get_class_name_by_item_code
        )
        # 添加Loss列，添加每种单品的损耗率
        self.sale_flow["Loss"] = self.sale_flow["单品编码"].apply(
            lambda code: self.get_item_recent_loss_rate(code)["loss_rate"]
        )
        # 添加LOSS列，添加每种品类的平均损耗率
        self.sale_flow["LOSS"] = self.sale_flow["Cate"].apply(
            lambda code: self.get_aveg_recent_loss_rate(code)["loss_rate"]
        )
        # 添加Cost列，添加每种商品的进价
        self.sale_flow["Cost"] = self.sale_flow.apply(
            self.__get_cost_by_item_code_and_date, axis=1
        )

    # 辅助函数--根据商品编码和日期获取成本
    def __get_cost_by_item_code_and_date(self, row: pd.Series):
        try:
            cost = self.reflect_dict_dc2c[row["date"]][row["单品编码"]]
        except KeyError:
            cost = 0
        return cost

    # 辅助函数--删除字符串后面的括号内容
    def __del_brace(self, str: str):
        return str.split("(")[0]

    # 根据编码获取商品信息
    def get_item_info_by_item_code(self, code: np.int64):
        return self.reflect_dict_ic2in[code]

    # 根据商品名获取商品编码
    def get_item_code_by_item_name(self, name: str):
        return self.reflect_dict_in2ic[name]

    # 根据商品编码获取类别编码
    def get_class_code_by_item_code(self, code: np.int64):
        return self.reflect_dict_ic2cc[code]

    # 根据品类编码获取商品编码
    def get_item_code_by_class_code(self, code: np.int64):
        return self.reflect_dict_cc2ic[code]

    # 根据品类编码获取品类名
    def get_class_name_by_class_code(self, code: np.int64):
        return self.reflect_dict_cc2cn[code]

    # 根据品类名获取品类编码
    def get_class_code_by_class_name(self, name: str):
        return self.reflect_dict_cn2cc[name]

    # 根据商品代码获取品类名
    def get_class_name_by_item_code(self, code: np.int64):
        return self.reflect_dict_cc2cn[self.get_class_code_by_item_code(code)]

    # 获取商品近期损耗率
    def get_item_recent_loss_rate(self, code: np.int64):
        return self.loss_dict_item[code]

    # 获取品类近期平均损耗率
    def get_aveg_recent_loss_rate(self, code: np.int64):
        return self.loss_dict_avge[code]
