import pandas as pd
import numpy as np

from .Enums import SPLITE_FLAG, GROUP_METHOD


class SaleUtils:
    def __init__(self):
        self.BASE_TIME = pd.Timestamp("1970-01-01 0:0:0.0")
        self.figure_size = (8, 6)

    # 辅助函数--根据时间筛选指定内容
    def __filter_dataframe_by_date(
        self, df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
    ):
        return df[df["Date"] >= start_date][df["Date"] <= end_date]

    # 辅助函数--根据类别筛选指定内容
    def __filter_dataframe_by_class_code(self, df: pd.DataFrame, code: np.int64):
        return df[df["Cate"] == code]

    # 辅助函数--根据商品编码筛选指定内容
    def __filter_dataframe_by_item_code(self, df: pd.DataFrame, code: np.int64):
        return df[df["单品编码"] == code]

    # 根据分割条件（时间）对某天的销售量进行聚合
    def __get_sum_data_by_splite_flag_total(
        self, df: pd.DataFrame, d_col: str, gap: SPLITE_FLAG
    ):
        return df.groupby(pd.Grouper(key="Date", freq=gap))[d_col].sum()

    # 根据分割条件及聚合方法对销售量进行分组聚合
    def __get_sum_data_by_splite_flag_cated(
        self, df: pd.DataFrame, group_method: GROUP_METHOD, d_col: str, gap: SPLITE_FLAG
    ):
        sub_condition = (
            "Cate"
            if group_method == GROUP_METHOD.CLASS_CODE
            else (
                "CNme"
                if group_method == GROUP_METHOD.CLASS_NAME
                else (
                    "单品编码"
                    if group_method == GROUP_METHOD.ITEM_CODE
                    else ("name" if group_method == GROUP_METHOD.ITEM_NAME else "Name")
                )
            )
        )
        return (
            df.groupby([pd.Grouper(key="Date", freq=gap), sub_condition])[d_col]
            .sum()
            .unstack(sub_condition)
        )

    # 绘制折线图
    def draw_line_with_gap_range_xl_yl_title(
        self,
        df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        group_method: GROUP_METHOD = None,
        selected_code: np.int64 = None,
        d_col="Sale",
        gap: SPLITE_FLAG = SPLITE_FLAG.DAY,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
    ):
        df = (
            self.__get_sum_data_by_splite_flag_cated(
                self.__filter_dataframe_by_date(df, start_date, end_date),
                group_method,
                d_col,
                gap,
            )
            if group_method is not None
            else self.__get_sum_data_by_splite_flag_total(
                self.__filter_dataframe_by_date(df, start_date, end_date), d_col, gap
            )
        )
        if group_method is None:
            ax = df.plot(kind="line", figsize=self.figure_size)
        elif selected_code is None:
            ax = df.plot(kind="line", figsize=self.figure_size)
        else:
            ax = df[selected_code].plot(kind="line", figsize=self.figure_size)
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if title is not None:
            ax.set_title(title)

    # 绘制饼图
    def __draw_pi(
        self,
        df: pd.DataFrame,
        group_method: GROUP_METHOD = None,
        slice=None,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
    ):
        if group_method is None:
            ax = df.plot(kind="pie", subplots=True, figsize=self.figure_size)
        elif slice is None:
            ax = df.plot(kind="pie", subplots=True, figsize=self.figure_size)
        else:
            ax = df[slice].plot(kind="pie", figsize=self.figure_size)
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if title is not None:
            ax.set_title(title)
        pass

    def draw_pi(
        self,
        df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        set_by_time: bool = True,
        group_method: GROUP_METHOD = None,
        slice=None,
        d_col="Sale",
        gap: SPLITE_FLAG = SPLITE_FLAG.DAY,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
    ):
        df = (
            self.__get_sum_data_by_splite_flag_cated(
                self.__filter_dataframe_by_date(df, start_date, end_date),
                group_method,
                d_col,
                gap,
            )
            if group_method is not None
            else self.__get_sum_data_by_splite_flag_total(
                self.__filter_dataframe_by_date(df, start_date, end_date), d_col, gap
            )
        )
        if set_by_time:
            self.__draw_pi(df, group_method, slice, x_label, y_label, title)
        else:
            self.__draw_pi(df.T, group_method, slice, x_label, y_label, title)
