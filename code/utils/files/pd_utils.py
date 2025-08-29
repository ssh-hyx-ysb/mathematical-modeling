import pandas as pd
import numpy as np


class PandasExcelReader(object):
    def __init__(self,path=None):
        ef = pd.ExcelFile(path)
        self.frames = [ef.parse(sheet) for sheet in ef.sheet_names]
        for frame in self.frames:
            frame.dropna(axis=1,inplace=True,how='all')
            frame.dropna(axis=0,inplace=True,how='all')
            frame.fillna(1e-6,inplace=True)
            frame.fillna(1e-6,axis=1,inplace=True)
        
    def get_frame(self,num=0,sheet_name=None):
        if sheet_name==None:
            return self.frames[num]
        else:
            return self.frames[sheet_name]
        
    def ignore_columns(self,num,column_names:list=None):
        self.frames[num].drop(column_names,axis=1,inplace=True)
    def ignore_index(self,num,index=None):
        self.frames[num].drop(index,inplace=True)