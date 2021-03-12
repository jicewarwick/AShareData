import numpy as np
import pandas as pd

from .FactorCompositor import FactorCompositor
from ..DBInterface import DBInterface


class NegativeBookEquityListingCompositor(FactorCompositor):
    def __init__(self, db_interface: DBInterface = None):
        """标识负净资产股票

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.table_name = '负净资产股票'

    def update(self):
        data = self.db_interface.read_table('合并资产负债表', '股东权益合计(不含少数股东权益)')
        storage = []
        for _, group in data.groupby('ID'):
            if any(group < 0):
                tmp = group.groupby('DateTime').tail(1) < 0
                t = tmp.iloc[tmp.argmax():].droplevel('报告期')
                t2 = t[np.concatenate(([True], t.values[:-1] != t.values[1:]))]
                if any(t2):
                    storage.append(t2)

        ret = pd.concat(storage)
        ret.name = '负净资产股票'
        self.db_interface.purge_table(self.table_name)
        self.db_interface.insert_df(ret, self.table_name)
