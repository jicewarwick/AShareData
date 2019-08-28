from functools import lru_cache

import numpy as np
import pandas as pd
# todo: import from functools from 3.9
from cached_property import cached_property

from SQLDBReader import SQLDBReader


class FactorZoo(object):
    def __init__(self, db_reader: SQLDBReader):
        self._db_reader = db_reader

    @cached_property
    def total_share(self) -> pd.DataFrame:
        return self._db_reader.get_factor('股票日行情', '总股本')

    @cached_property
    def close(self) -> pd.DataFrame:
        close = self._db_reader.get_factor('股票日行情', '收盘价')
        return close

    @cached_property
    def adj_factor(self) -> pd.DataFrame:
        adj_factor = self._db_reader.get_factor('股票日行情', '复权因子')
        return adj_factor

    @cached_property
    def hfq_close(self) -> pd.DataFrame:
        close = self._db_reader.get_factor('股票日行情', '收盘价')
        adj_factor = self.adj_factor
        hfq_close = (close * adj_factor).ffill()
        return hfq_close

    def qfq_close(self) -> pd.DataFrame:
        close = self._db_reader.get_factor('股票日行情', '收盘价')
        adj_factor = self.adj_factor
        max_adj_factor = adj_factor.max()
        adj_factor_ratio = adj_factor.div(max_adj_factor, axis=1)
        qfq_close = (close * adj_factor_ratio).ffill()
        return qfq_close

    def pct_chg(self) -> pd.DataFrame:
        return self.hfq_close.pct_change()

    @cached_property
    def names(self) -> pd.DataFrame:
        df = self._db_reader.get_factor('股票曾用名', '证券名称', ffill=True)
        return df

    def trading_status(self) -> pd.DataFrame:
        close_copy = self.close.copy()

        def fill_series(series):
            index = series.dropna().index[0]
            series.where(np.logical_not(series > 0), True, inplace=True)
            series.loc[index:].fillna(False, inplace=True)
            return series

        return close_copy.apply(fill_series)

    def is_st(self) -> pd.DataFrame:
        return self.names.str.startswith('ST')

    @lru_cache(None)
    def index_close(self, ts_code: str) -> pd.Series:
        return self._db_reader.get_factor('指数日行情', '收盘点位', stock_list=[ts_code])[ts_code]

    def index_return(self, ts_code: str) -> pd.Series:
        return self.index_close(ts_code).pct_change()

    def excess_market_return(self) -> pd.DataFrame:
        pass

    def excess_return(self) -> pd.DataFrame:
        pass
