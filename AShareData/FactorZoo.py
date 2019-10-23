from functools import lru_cache, reduce

import numpy as np
import pandas as pd
# todo: import from functools from 3.9
from cached_property import cached_property

from .AShareDataReader import AShareDataReader
from .constants import STOCK_INDEXES, TRADING_DAYS_IN_YEAR


class FactorZoo(object):
    def __init__(self, db_reader: AShareDataReader):
        self._db_reader = db_reader

    @cached_property
    def total_share(self) -> pd.DataFrame:
        return self._db_reader.get_factor('股票日行情', '总股本')

    def floating_share(self) -> pd.DataFrame:
        return self._db_reader.get_factor('股票日行情', '总股本')

    @cached_property
    def market_cap(self) -> pd.DataFrame:
        return self.close * self.total_share

    def log_cap(self) -> pd.DataFrame:
        return self.market_cap.log()

    @cached_property
    def cap_weight(self) -> pd.DataFrame:
        return self.market_cap.div(self.market_cap.sum(axis=1, skipna=True), axis=0, skipna=True)

    @cached_property
    def cap_wighted_return(self) -> pd.Series:
        return (self.equity_daily_return * self.cap_weight).sum(axis=1, skipna=True)

    @cached_property
    def close(self) -> pd.DataFrame:
        close = self._db_reader.get_factor('股票日行情', '收盘价')
        return close

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
        adj_factor = self.adj_factor()
        max_adj_factor = adj_factor.max()
        adj_factor_ratio = adj_factor.div(max_adj_factor, axis=1)
        qfq_close = (close * adj_factor_ratio).ffill()
        return qfq_close

    def daily_return(self) -> pd.DataFrame:
        return self.hfq_close.pct_change()

    def log_return(self) -> pd.DataFrame:
        return self.hfq_close.log().diff()

    @cached_property
    def equity_daily_return(self) -> pd.DataFrame:
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

    @lru_cache(None)
    def shibor_rate(self, maturity: str = '1年') -> pd.Series:
        return self._db_reader.get_factor('Shibor利率数据', maturity).div(TRADING_DAYS_IN_YEAR)

    def log_shibor_return(self, maturity: str = '1年'):
        return self.shibor_rate(maturity).add(1).log()

    def excess_market_return(self, index_name: str = '沪深300') -> pd.DataFrame:
        assert index_name in STOCK_INDEXES, f'支持的指数为{list(STOCK_INDEXES)}'
        market_index_close = self.index_close(ts_code=STOCK_INDEXES[index_name])
        market_return = market_index_close.pct_change()
        return self.equity_daily_return.sub(market_return, axis=0)

    def excess_return(self) -> pd.DataFrame:
        return self.equity_daily_return.sub(self.shibor_rate(), axis=0)

    @staticmethod
    def exponential_weight(n: int, half_life: int) -> np.array:
        series = range(-(n-1), 1)
        return np.exp(np.log(2) * series / half_life)

    def estimation_universe_bool(self) -> pd.DataFrame:
        return reduce(np.bitwise_and, [self.listed_more_than_n_days(30), self.trading_status(), self.is_st()])

    def list_status(self) -> pd.DataFrame:
        return self._db_reader.get_factor('股票上市退市', '上市状态', ffill=True)

    def listed_more_than_n_days(self, n: int) -> pd.DataFrame:
        return self.list_status().shift(n)

    def industry(self, provider: str, level: int = 1):
        return self._db_reader.get_industry(provider=provider, level=level)
