import datetime as dt
from functools import cached_property, lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from ..config import get_db_interface
from ..data_reader import DataReader, StockDataReader
from ..database_interface import DBInterface
from ..factor import AccountingFactor, BinaryFactor, ContinuousFactor, CumulativeInterestRateFactor, InterestRateFactor, \
    LatestAccountingFactor, TTMAccountingFactor, UnaryFactor
from ..tickers import StockTickerSelector
from ..utils import MARKET_STOCK_SELECTION, Singleton


class GrowthRateFactor(AccountingFactor):
    """Growth Rate"""

    def __init__(self, factor_name: str, db_interface: DBInterface = None):
        super().__init__(factor_name, db_interface)
        self.offset_strs = ['y1', 'y2', 'y3', 'y4', 'y5']
        self.buffer_length = 365 * 7
        self.name = f'{self._factor_name}增长率'

    @staticmethod
    def func(data: pd.DataFrame) -> pd.Series:
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
        y = data.loc[:, ['y5', 'y4', 'y3', 'y2', 'y1']].values[0]
        val = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ y / y.mean()
        ret = pd.Series(val[1], index=data.index)
        return ret


class BarraDataSourceReader(StockDataReader):
    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)

    @cached_property
    def risk_free_rate(self) -> ContinuousFactor:
        """三月期shibor"""
        return InterestRateFactor('shibor利率数据', '3个月', self.db_interface).set_factor_name('3个月shibor')

    @cached_property
    def cumulative_risk_free_rate(self) -> CumulativeInterestRateFactor:
        """累计三月期shibor收益率"""
        return CumulativeInterestRateFactor('shibor利率数据', '3个月', self.db_interface).set_factor_name('累计3个月shibor收益率')

    @cached_property
    def log_risk_free_return(self) -> UnaryFactor:
        return (self.risk_free_rate + 1).log()

    @cached_property
    def excess_return(self) -> BinaryFactor:
        return (self.returns - self.risk_free_rate).set_factor_name('excess_stock_return')

    @cached_property
    def excess_log_return(self) -> BinaryFactor:
        return (self.log_return - self.log_risk_free_return).set_factor_name('excess_lg_return')

    @cached_property
    def excess_market_return(self) -> BinaryFactor:
        return (self.market_return - self.risk_free_rate).set_factor_name('excess_market_return')

    @cached_property
    def operation_cash_flow_ttm(self) -> TTMAccountingFactor:
        return TTMAccountingFactor('经营活动产生的现金流量净额', self.db_interface)

    @cached_property
    def cep_ttm(self) -> BinaryFactor:
        return (self.operation_cash_flow_ttm / self.total_market_cap).set_factor_name('CEP')

    @cached_property
    def long_term_debt(self) -> LatestAccountingFactor:
        return LatestAccountingFactor('非流动负债合计')

    @cached_property
    def preferred_equity(self) -> LatestAccountingFactor:
        return LatestAccountingFactor('优先股')

    @cached_property
    def earning_growth(self) -> GrowthRateFactor:
        return GrowthRateFactor('净利润(不含少数股东损益)').set_factor_name('净利润增长率')

    @cached_property
    def sales_growth(self) -> GrowthRateFactor:
        return GrowthRateFactor('营业收入')


class TotalCap(Singleton):
    def __init__(self, db_interface: DBInterface = None):
        if db_interface is None:
            db_interface = get_db_interface()
        self.db_interface = db_interface
        self.stock_data_reader = StockDataReader(self.db_interface)

    @lru_cache(2)
    def cap(self, date: dt.datetime) -> pd.Series:
        weight = self.stock_data_reader.total_market_cap.get_data(dates=date)
        return weight


class BarraComputer(object):
    def __init__(self, db_interface: DBInterface = None):
        """ Base Class for Barra Computing.

        :param db_interface: DBInterface
        """
        if db_interface is None:
            db_interface = get_db_interface()
        self.db_interface = db_interface
        self.name = self.__class__.__name__
        self.data_reader = BarraDataSourceReader(self.db_interface)
        self.universe = StockTickerSelector(MARKET_STOCK_SELECTION, self.db_interface)
        self.cap = TotalCap(self.db_interface)

    def compute(self, date: dt.datetime):
        """ Do transformations to the data from raw computation."""
        raise NotImplementedError()

    def standardize(self, val: pd.Series) -> pd.Series:
        """ Standardize factors

        USE4 Methodology, Pg.9:

        Descriptors are standardized to have a mean of 0 and a standard deviation of 1.

        In other words, if :math:`d_{nl}^{Raw}` is the raw value of stock `n` for descriptor `l` , then
        the standardized descriptor value is given by

        .. math:: d_{nl} = \\frac{d_{nl}^{Raw}-\\mu_l}{\\sigma_l}

        where
        :math:`\mu_l` is the cap-weighted mean of the descriptor (within the estimation universe), and
        :math:`\sigma_l` is the equal-weighted standard deviation.

        We adopt the convention of standardizing using the cap-weighted mean so that
        a well-diversified cap-weighted portfolio has approximately zero exposure to all style factors.

        For the standard deviation, however, we use equal weights to prevent large-cap stocks from having
        an undue influence on the overall scale of the exposures.

        :param val: raw factor value
        :return: standardized factor
        """
        date = val.index.get_level_values('DateTime')[0]
        weight = self.cap.cap(date)
        tmp = pd.concat([val, weight], axis=1).dropna()
        ret = self._standardize(tmp.iloc[:, 0], tmp.iloc[:, 1])
        return ret

    @staticmethod
    def _standardize(val: pd.Series, weight: pd.Series) -> pd.Series:
        weight = weight / weight.sum()
        mu = val.dot(weight)
        sigma = val.std()
        return (val - mu) / sigma

    @staticmethod
    def trim_extreme(val: pd.Series, limit: float = 3.0, inplace=False) -> Optional[pd.Series]:
        """ Trim extreme values to `limit` standard deviation away from mean

        USE4 Methodology, Pg.8:

        The second group represents values that are regarded as legitimate,
        but nonetheless so large that their impact on the model must be limited.

        We trim these observations to three standard deviations from the mean.

        :param val: raw value
        :param limit: ranges in multiples of std
        :param inplace: Whether to perform the operation in place on the data.
        :return: trimmed data
        """
        mu = val.mean()
        sigma = val.std()
        lower_bound = mu - limit * sigma
        upper_bound = mu + limit * sigma
        return val.clip(lower=lower_bound, upper=upper_bound, inplace=inplace)

    @staticmethod
    def orthogonalize(val: pd.Series, base: pd.Series) -> pd.Series:
        res = val - (val * base).sum() / (base ** 2).sum() * base
        return res


class BarraStyleFactorReader(DataReader):
    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)

    @cached_property
    def size(self):
        return ContinuousFactor('BarraStyleFactor', 'Size', self.db_interface)


def exponential_weight(n: int, half_life: int):
    series = range(-(n - 1), 1)
    return np.exp(np.log(2) * series / half_life)
