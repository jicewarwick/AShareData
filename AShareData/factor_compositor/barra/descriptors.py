import datetime as dt
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from singleton_decorator import singleton

from ..factor_compositor import FactorCompositor
from ...config import get_db_interface
from ...data_reader import StockDataReader
from ...database_interface import DBInterface
from ...factor import BinaryFactor, ContinuousFactor, InterestRateFactor, UnaryFactor


class BarraDataReader(StockDataReader):
    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)

    @property
    def risk_free_rate(self) -> ContinuousFactor:
        """三月期shibor"""
        return InterestRateFactor('shibor利率数据', '3个月', self.db_interface).set_factor_name('3个月shibor')

    @property
    def log_risk_free_return(self) -> UnaryFactor:
        return (self.risk_free_rate + 1).log()

    @property
    def excess_return(self) -> BinaryFactor:
        return (self.returns - self.risk_free_rate).set_factor_name('excess_stock_return')

    @property
    def excess_log_return(self) -> BinaryFactor:
        return (self.log_return - self.log_risk_free_return).set_factor_name('excess_lg_return')

    @property
    def excess_market_return(self) -> BinaryFactor:
        return (self.market_return - self.risk_free_rate).set_factor_name('excess_market_return')


@singleton
class BarraCAPMRegression(object):
    def __init__(self, window: int = 252, half_life: int = 63, db_interface: DBInterface = None):
        if db_interface is None:
            db_interface = get_db_interface()
        self.db_interface = db_interface
        self.data_reader = BarraDataReader(self.db_interface)
        self.window = window
        self.half_life = half_life

    @lru_cache(2)
    def compute(self, date: dt.datetime) -> Tuple[pd.Series, pd.Series]:
        start_date = self.data_reader.calendar.offset(date, -(self.window - 1))
        excess_market_return_df = self.data_reader.excess_market_return.get_data(start_date=start_date,
                                                                                 end_date=date).unstack()
        market_ret_name = excess_market_return_df.columns[0]
        X = sm.add_constant(excess_market_return_df)
        X['weight'] = exponential_weight(self.window, self.half_life)

        index_tuple = []
        beta = []
        hsigma = []
        tickers = self.data_reader.stocks.alive_tickers([start_date, date])
        for ticker in tickers:
            excess_stock_return_df = self.data_reader.excess_return.get_data(start_date=start_date, end_date=date,
                                                                             ids=ticker).unstack()
            data = pd.concat([excess_stock_return_df, X], axis=1).dropna()
            model = sm.WLS(data.iloc[:, 0], data.loc[:, ['const', market_ret_name]], data['weight'])
            res = model.fit()
            index_tuple.append((ticker, date))
            beta.append(res.params[market_ret_name])
            hsigma.append(res.resid.std())
        index = pd.MultiIndex.from_tuples(index_tuple)
        return pd.Series(beta, index=index), pd.Series(hsigma, index=index)


class BarraComputer(object):
    @staticmethod
    def standardize(val: pd.Series, weight: pd.Series) -> pd.Series:
        """ Standardize factors

        USE4 Methodology, page 9
        Descriptors are standardized to have a mean of 0 and a standard deviation of 1.

        In other words, if :math:`d_{nl}^{Raw}` is the raw value of stock `n` for descriptor `l` , then

        the standardized descriptor value is given by

        .. math:: d_{nl} = \frac{d_{nl}^{Raw}-\mu_l}{\sigma_l}

        where
        :math:`\mu_l` is the cap-weighted mean of the descriptor (within the estimation universe), and
        :math:`\sigma_l` is the equal-weighted standard deviation.

        We adopt the convention of standardizing using the cap-weighted mean so that a well-diversified cap-weighted portfolio has approximately zero exposure to all style factors.

        For the standard deviation, however, we use equal weights to prevent large-cap stocks from having an undue influence on the overall scale of the exposures.

        :param val: raw factor value
        :param weight: market cap
        :return: standardized factor
        """
        mu = val.dot(weight)
        sigma = val.std()
        return (val - mu) / sigma

    @staticmethod
    def trim_extreme(val: pd.Series, limit: float = 3.0, inplace=False) -> Optional[pd.Series]:
        """ Trim extreme values to `limit` standard deviation away from mean

        USE4 Methodology, page 8
        The second group represents values that are regarded as legitimate, but nonetheless so large that their impact on the model must be limited.
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


class BarraDescriptorComputer(BarraComputer):
    def __init__(self, db_interface: DBInterface):
        self.db_interface = db_interface
        self.data_reader = BarraDataReader(self.db_interface)

    def compute(self, date: dt.datetime):
        data = self.compute_raw(date)
        weight = self.data_reader.total_market_cap.get_data(dates=date)
        return self.standardize(data, weight)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        raise NotImplementedError()


class LNCAP(BarraDescriptorComputer):
    """Natural Log of Market Cap"""

    def __init__(self, db_interface: DBInterface):
        super().__init__(db_interface)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        return self.data_reader.log_cap.get_data(dates=date)


class BETA(BarraDescriptorComputer):
    """Beta"""

    def __init__(self, db_interface: DBInterface):
        super().__init__(db_interface)
        self.capm_computer = BarraCAPMRegression(db_interface=self.db_interface)

    def compute(self, date: dt.datetime):
        return self.compute_raw(date)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        beta, _ = self.capm_computer.compute(date)
        return beta


class RSTR(BarraDescriptorComputer):
    """Relative Strength"""

    def __init__(self, days: int = 504, lag: int = 21, half_life: int = 126, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.days, self.lag, self.half_life = days, lag, half_life
        self.weight = exponential_weight(self.days, self.half_life)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        end_date = self.data_reader.calendar.offset(date, -self.lag)
        start_date = self.data_reader.calendar.offset(date, -self.lag - self.days + 1)
        data = self.data_reader.excess_log_return.get_data(start_date=start_date, end_date=end_date)
        for ticker, group in data.groupby('ID'):
            if group.shape[0] == self.days:
                pass


class BarraDescriptorCompositor(FactorCompositor):
    def update(self):
        pass

    def capm_regression(self, date: dt.datetime):
        pass


def exponential_weight(n: int, half_life: int):
    series = range(-(n - 1), 1)
    return np.exp(np.log(2) * series / half_life)
