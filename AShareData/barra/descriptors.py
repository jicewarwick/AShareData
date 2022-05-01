import datetime as dt
from functools import cached_property, lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..config import get_db_interface
from ..data_reader import StockDataReader
from ..database_interface import DBInterface
from ..factor import (BinaryFactor, ContinuousFactor, InterestRateFactor, LatestAccountingFactor, TTMAccountingFactor,
                      UnaryFactor)
from ..factor_compositor import FactorCompositor
from ..utils import Singleton


class BarraDataReader(StockDataReader):
    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)

    @cached_property
    def risk_free_rate(self) -> ContinuousFactor:
        """三月期shibor"""
        return InterestRateFactor('shibor利率数据', '3个月', self.db_interface).set_factor_name('3个月shibor')

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


class BarraCAPMRegression(Singleton):
    """CAPM regression with Barra's parameters

    .. math:: r_t - r_{ft} = \\alpha + \\beta R_t + \\epsilon_t

    where

    :math:`R_t` is the cap-weighted excess return of the estimation universe and

    :math:`r_t-r_{ft}` is the excess stock return

    The regression coefficients are estimated over the trailing 252 trading days of returns
    with a half-life of 63 trading days.

    """

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

        USE4 Methodology, page 9:

        Descriptors are standardized to have a mean of 0 and a standard deviation of 1.

        In other words, if :math:`d_{nl}^{Raw}` is the raw value of stock `n` for descriptor `l` , then

        the standardized descriptor value is given by

        .. math:: d_{nl} = \\frac{d_{nl}^{Raw}-\\mu_l}{\\sigma_l}

        where
        :math:`\\mu_l` is the cap-weighted mean of the descriptor (within the estimation universe), and
        :math:`\\sigma_l` is the equal-weighted standard deviation.

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

        USE4 Methodology, page 8:

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

    @staticmethod
    def weighted_std(val: pd.Series, weight: pd.Series) -> float:
        pass


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


#######################################
# Size
#######################################
class LNCAP(BarraDescriptorComputer):
    """Natural Log of Market Cap

    Given by the logarithm of the total market capitalization of the firm.
    """

    def __init__(self, db_interface: DBInterface):
        super().__init__(db_interface)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        return self.data_reader.log_cap.get_data(dates=date)


#######################################
# Beta
#######################################
class BETA(BarraDescriptorComputer):
    """Beta

    The :math:`\\beta` parameter from :py:class:`.BarraCAPMRegression`
    """

    def __init__(self, window: int = 252, half_life: int = 63, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.capm_computer = BarraCAPMRegression(window, half_life, db_interface=self.db_interface)

    def compute(self, date: dt.datetime):
        return self.compute_raw(date)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        beta, _ = self.capm_computer.compute(date)
        return beta


#######################################
# Momentum
#######################################
class RSTR(BarraDescriptorComputer):
    """Relative Strength

    Computed as the sum of excess log returns over the trailing :math:`T = 504` trading days
    with a lag of :math:`L=21` trading days,

    .. math:: RSTR=\\sum^{T+L}_{t=L}w_t[\\ln(1+r_t)-\\ln(1+r_{ft})]

    where :math:`r_t` is the stock return on day t, :math:`r_{ft}` is the risk-free return,
    and :math:`w_t` is an exponential weight with a half-life of 126 trading days.
    """

    def __init__(self, window: int = 504, lag: int = 21, half_life: int = 126, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window, self.lag, self.half_life = window, lag, half_life
        self.weight = exponential_weight(self.window, self.half_life)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        start_date = self.data_reader.calendar.offset(date, -self.lag - self.window + 1)
        end_date = self.data_reader.calendar.offset(date, -self.lag)
        weight = pd.Series(self.weight, index=self.data_reader.calendar.select_dates(start_date, end_date),
                           name='weight')

        tickers = self.data_reader.stocks.ticker(date)
        log_excess_stock_ret = self.data_reader.excess_log_return.get_data(start_date=start_date, end_date=end_date,
                                                                           ids=tickers)
        index = []
        values = []
        for ticker, group in log_excess_stock_ret.groupby('ID'):
            data = pd.concat([group.unstack(), weight], axis=1).dropna()
            val = data.iloc[:, 0].dot(data['weight']) / data['weight'].sum()
            index.append((date, ticker))
            values.append(val)
        return pd.Series(values, index=pd.MultiIndex.from_tuples(index), name='RSTR')


#######################################
# Residual Volatility
#######################################
class DASTD(BarraDescriptorComputer):
    """Daily Standard Deviation

    Computed as the volatility of daily excess returns over the past 252 trading days
    with a half-life of 42 trading days.
    """

    def __init__(self, window: int = 252, half_life: int = 42, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window, self.half_life = window, half_life
        self.weight = exponential_weight(self.window, self.half_life)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        start_date = self.data_reader.calendar.offset(date, -self.window + 1)
        tickers = self.data_reader.stocks.ticker(date)
        excess_return = self.data_reader.excess_return.get_data(start_date=start_date, end_date=date, ids=tickers)
        weight = pd.Series(self.weight, index=self.data_reader.calendar.select_dates(start_date, date), name='weight')

        index = []
        values = []
        for ticker, group in excess_return.groupby('ID'):
            data = pd.concat([group.unstack(), weight], axis=1).dropna()
            val = np.sqrt(np.cov(data.iloc[:, 0], aweights=data['weight']))
            index.append((date, ticker))
            values.append(val)
        return pd.Series(values, index=pd.MultiIndex.from_tuples(index), name='DASTD')


class CMRA(BarraDescriptorComputer):
    """Cumulative Range

    This descriptor differentiates stocks that have experienced wide swings over the last 12 months
    from those that have traded within a narrow range.

    Let :math:`Z(T)` be the cumulative excess log return over the past T months,
    with each month defined as the previous 21 trading days.

    .. math:: Z(T) = \\sum^T_\\tau[\\ln(1+r_t)-\\ln(1+r_{f\\tau})]

    where :math:`r_\\tau` is the stock return for month :math:`\\tau` (compounded over 21 days),
    and :math:`r_{f\\tau}` is the risk-free return. The cumulative range is given by

    .. math:: CMRA = \\ln(1+S_{max}) - \\ln(1+Z_{min})

    where :math:`Z_{max} = \\max[Z(T)],  Z_{min} = \\min[Z(T)]`, for :math:`T =1, \\dots, 12`.
    """

    def __init__(self, months: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.days_in_a_month = 21
        self.months = months

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        self.data_reader.calendar.fixed_duration_date_sequence(date, -self.days_in_a_month, self.months + 1)
        self.data_reader.stocks.alive_tickers()
        self.data_reader.hfq_close.log().diff().get_data()


class HSIGMA(BarraDescriptorComputer):
    """Historical Sigma

    Computed as the volatility of residual returns :math:`\\epsilon` in :py:class:`.BarraCAPMRegression`:

    .. math:: \\sigma = std(\\epsilon_t)

    """

    def __init__(self, window: int = 252, half_life: int = 63, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.capm_computer = BarraCAPMRegression(window, half_life, db_interface=self.db_interface)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        _, std_epsilon = self.capm_computer.compute(date)
        return std_epsilon


#######################################
# Non-linear Size
#######################################
class NLSIZE(BarraDescriptorComputer):
    """Cube of Size

    First, the standardized Size exposure (i.e., log of market cap) is cubed.
    The resulting factor is then orthogonalized with respect to the Size factor on a regression-weighted basis.
    Finally, the factor is winsorized and standardized.

    """

    def __init__(self, db_interface: DBInterface):
        super().__init__(db_interface)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        return self.data_reader.log_cap.get_data(dates=date)


#######################################
# Book-to-Price
#######################################
class BTOP(BarraDescriptorComputer):
    """Book to Price ratio

    Last reported book value of common equity divided by current market capitalization.
    """

    def __init__(self, db_interface: DBInterface):
        super().__init__(db_interface)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        return self.data_reader.bp.get_data(date)


#######################################
# Liquidity
#######################################
class STOM(BarraDescriptorComputer, Singleton):
    """Share Turnover, one month

    Computed as the log of the sum of daily turnover during the previous 21 trading days,

    .. math:: STOM = \\ln(\\sum^{21}_{t=1}\\frac{V_t}{S_t})

    where :math:`V_t` is the trading volume on day :math:`t`, and :math:`S_t` is the number of shares outstanding.
    """

    def __init__(self, window: int = 21, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window

    @lru_cache(12)
    def compute_raw(self, date: dt.datetime) -> pd.Series:
        start_date = self.data_reader.calendar.offset(date, -self.window)
        tickers = self.data_reader.stocks.ticker(date)
        data = self.data_reader.turnover_rate.get_data(start_date=start_date, end_date=date, ids=tickers)
        ret = np.log(data.groupby('ID').sum())
        ret.index = pd.MultiIndex.from_product([[date], ret.index.tolist()], names=('DateTime', 'ID'))
        return ret


# TODO: refactor STOQ and STOA
class STOQ(BarraDescriptorComputer):
    """Average Share Turnover, trailing 3 months

    Let :math:`STOM_{\\tau}` be the share turnover for month :math:`\\tau` ,
    with each month consisting of 21 trading days. The quarterly share turnover is defined by

    .. math:: STOQ = ln[\\frac{1}{T}\\sum^T_{\\tau = 1}\\exp(STOM_\\tau)]

    where :math:`T=3` months.
    """

    def __init__(self, window: int = 21, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window
        self.stom_helper = STOM(window)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        dates = self.data_reader.calendar.fixed_duration_date_sequence(date, -self.window, 3)
        monthly_stom = [self.stom_helper.compute_raw(date).droplevel('DateTime') for date in dates]
        data = np.log(np.exp(pd.concat(monthly_stom, axis=1).dropna()).sum(axis=1))
        data.index = pd.MultiIndex.from_product([[date], data.index.tolist()], names=['DateTime', 'ID'])
        return data


class STOA(BarraDescriptorComputer):
    """Average Share Turnover, trailing 12 months

    Let :math:`STOM_{\\tau}` be the share turnover for month :math:`\\tau` ,
    with each month consisting of 21 trading days.

    The quarterly share turnover is defined by

    .. math:: STOQ = ln[\\frac{1}{T}\\sum^T_{\\tau = 1}\\exp(STOM_\\tau)]

    where :math:`T=12` months.
    """

    def __init__(self, window: int = 21, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window
        self.stom_helper = STOM(window)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        dates = self.data_reader.calendar.fixed_duration_date_sequence(date, -self.window, 12)
        monthly_stom = [self.stom_helper.compute_raw(date).droplevel('DateTime') for date in dates]
        data = np.log(np.exp(pd.concat(monthly_stom, axis=1).dropna()).sum(axis=1))
        data.index = pd.MultiIndex.from_product([[date], data.index.tolist()], names=['DateTime', 'ID'])
        return data


#######################################
# Earnings Yield
#######################################
class EPFWD(BarraDescriptorComputer):
    """Predicted Earning to Price Ratio(NOT Implemented due to lack of data)

    Given by the 12-month forward-looking earnings divided by the current market capitalization.

    Forward-looking earnings are defined as
    a weighted average between the average analyst-predicted earnings for the current and next fiscal years.
    """


class CETOP(BarraDescriptorComputer):
    """Cash Earning to Price Ratio

    Given by the trailing 12-month cash earnings divided by current price.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        ticker = self.data_reader.stocks.ticker(date)
        return self.data_reader.cep_ttm.get_data(dates=date, ids=ticker)


class ETOP(BarraDescriptorComputer):
    """Trailing Earning to Price Ratio

    Given by the trailing 12-month earnings divided by the current market capitalization.

    Trailing earnings are defined as the last reported fiscal-year earnings
    plus the difference between current interim figure and the comparative interim figure from the previous year.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        ticker = self.data_reader.stocks.ticker(date)
        return self.data_reader.earning_ttm.get_data(dates=date, ids=ticker)


#######################################
# Growth
#######################################
class EGRLF(BarraDescriptorComputer):
    """Long-term predicted earnings growth(NOT Implemented due to lack of data)

    Long-term (3-5 years) earnings growth forecasted by analysts.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        pass


class EGRSF(BarraDescriptorComputer):
    """Short-term predicted earnings growth(NOT Implemented due to lack of data)

    Short-term (1 year) earnings growth forecasted by analysts.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        pass


class SGRO(BarraDescriptorComputer):
    """Earnings growth (trailing five years)

    Annual reported earnings per share are regressed against time over the past five fiscal years.
    The slope coefficient is then divided by the average annual earnings per share to obtain the earnings growth.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        return self.data_reader.log_cap.get_data(dates=date)


class EGRO(BarraDescriptorComputer):
    """Sales growth (trailing five years)

    Annual reported sales per share are regressed against time over the past five fiscal years.
    The slope coefficient is then divided by the average annual sales per share to obtain the sales growth.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        return self.data_reader.log_cap.get_data(dates=date)


#######################################
# Leverage
#######################################
class MLEV(BarraDescriptorComputer):
    """Market leverage

    Computed as

    .. math:: MLEV = \\frac{ME + PE + LD}{ME}

    where :math:`ME` is the market value of common equity on the last trading day,

    :math:`PE` is the most recent book value of preferred equity,

    and :math:`LD` is the most recent book value of long-term debt.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window
        self.mlev = (self.data_reader.total_market_cap + self.data_reader.preferred_equity +
                     self.data_reader.long_term_debt) / self.data_reader.total_market_cap

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        tickers = self.data_reader.stocks.ticker(date)
        return self.mlev.get_data(dates=date, ids=tickers)


class DTOA(BarraDescriptorComputer):
    """Debt-to-assets

    Computed as

    .. math:: DTOA = \\frac{TD}{TA}

    where :math:`TD` is the book value of total debt (long-term debt and current liabilities), and
    :math:`TA` is most recent book value of total assets.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        tickers = self.data_reader.stocks.ticker(date)
        return self.data_reader.debt_to_asset.get_data(dates=date, ids=tickers)


class BLEV(BarraDescriptorComputer):
    """Book leverage

    Computed as

    .. math:: BLEV = \\frac{BE + PE + LD}{BE}

    where :math:`BE` is the most recent book value of common equity,
    :math:`PE` is the most recent book value of preferred equity, and
    :math:`LD` is the most recent book value of long-term debt.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window
        self.blev = (self.data_reader.book_val + self.data_reader.preferred_equity +
                     self.data_reader.long_term_debt) / self.data_reader.book_val

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        tickers = self.data_reader.stocks.ticker(date)
        return self.blev.get_data(dates=date, ids=tickers)


class BarraDescriptorCompositor(FactorCompositor):
    def update(self):
        pass

    def capm_regression(self, date: dt.datetime):
        pass


def exponential_weight(n: int, half_life: int):
    series = range(-(n - 1), 1)
    return np.exp(np.log(2) * series / half_life)
