import datetime as dt
from functools import lru_cache
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .common import BarraComputer, BarraDataSourceReader, exponential_weight
from ..config import get_db_interface
from ..database_interface import DBInterface
from ..tickers import StockTickerSelector
from ..utils import MARKET_STOCK_SELECTION, Singleton


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
        """

        :param window: look-back period
        :param half_life: half life of weight
        :param db_interface: DBInterface
        """
        if db_interface is None:
            db_interface = get_db_interface()
        self.db_interface = db_interface
        self.universe = StockTickerSelector(MARKET_STOCK_SELECTION, self.db_interface)
        self.data_reader = BarraDataSourceReader(self.db_interface)
        self.window = window
        self.half_life = half_life

    @lru_cache(2)
    def compute(self, date: dt.datetime) -> Tuple[pd.Series, pd.Series]:
        start_date = self.data_reader.calendar.offset(date, -(self.window - 1))
        tickers = self.universe.ticker(date)
        excess_market_return_df = self.data_reader.excess_market_return.get_data(start_date=start_date, end_date=date,
                                                                                 ids=tickers).unstack()
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


class BarraDescriptorComputer(BarraComputer):
    TABLE_NAME = 'BarraDescriptor'

    def __init__(self, db_interface: DBInterface = None):
        """ Base Class for Barra Descriptor Computing

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)

    def compute(self, date: dt.datetime):
        """ Do transformations to the data from raw computation.

        Trim observations to three standard deviations from the mean, then
        Descriptors are standardized to have a mean of 0 and a standard deviation of 1.
        """
        data = self.compute_raw(date)
        data = self.standardize(self.trim_extreme(data))
        data.name = self.name
        self.db_interface.update_df(data, self.TABLE_NAME)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        """ Compute raw data given by the formula in the methodology book"""
        raise NotImplementedError()


class SimpleBarraDescriptorComputer(BarraDescriptorComputer):
    def __init__(self, db_interface: DBInterface = None):
        """ Base Class for Simple Barra Descriptor Computing, i.e. from single `factor`

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.simple_factor = None

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        """ Get raw raw data that can get from single `factor`"""
        tickers = self.universe.ticker(date)
        return self.simple_factor.get_data(dates=date, ids=tickers)


#######################################
# Size
#######################################
class LNCAP(SimpleBarraDescriptorComputer):
    """Natural Log of Market Cap

    Given by the logarithm of the total market capitalization of the firm.
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.simple_factor = self.data_reader.log_cap


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

        tickers = self.universe.ticker(date)
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
        tickers = self.universe.ticker(date)
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
        dates = self.data_reader.calendar.fixed_duration_date_sequence(date, -self.days_in_a_month, self.months + 1)
        rf_storage = []
        for i in range(0, self.months):
            start_date = self.data_reader.calendar.offset(dates[i], 1)
            rf_info = self.data_reader.cumulative_risk_free_rate.get_data(start_date=start_date, end_date=dates[i + 1])
            rf_storage.append(np.log(1 + rf_info))
        log_rf_info = pd.concat(rf_storage)

        tickers = self.data_reader.stocks.ticker(dates[0])
        tickers = sorted(list(set(tickers) & self.universe.ticker(date)))
        ret_info = self.data_reader.hfq_close.log().diff().get_data(dates=dates, ids=tickers)

        storage = []
        for ticker, ret in ret_info.groupby('ID'):
            tmp = pd.concat([ret.droplevel('ID'), log_rf_info], axis=1)
            diff = tmp.iloc[:, 0] - tmp.iloc[:, 1]
            z_t = diff.sort_index(ascending=False).cumsum()
            cmra = np.log(1 + z_t.max()) - np.log(1 + z_t.min())
            storage.append(pd.Series(cmra, index=pd.MultiIndex.from_tuples([(date, ticker)], names=('DateTime', 'ID'))))
        res = pd.concat(storage)
        return res


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
class NLSIZE(SimpleBarraDescriptorComputer):
    """Cube of Size

    Cube the standardized Size exposure (i.e., log of market cap)
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.simple_factor = self.data_reader.log_cap

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        raw = super().compute_raw(date)
        return np.power(raw, 3)


#######################################
# Book-to-Price
#######################################
class BTOP(SimpleBarraDescriptorComputer):
    """Book to Price ratio

    Last reported book value of common equity divided by current market capitalization.
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.simple_factor = self.data_reader.bp


#######################################
# Liquidity
#######################################
class STOComputer(BarraDescriptorComputer):
    """Share Turnover Computer

    Computed as the log of the sum of daily turnover during the previous T months:

    .. math:: STO = \\ln(\\frac{1}{T}\\sum^{21T}_{t=1}\\frac{V_t}{S_t})

    where :math:`V_t` is the trading volume on day :math:`t`, and :math:`S_t` is the number of shares outstanding.
    """

    def __init__(self, months: int, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.months = months

    @lru_cache(12)
    def compute_raw(self, date: dt.datetime) -> pd.Series:
        start_date = self.data_reader.calendar.offset(date, -self.months * 21)
        tickers = self.universe.ticker(date)
        data = self.data_reader.turnover_rate.get_data(start_date=start_date, end_date=date, ids=tickers)
        ret = np.log(data.groupby('ID').sum())
        ret.index = pd.MultiIndex.from_product([[date], ret.index.tolist()], names=('DateTime', 'ID'))
        return ret


class STOM(STOComputer):
    """Share Turnover, one month

    Computed as the log of the sum of daily turnover during the previous 21 trading days,

    .. math:: STOM = \\ln(\\sum^{21}_{t=1}\\frac{V_t}{S_t})

    where :math:`V_t` is the trading volume on day :math:`t`, and :math:`S_t` is the number of shares outstanding.
    """

    def __init__(self, months: int = 1, db_interface: DBInterface = None):
        super().__init__(months, db_interface)


class STOQ(STOComputer):
    """Average Share Turnover, trailing 3 months

    Let :math:`STOM_{\\tau}` be the share turnover for month :math:`\\tau` ,
    with each month consisting of 21 trading days. The quarterly share turnover is defined by

    .. math:: STOQ = ln[\\frac{1}{T}\\sum^T_{\\tau = 1}\\exp(STOM_\\tau)]

    where :math:`T=3` months.
    """

    def __init__(self, months: int = 3, db_interface: DBInterface = None):
        super().__init__(months, db_interface)


class STOA(STOComputer):
    """Average Share Turnover, trailing 12 months

    Let :math:`STOM_{\\tau}` be the share turnover for month :math:`\\tau` ,
    with each month consisting of 21 trading days.

    The quarterly share turnover is defined by

    .. math:: STOA = ln[\\frac{1}{T}\\sum^T_{\\tau = 1}\\exp(STOM_\\tau)]

    where :math:`T=12` months.
    """

    def __init__(self, months: int = 12, db_interface: DBInterface = None):
        super().__init__(months, db_interface)


#######################################
# Earnings Yield
#######################################
class EPFWD(BarraDescriptorComputer):
    """Predicted Earning to Price Ratio(NOT Implemented due to lack of data)

    Given by the 12-month forward-looking earnings divided by the current market capitalization.

    Forward-looking earnings are defined as
    a weighted average between the average analyst-predicted earnings for the current and next fiscal years.
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        raise NotImplementedError()


class CETOP(SimpleBarraDescriptorComputer):
    """Cash Earning to Price Ratio

    Given by the trailing 12-month cash earnings divided by current price.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        ticker = self.universe.ticker(date)
        return self.data_reader.cep_ttm.get_data(dates=date, ids=ticker)


class ETOP(SimpleBarraDescriptorComputer):
    """Trailing Earning to Price Ratio

    Given by the trailing 12-month earnings divided by the current market capitalization.

    Trailing earnings are defined as the last reported fiscal-year earnings
    plus the difference between current interim figure and the comparative interim figure from the previous year.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        ticker = self.universe.ticker(date)
        return self.data_reader.earning_ttm.get_data(dates=date, ids=ticker)


#######################################
# Growth
#######################################
class EGRLF(BarraDescriptorComputer):
    """Long-term predicted earnings growth(NOT Implemented due to lack of data)

    Long-term (3-5 years) earnings growth forecasted by analysts.
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        raise NotImplementedError()


class EGRSF(BarraDescriptorComputer):
    """Short-term predicted earnings growth(NOT Implemented due to lack of data)

    Short-term (1 year) earnings growth forecasted by analysts.
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)

    def compute_raw(self, date: dt.datetime) -> pd.Series:
        raise NotImplementedError()


class EGRO(SimpleBarraDescriptorComputer):
    """Earnings growth (trailing five years)

    Annual reported earnings per share are regressed against time over the past five fiscal years.
    The slope coefficient is then divided by the average annual earnings per share to obtain the earnings growth.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window
        self.simple_factor = self.data_reader.earning_growth


class SGRO(SimpleBarraDescriptorComputer):
    """Sales growth (trailing five years)

    Annual reported sales per share are regressed against time over the past five fiscal years.
    The slope coefficient is then divided by the average annual sales per share to obtain the sales growth.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window
        self.simple_factor = self.data_reader.sales_growth


#######################################
# Leverage
#######################################
class MLEV(SimpleBarraDescriptorComputer):
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
        mlev = (self.data_reader.total_market_cap + self.data_reader.preferred_equity +
                self.data_reader.long_term_debt) / self.data_reader.total_market_cap
        self.simple_factor = mlev


class DTOA(SimpleBarraDescriptorComputer):
    """Debt-to-assets

    Computed as

    .. math:: DTOA = \\frac{TD}{TA}

    where :math:`TD` is the book value of total debt (long-term debt and current liabilities), and
    :math:`TA` is most recent book value of total assets.
    """

    def __init__(self, window: int = 12, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.window = window
        self.simple_factor = self.data_reader.debt_to_asset


class BLEV(SimpleBarraDescriptorComputer):
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
        blev = (self.data_reader.book_val + self.data_reader.preferred_equity +
                self.data_reader.long_term_debt) / self.data_reader.book_val
        self.simple_factor = blev
