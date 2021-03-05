import datetime as dt
import logging
from typing import Sequence, Tuple, Union

import alphalens
import numpy as np
import pandas as pd
from jqfactor_analyzer import FactorAnalyzer
from scipy.stats.mstats import winsorize
from statsmodels.stats import descriptivestats

from . import AShareDataReader, DateUtils
from .config import get_db_interface
from .DBInterface import DBInterface
from .Factor import BinaryFactor, FactorBase, IndustryFactor, UnaryFactor
from .Tickers import StockTickerSelector
from .utils import StockSelectionPolicy
from .algo import human_sort


class CrossSectionalPortfolioAnalysis(object):
    def __init__(self, forward_return: UnaryFactor, ticker_selector: StockTickerSelector, dates: Sequence[dt.datetime],
                 factors: Union[FactorBase, Sequence[FactorBase]] = None,
                 market_cap: BinaryFactor = None, industry: IndustryFactor = None):
        self.forward_return = forward_return
        self.ticker_selector = ticker_selector
        self.industry = industry
        self.dates = dates
        self.market_cap = market_cap

        self.cache_data = None
        self.market_cap_name = None

        self.factors = {}
        self.factor_names = []
        self.append_factor(factors)
        self.sorting_factor = []

    def append_factor(self, factor: Union[FactorBase, Sequence[FactorBase]]):
        if factor:
            if isinstance(factor, Sequence):
                for f in factor:
                    self.append_factor(f)
            else:
                if factor.name not in self.factor_names:
                    self.factors[factor.name] = factor
                    self.factor_names.append(factor.name)
                    if self.cache_data:
                        data = factor.get_data(dates=self.dates, ticker_selector=self.ticker_selector)
                        self.cache_data = pd.concat([self.cache_data, data], axis=1).dropna()

    def cache(self):
        logging.getLogger(__name__).info('Cache cross-sectional data')
        storage = [self.forward_return.get_data(dates=self.dates, ticker_selector=self.ticker_selector)]
        if self.market_cap:
            logging.getLogger(__name__).debug('Cache market cap data')
            storage.append(self.market_cap.get_data(dates=self.dates, ticker_selector=self.ticker_selector))
            self.market_cap_name = self.market_cap.name
        else:
            self.market_cap_name = 'cap_weight'
        if self.industry:
            logging.getLogger(__name__).debug('Cache industry data')
            storage.append(self.industry.get_data(dates=self.dates, ticker_selector=self.ticker_selector))
        for it in self.factors.values():
            storage.append(it.get_data(dates=self.dates, ticker_selector=self.ticker_selector))

        self.cache_data = pd.concat(storage, axis=1).dropna()

    def _factor_sorting(self, factor_name: str = None, quantile: int = None, separate_neg_vals: bool = False,
                        gb_vars: Union[str, Sequence[str]] = 'DateTime'):
        var_name = f'G_{factor_name}'
        if var_name in self.cache_data.columns:
            return

        if factor_name is None:
            if len(self.factor_names) == 1:
                factor_name = self.factor_names[0]
            else:
                raise ValueError('Ambiguous factor name, please specify in `factor_name=`')

        quantile_labels = [f'G{i}' for i in range(1, quantile + 1)]
        if separate_neg_vals:
            negative_ind = self.cache_data[factor_name] < 0
        else:
            negative_ind = pd.Series(False, index=self.cache_data.index)

        tmp = self.cache_data.loc[~negative_ind, :].groupby(gb_vars)[factor_name].apply(
            lambda x: pd.qcut(x, quantile, labels=quantile_labels))
        neg_vals = pd.Series('G0', index=self.cache_data.loc[negative_ind, :].index)
        tmp = pd.concat([tmp, neg_vals]).sort_index()

        self.cache_data[var_name] = tmp.values

    def single_factor_sorting(self, factor_name: str = None, quantile: int = None, separate_neg_vals: bool = False):
        self.sorting_factor = factor_name
        self._factor_sorting(factor_name, quantile, separate_neg_vals)

    def two_factor_sorting(self, factor_names: Tuple[str, str], independent: bool,
                           quantile: Union[int, Sequence[float], Tuple[int, int]] = None,
                           separate_neg_vals: Union[bool, Tuple[bool, bool]] = False):
        assert factor_names[0] in self.factor_names and factor_names[1] in self.factor_names, 'Unknown factor name.'
        self.sorting_factor = list(factor_names)
        if not isinstance(quantile, Tuple):
            quantile = (quantile, quantile)
        if not isinstance(separate_neg_vals, Tuple):
            separate_neg_vals = (separate_neg_vals, separate_neg_vals)

        self._factor_sorting(factor_names[0], quantile[0], separate_neg_vals[0])
        if independent:
            self._factor_sorting(factor_names[1], quantile[1], separate_neg_vals[1])
        else:
            self._factor_sorting(factor_names[1], quantile[1], separate_neg_vals[1],
                                 gb_vars=['DateTime', f'G_{factor_names[0]}'])

    def fm_regression(self):
        data = self.cache_data.loc[:, ['forward_return'] + self.cache['factor_names']].copy()
        # need to winsorize
        for factor in self.cache['factor_names']:
            data[factor] = data[factor].apply(lambda x: winsorize(x, (0.25, 0.25)))
        # cross-sectional regression

        # time-series regression

        # test
        pass

    def returns_results(self, cap_weighted: bool = False) -> pd.DataFrame:
        if cap_weighted and not self.market_cap:
            raise ValueError('market cap is not specified.')

        def weighted_ret(x):
            return x[self.forward_return.name].dot(x[self.market_cap_name] / x[self.market_cap_name].sum())

        func = weighted_ret if cap_weighted else np.mean
        g_vars = [f'G_{it}' for it in self.factor_names]
        tmp = self.cache_data.groupby(g_vars + ['DateTime']).apply(func)
        storage = [tmp.groupby(g_vars).mean().reset_index()]
        for var in g_vars:
            t2 = self.cache_data.groupby([var, 'DateTime']).apply(func)
            t3 = t2.groupby(var).mean().reset_index()
            other_var = list(set(g_vars) - {var})[0]
            t3[other_var] = 'ALL'
            storage.append(t3)

        tmp = pd.concat(storage)
        res = tmp.pivot(index=g_vars[0], columns=g_vars[1], values=tmp.columns[-1])
        index = human_sort(res.index.tolist())
        col = human_sort(res.columns.tolist())
        res = res.loc[index, col]
        return res

    def summary_statistics(self, factor_name: str = None) -> pd.DataFrame:
        if factor_name is None:
            if len(self.factor_names) == 1:
                factor_name = self.factor_names[0]
            else:
                raise ValueError('Ambiguous factor name, please specify in `factor_name=`')

        res = self.cache_data.groupby([f'G_{factor_name}', 'DateTime']).mean()
        return res.groupby(f'G_{factor_name}').mean()

    def factor_corr(self, factor_names: Tuple[str, str]) -> pd.Series:
        return self.cache_data.groupby('DateTime').apply(
            lambda x: np.corrcoef(x[factor_names[0]], x[factor_names[1]])[0, 1])


class ASharePortfolioAnalysis(object):
    def __init__(self, db_interface: DBInterface = None):
        super().__init__()
        self.db_interface = db_interface if db_interface else get_db_interface()
        self.data_reader = AShareDataReader(self.db_interface)

    def market_return(self, start_date: DateUtils.DateType, end_date: DateUtils.DateType):
        pass

    def beta_portfolio(self):
        pass

    def size_portfolio(self, start_date: DateUtils.DateType, end_date: DateUtils.DateType) -> pd.DataFrame:
        policy = StockSelectionPolicy(ignore_new_stock_period=360, ignore_st=True)
        selector = StockTickerSelector(policy)
        dates = self.data_reader.calendar.last_day_of_month(start_date, end_date)
        hfq_price = self.data_reader.hfq_close.get_data(dates, ticker_selector=selector).dropna().unstack()
        market_size = self.data_reader.stock_free_floating_market_cap.get_data(dates=dates,
                                                                               ticker_selector=selector).dropna()

        factor_data = alphalens.utils.get_clean_factor_and_forward_returns(market_size, hfq_price, quantiles=10)

        fa = FactorAnalyzer(market_size, hfq_price, bins=5, max_loss=0.5)

        return factor_data

    @staticmethod
    def summary_statistics(factor_data: pd.DataFrame) -> pd.DataFrame:
        storage = []
        for name, group in factor_data['factor'].groupby('date'):
            content = descriptivestats.describe(group).T
            content.index = [name]
            storage.append(content)
        cross_section_info = pd.concat(storage)

        return cross_section_info
