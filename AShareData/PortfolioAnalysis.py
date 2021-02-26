import datetime as dt
import logging
from typing import Sequence, Union

import alphalens
import pandas as pd
from jqfactor_analyzer import FactorAnalyzer
from scipy.stats.mstats import winsorize
from statsmodels.stats import descriptivestats

from . import AShareDataReader, DateUtils
from .config import get_db_interface
from .DBInterface import DBInterface
from .Factor import BinaryFactor, ContinuousFactor, IndustryFactor, UnaryFactor
from .Tickers import StockTickerSelector
from .utils import sort_nicely, StockSelectionPolicy


class CrossSectionalPortfolioAnalysis(object):
    def __init__(self, forward_return: UnaryFactor, factors: Union[ContinuousFactor, Sequence[ContinuousFactor]],
                 ticker_selector: StockTickerSelector, dates: Sequence[dt.datetime],
                 industry: IndustryFactor = None,
                 market_cap: BinaryFactor = None):
        self.forward_return = forward_return
        self.factors = factors
        self.ticker_selector = ticker_selector
        self.industry = industry
        self.dates = dates
        self.market_cap = market_cap
        self.quantile = None
        self.quantile_labels = None
        self.cache = {}
        self.cache_data = None

    def _append_factor(self, factor: ContinuousFactor):
        if factor.display_factor_name not in self.cache['factor_names']:
            self.cache['factor_names'].append(factor.display_factor_name)
            self.cache['factor_data'][factor.display_factor_name] = \
                factor.get_data(dates=self.dates, ticker_selector=self.ticker_selector)

    def cache_data(self):
        logging.getLogger(__name__).info('Cache cross-sectional data')
        self.cache['forward_return'] = self.forward_return.get_data(dates=self.dates,
                                                                    ticker_selector=self.ticker_selector)
        self.cache['factor_data'] = {}
        self.cache['factor_names'] = []
        if self.market_cap:
            self.cache['factor_data']['market_cap'] = \
                self.forward_return.get_data(self.dates, ticker_selector=self.ticker_selector)
        if not isinstance(self.factors, Sequence):
            self.factors = [self.factors]
        for factor in self.factors:
            self._append_factor(factor)
        self.cache_data = pd.concat([self.cache['forward_return']] + list(self.cache['factor_data'].values()), axis=1)
        self.cache_data = self.cache_data.dropna()

    # def set_quantile(self, quantile: Union[int, Mapping[str, Union[int, Sequence[float]]]],
    #                  quantile_labels: Union[Sequence[str], Mapping[str, Sequence[str]]] = None):
    #     # param sanity check
    #     if isinstance(quantile, int) and quantile_labels is not None:
    #         # assert len(self.cache['factor_names']) == 1, 'ambiguous factor name. please specify as a dict.'
    #         assert len(quantile_labels) == quantile, 'parameter size do not match.'
    #
    #     if quantile_labels is None:
    #         if isinstance(quantile, int):
    #             quantile_labels = [f'L{i}' for i in range(1, quantile + 1)]
    #         else:
    #             quantile_labels = [f'L{i}' for i in range(1, len(quantile) + 1)]
    #
    #     self.quantile_labels = quantile_labels

    def set_quantile(self, quantile: int):
        self.quantile = quantile
        self.quantile_labels = [f'G{i}' for i in range(1, quantile + 1)]
        for factor_name in self.cache['factor_names']:
            tmp = self.cache_data.groupby('DateTime').apply(
                lambda x: pd.qcut(x.loc[:, factor_name], quantile, labels=self.quantile_labels))
            self.cache_data[f'G_{factor_name}'] = tmp.values

    def fm_regression(self):
        data = self.cache_data.loc[:, ['forward_return'] + self.cache['factor_names']].copy()
        # need to winsorize
        for factor in self.cache['factor_names']:
            data[factor] = data[factor].apply(lambda x: winsorize(x, (0.25, 0.25)))
        # cross-sectional regression

        # time-series regression

        # test
        pass

    def returns_results(self) -> pd.DataFrame:
        g_vars = [f'G_{it}' for it in self.cache['factor_names']]
        tmp = self.cache_data.groupby(g_vars + ['DateTime'])[self.forward_return.display_factor_name].mean()
        storage = [tmp.groupby(g_vars).mean().reset_index()]
        for var in g_vars:
            t2 = self.cache_data.groupby([var, 'DateTime'])[self.forward_return.display_factor_name].mean()
            t3 = t2.groupby(var).mean().reset_index()
            other_var = list(set(g_vars) - {var})[0]
            t3[other_var] = 'ALL'
            storage.append(t3)

        res = pd.concat(storage).pivot(index=g_vars[0], columns=g_vars[1],
                                       values=self.forward_return.display_factor_name)
        index = sort_nicely(res.index.tolist())
        col = sort_nicely(res.columns.tolist())
        res = res.loc[index, col]
        return res

    def summary_statistics(self, factor_name: str = None) -> pd.DataFrame:
        if factor_name is None and self.cache['']:
            pass
        assert factor_name in self.cache['factor_names']
        res = self.cache_data.groupby([f'G_{factor_name}', 'DateTime']).mean()
        return res.groupby(f'G_{factor_name}').mean()


class ASharePortfolioAnalysis(object):
    def __init__(self, db_interface: DBInterface = None):
        super().__init__()
        if db_interface is None:
            db_interface = get_db_interface()
        self.db_interface = db_interface
        self.data_reader = AShareDataReader(db_interface)

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
