import alphalens
import pandas as pd
from jqfactor_analyzer import FactorAnalyzer
from statsmodels.stats import descriptivestats

from . import AShareDataReader, DateUtils
from .config import get_db_interface
from .DBInterface import DBInterface
from .Tickers import StockTickerSelector
from .utils import StockSelectionPolicy


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
