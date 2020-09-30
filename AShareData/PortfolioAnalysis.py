import alphalens
import numpy as np
import pandas as pd
import statsmodels as sm

from . import AShareDataReader, MySQLInterface, prepare_engine, utils
from .TradingCalendar import TradingCalendar


class ASharePortfolioAnalysis(object):
    def __init__(self, config_loc):
        super().__init__()
        engine = prepare_engine(config_loc)
        mysql_writer = MySQLInterface(engine, init=True)
        self.data_reader = AShareDataReader(mysql_writer)
        self.calendar = TradingCalendar(mysql_writer)

    def market_return(self, start_date: utils.DateType, end_date: utils.DateType):
        pass

    def beta_portfolio(self):
        pass

    def size_portfolio(self, start_date: utils.DateType, end_date: utils.DateType) -> pd.DataFrame:
        dates = self.calendar.last_day_of_month(start_date, end_date)
        price = self.data_reader.close.get_data(dates=dates)
        adj_factor = self.data_reader.adj_factor.get_data(dates=dates)
        units = self.data_reader.free_a_shares.get_data(dates=dates)
        market_cap = price * units
        market_size = market_cap.apply(np.log).stack()
        hfq_price = price * adj_factor

        factor_data = alphalens.utils.get_clean_factor_and_forward_returns(market_size, hfq_price, quantiles=10)
        return factor_data

    @staticmethod
    def summary_statistics(factor_data: pd.DataFrame) -> pd.DataFrame:
        storage = []
        for name, group in factor_data['factor'].groupby('date'):
            content = sm.stats.descriptivestats.describe(group).T
            content.index = [name]
            storage.append(content)
        cross_section_info = pd.concat(storage)

        return cross_section_info
