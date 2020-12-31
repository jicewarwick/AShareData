import alphalens
import numpy as np
import pandas as pd
from statsmodels.stats import descriptivestats

from . import AShareDataReader, DateUtils
from .config import get_db_interface
from .DBInterface import DBInterface


class ASharePortfolioAnalysis(object):
    def __init__(self, db_interface: DBInterface = None):
        super().__init__()
        if not db_interface:
            db_interface = get_db_interface()
        self.db_interface = db_interface
        self.data_reader = AShareDataReader(db_interface)

    def market_return(self, start_date: DateUtils.DateType, end_date: DateUtils.DateType):
        pass

    def beta_portfolio(self):
        pass

    def size_portfolio(self, start_date: DateUtils.DateType, end_date: DateUtils.DateType) -> pd.DataFrame:
        dates = self.data_reader.calendar.last_day_of_month(start_date, end_date)
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
            content = descriptivestats.describe(group).T
            content.index = [name]
            storage.append(content)
        cross_section_info = pd.concat(storage)

        return cross_section_info
