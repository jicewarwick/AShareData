import datetime as dt

import alphalens
import pandas as pd

from . import AShareDataReader, MySQLInterface, prepare_engine
from .TradingCalendar import TradingCalendar


class ASharePortfolioAnalysis(object):
    def __init__(self, config_loc):
        super().__init__()
        engine = prepare_engine(config_loc)
        mysql_writer = MySQLInterface(engine, init=True)
        self.data_reader = AShareDataReader(mysql_writer)
        self.calendar = TradingCalendar(mysql_writer)

    def beta_portfolio(self):
        pass

    def size_portfolio(self):
        price = self.data_reader.get_factor('股票日行情', '收盘价', start_date=dt.date(2010, 1, 1),
                                            end_date=dt.date(2019, 12, 31))
        units = self.data_reader.get_factor('总股本', '总股本', ffill=True, start_date=dt.date(2010, 1, 1),
                                            end_date=dt.date(2019, 12, 31))
        market_cap = price * units
        market_size = market_cap.log()

        factor_data = alphalens.utils.get_clean_factor_and_forward_returns(market_size, price, quantiles=10)
        alphalens.tears.create_full_tear_sheet(factor_data)

    @staticmethod
    def summary_statistics(data) -> pd.DataFrame:
        pass
