import alphalens
import numpy as np
import pandas as pd
import scipy

from . import AShareDataReader, MySQLInterface, prepare_engine, utils
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

    def size_portfolio(self, start_date: utils.DateType, end_date: utils.DateType):
        dates = self.calendar.last_day_of_month(start_date, end_date)
        price = self.data_reader.get_factor('股票日行情', '收盘价', dates=dates)
        adj_factor = self.data_reader.get_compact_factor('复权因子', dates=dates)
        units = self.data_reader.get_compact_factor('A股流通股本', dates=dates)
        market_cap = price * units
        market_size = market_cap.apply(np.log)
        hfq_price = price * adj_factor

        factor_data = alphalens.utils.get_clean_factor_and_forward_returns(market_size.stack(), hfq_price, quantiles=10)
        alphalens.tears.create_full_tear_sheet(factor_data)

    @staticmethod
    def summary_statistics(factor_data: pd.DataFrame) -> pd.DataFrame:
        cross_section_info = factor_data.groupby('DateTime').agg(
            {'Mean': np.nanmean, 'SD': np.nanstd, 'Skew': scipy.stats.skew, 'Kurt': scipy.stats.kurtosis,
             'Min': np.nanmin, '5%': np.percentile(5), '25%': np.percentile(25), 'Median': np.median,
             '75%': np.percentile(75), '95%': np.percentile(95), 'Max': np.nanmax,
             # 'n':
             }, axis=1)
        return cross_section_info.mean()
