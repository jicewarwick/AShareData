from functools import partial

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

    def size_portfolio(self, start_date: utils.DateType, end_date: utils.DateType) -> pd.DataFrame:
        dates = self.calendar.last_day_of_month(start_date, end_date)
        price = self.data_reader.get_factor('股票日行情', '收盘价', dates=dates)
        adj_factor = self.data_reader.get_compact_factor('复权因子', dates=dates)
        units = self.data_reader.get_compact_factor('A股流通股本', dates=dates)
        market_cap = price * units
        market_size = market_cap.apply(np.log).stack()
        hfq_price = price * adj_factor

        factor_data = alphalens.utils.get_clean_factor_and_forward_returns(market_size, hfq_price, quantiles=10)
        return factor_data

    @staticmethod
    def summary_statistics(factor_data: pd.DataFrame) -> pd.DataFrame:
        quantile_1 = partial(np.percentile, q=1)
        quantile_1.__name__ = "quantile_1"
        quantile_5 = partial(np.percentile, q=5)
        quantile_5.__name__ = "quantile_5"
        quantile_10 = partial(np.percentile, q=10)
        quantile_10.__name__ = "quantile_10"
        quantile_25 = partial(np.percentile, q=25)
        quantile_25.__name__ = "quantile_25"
        quantile_75 = partial(np.percentile, q=75)
        quantile_75.__name__ = "quantile_75"
        quantile_90 = partial(np.percentile, q=90)
        quantile_90.__name__ = "quantile_90"
        quantile_95 = partial(np.percentile, q=95)
        quantile_95.__name__ = "quantile_95"
        quantile_99 = partial(np.percentile, q=99)
        quantile_99.__name__ = "quantile_99"

        cross_section_info = factor_data.stack().groupby('DateTime').agg(
            Mean=np.nanmean, SD=np.nanstd, Skew=scipy.stats.skew, Kurt=scipy.stats.kurtosis, Min=np.nanmin,
            p_1=quantile_1, p_5=quantile_5, p_10=quantile_10, p_25=quantile_25, Median=np.median, p_75=quantile_75,
            p_90=quantile_90, p_95=quantile_95, p_99=quantile_99, Max=np.nanmax, N='size')
        return cross_section_info

