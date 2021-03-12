import logging

import pandas as pd
from tqdm import tqdm

from .FactorCompositor import FactorCompositor
from ..DBInterface import DBInterface
from ..Tickers import FundTickers


class FundAdjFactorCompositor(FactorCompositor):
    def __init__(self, db_interface: DBInterface = None):
        """
        计算基金的复权因子

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.fund_tickers = FundTickers(self.db_interface)

    def compute_adj_factor(self, ticker):
        table_name = '复权因子'
        div_table_name = '公募基金分红'

        list_date = self.fund_tickers.get_list_date(ticker)
        index = pd.MultiIndex.from_tuples([(list_date, ticker)], names=('DateTime', 'ID'))
        list_date_adj_factor = pd.Series(1, index=index, name=table_name)
        self.db_interface.update_df(list_date_adj_factor, table_name)

        div_info = self.db_interface.read_table(div_table_name, ids=ticker)
        if div_info.empty:
            return
        div_dates = div_info.index.get_level_values('DateTime').tolist()
        after_date = [self.calendar.offset(it, 1) for it in div_dates]

        if ticker.endswith('.OF'):
            price_table_name, col_name = '场外基金净值', '单位净值'
        else:
            price_table_name, col_name = '场内基金日行情', '收盘价'
        price_data = self.db_interface.read_table(price_table_name, col_name, dates=div_dates, ids=ticker)
        if price_data.shape[0] != div_info.shape[0]:
            logging.getLogger(__name__).warning(f'{ticker}的价格信息不完全')
            return
        adj_factor = (price_data / (price_data - div_info)).cumprod()
        adj_factor.index = adj_factor.index.set_levels(after_date, level=0)
        adj_factor.name = table_name
        self.db_interface.update_df(adj_factor, table_name)

    def update(self):
        all_tickers = self.fund_tickers.all_ticker()
        for ticker in tqdm(all_tickers):
            self.compute_adj_factor(ticker)
