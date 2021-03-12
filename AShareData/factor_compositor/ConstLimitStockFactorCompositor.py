import datetime as dt

import pandas as pd
from tqdm import tqdm

from .. import utils
from ..DBInterface import DBInterface
from .FactorCompositor import FactorCompositor
from ..Tickers import StockTickerSelector


class ConstLimitStockFactorCompositor(FactorCompositor):
    def __init__(self, db_interface: DBInterface = None):
        """
        标识一字涨跌停板

        判断方法: 取最高价和最低价一致 且 当日未停牌
         - 若价格高于昨前复权价, 则视为涨停一字板
         - 若价格低于昨前复权价, 则视为跌停一字板

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.table_name = '一字涨跌停'
        stock_selection_policy = utils.StockSelectionPolicy(select_pause=True)
        self.paused_stock_selector = StockTickerSelector(stock_selection_policy, db_interface)

    def update(self):
        price_table_name = '股票日行情'

        start_date = self.db_interface.get_latest_timestamp(self.table_name, dt.date(1999, 5, 4))
        end_date = self.db_interface.get_latest_timestamp(price_table_name, dt.date(1990, 12, 10))

        pre_data = self.db_interface.read_table(price_table_name, ['最高价', '最低价'], dates=start_date)
        dates = self.calendar.select_dates(start_date, end_date)
        pre_date = dates[0]
        dates = dates[1:]

        with tqdm(dates) as pbar:
            pbar.set_description('更新股票一字板')
            for date in dates:
                data = self.db_interface.read_table(price_table_name, ['最高价', '最低价'], dates=date)
                no_price_move_tickers = data.loc[data['最高价'] == data['最低价']].index.get_level_values('ID').tolist()
                if no_price_move_tickers:
                    target_stocks = list(set(no_price_move_tickers) - set(self.paused_stock_selector.ticker(date)))
                    if target_stocks:
                        adj_factor = self.data_reader.adj_factor.get_data(start_date=pre_date, end_date=date,
                                                                          ids=target_stocks)
                        price = data.loc[(slice(None), target_stocks), '最高价'] * adj_factor.loc[(date, target_stocks)]
                        pre_price = pre_data.loc[(slice(None), target_stocks), '最高价'] * \
                                    adj_factor.loc[(pre_date, target_stocks)]
                        diff_price = pd.concat([pre_price, price]).unstack().diff().iloc[1, :].dropna()
                        diff_price = diff_price.loc[diff_price != 0]
                        if diff_price.shape[0] > 1:
                            ret = (diff_price > 0) * 2 - 1
                            ret = ret.to_frame().reset_index()
                            ret['DateTime'] = date
                            ret.set_index(['DateTime', 'ID'], inplace=True)
                            ret.columns = ['涨跌停']
                            self.db_interface.insert_df(ret, self.table_name)
                pre_data = data
                pre_date = date
                pbar.update()
