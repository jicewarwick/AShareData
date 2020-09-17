import datetime as dt
from tqdm import tqdm

import pandas as pd

from AShareData.DataSource import DataSource
from . import AShareDataReader, DBInterface


class FactorCompositor(DataSource):
    def __init__(self, db_interface: DBInterface):
        """
        Factor Compositor

        This class composite factors from raw market/financial info
        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.data_reader = AShareDataReader(db_interface)

    def compute_const_limit(self):
        """ 标识一字涨跌停板的

        判断方法: 取最高价和最低价一致 且 当日未停牌
         - 若价格高于昨前复权价, 则视为涨停一字板
         - 若价格低于昨前复权价, 则视为跌停一字板
        """
        table_name = '一字涨跌停'
        price_table_name = '股票日行情'
        pause_table_name = '股票停牌'

        start_date = self._check_db_timestamp(table_name, dt.date(1999, 5, 4))
        end_date = self._check_db_timestamp(price_table_name, dt.date(1990, 12, 10))

        pre_data = self.db_interface.read_table(price_table_name, ['最高价', '最低价'], dates=[start_date])
        dates = self.calendar.select_dates(start_date, end_date)
        pre_date = dates[0]
        dates = dates[1:]

        with tqdm(dates) as pbar:
            pbar.set_description('更新股票一字板')
            for date in dates:
                data = self.db_interface.read_table(price_table_name, ['最高价', '最低价'], dates=[date])
                no_price_move_tickers = data.loc[data['最高价'] == data['最低价']].index.get_level_values('ID').tolist()
                if no_price_move_tickers:
                    paused_stocks = self.db_interface.read_table(pause_table_name, '停牌类型', dates=[date])
                    paused_stocks = paused_stocks.index.get_level_values('ID')
                    target_stocks = list(set(no_price_move_tickers) - set(paused_stocks))
                    if target_stocks:
                        adj_factor = self.data_reader.adj_factor(start_date=pre_date, end_date=date, ids=target_stocks)
                        price = data.loc[(slice(None), target_stocks), '最高价'] * adj_factor.loc[(date, target_stocks)]
                        pre_price = pre_data.loc[(slice(None), target_stocks), '最高价'] * adj_factor.loc[
                            (date, target_stocks)]
                        diff_price = pd.concat([pre_price, price]).unstack().diff().iloc[1, :].dropna()
                        diff_price = diff_price.loc[diff_price != 0]
                        if diff_price.shape[0] > 1:
                            ret = (diff_price > 0) * 2 - 1
                            ret = ret.to_frame().reset_index()
                            ret['DateTime'] = date
                            ret.set_index(['DateTime', 'ID'], inplace=True)
                            ret.columns = ['涨跌停']
                            self.db_interface.insert_df(ret, table_name)
                pre_data = data
                pre_date = date
                pbar.update()
