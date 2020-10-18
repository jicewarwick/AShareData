import datetime as dt

import pandas as pd
from tqdm import tqdm

from . import AShareDataReader, DBInterface, utils
from .AShareDataReader import CompactFactor
from .DataSource import DataSource


class FactorCompositor(DataSource):
    def __init__(self, db_interface: DBInterface):
        """
        Factor Compositor

        This class composite factors from raw market/financial info
        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.data_reader = AShareDataReader(db_interface)

    def update_const_limit_stock(self):
        """ 标识一字涨跌停板

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
                        adj_factor = self.data_reader.adj_factor.get_data(start_date=pre_date, end_date=date,
                                                                          ids=target_stocks)
                        price = data.loc[(slice(None), target_stocks), '最高价'] * adj_factor.loc[(date, target_stocks)]
                        pre_price = pre_data.loc[(slice(None), target_stocks), '最高价'] * \
                                    adj_factor.loc[(date, target_stocks)]
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

    @utils.format_input_dates
    def update_market_return(self, ticker: str, ignore_st: bool = True, ignore_new_stock_period: dt.timedelta = None,
                             ignore_pause: bool = True, ignore_const_limit: bool = True,
                             unit_base: str = '自由流通股本', start_date: utils.DateType = None):
        """ 更新市场收益率

        :param ticker: 新建指数入库代码. 建议以`.IND`结尾, 代表自合成指数
        :param ignore_st: 排除 风险警告股, 包括 PT, ST, SST, *ST, (即将)退市股 等
        :param ignore_new_stock_period: 新股纳入市场收益计算的时间
        :param ignore_pause: 排除停牌股
        :param ignore_const_limit: 排除一字板股票
        :param unit_base: 股本字段
        :param start_date: 指数开始时间
        """
        assert unit_base in ['自由流通股本', '总股本', 'A股流通股本', 'A股总股本'], '非法股本字段!'
        table_name = '自合成指数'
        price_table = '股票日行情'

        end_date = self.db_interface.get_latest_timestamp(price_table)
        dates = self.calendar.select_dates(start_date, end_date)
        units_factor = CompactFactor(self.db_interface, unit_base)

        with tqdm(dates) as pbar:
            for date in dates:
                ids = set(self.data_reader.stocks(self.calendar.offset(date, -ignore_new_stock_period.days))) & \
                      set(self.data_reader.stocks(date))
                if ignore_pause:
                    ids = ids - set(self.data_reader.paused_stocks.get_data(date))
                if ignore_const_limit:
                    ids = ids - set(self.data_reader.const_limit.get_data(date))
                if ignore_st:
                    ids = ids - set(self.data_reader.risk_warned_stocks.get_data(date=date))
                ids = list(ids)

                # pre data
                pre_date = self.calendar.offset(date, -1)
                pre_units = units_factor.get_data(dates=pre_date, ids=ids)
                pre_close_data = self.data_reader.close().get_data(dates=pre_date, ids=ids)
                pre_adj = self.data_reader.adj_factor.get_data(dates=pre_date, ids=ids)

                # data
                close_data = self.data_reader.close().get_data(dates=date, ids=ids)
                adj = self.data_reader.adj_factor.get_data(dates=date, ids=ids)

                # computation
                stock_daily_ret = (close_data * adj).values / (pre_close_data * pre_adj).values - 1
                weight = pre_units * pre_close_data
                weight = weight / weight.sum(axis=1).values[0]
                daily_ret = stock_daily_ret.dot(weight.T.values)[0][0]

                # write to db
                index = pd.MultiIndex.from_tuples([(date, ticker)], names=['DateTime', 'ID'])
                ret = pd.Series(daily_ret, index=index, name='收益率')
                self.db_interface.update_df(ret, table_name)

                pbar.update()
