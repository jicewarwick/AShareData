import datetime as dt
import os
import pickle
import zipfile
from pathlib import Path
from typing import Sequence

import pandas as pd
from sortedcontainers import SortedDict
from tqdm import tqdm

from . import AShareDataReader, constants, DateUtils, DBInterface, utils
from .DataSource import DataSource
from .Factor import CompactFactor
from .Tickers import StockTickerSelector


class FactorCompositor(DataSource):
    def __init__(self, db_interface: DBInterface):
        """
        Factor Compositor

        This class composite factors from raw market/financial info
        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.data_reader = AShareDataReader(db_interface)

    def update(self):
        raise NotImplementedError()


class ConstLimitStockFactorCompositor(FactorCompositor):
    def __init__(self, db_interface: DBInterface):
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
        self.paused_stock_selector = StockTickerSelector(db_interface, stock_selection_policy)

    def update(self):
        price_table_name = '股票日行情'

        start_date = self._check_db_timestamp(self.table_name, dt.date(1999, 5, 4))
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
                    target_stocks = list(set(no_price_move_tickers) - set(self.paused_stock_selector.ticker(date)))
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
                            self.db_interface.insert_df(ret, self.table_name)
                pre_data = data
                pre_date = date
                pbar.update()


class IndexCompositor(FactorCompositor):
    def __init__(self, db_interface: DBInterface, index_composition_policy: utils.IndexCompositionPolicy):
        super().__init__(db_interface)
        self.table_name = '自合成指数'
        self.policy = index_composition_policy
        self.units_factor = CompactFactor(self.db_interface, index_composition_policy.unit_base)
        self.stock_ticker_selector = StockTickerSelector(self.db_interface, self.policy.stock_selection_policy)

    def update(self):
        """ 更新市场收益率 """
        price_table = '股票日行情'

        start_date = self._check_db_timestamp(self.table_name, self.policy.start_date,
                                              column_condition=('ID', self.policy.ticker))
        end_date = self.db_interface.get_latest_timestamp(price_table)
        dates = self.calendar.select_dates(start_date, end_date)

        with tqdm(dates) as pbar:
            for date in dates:
                ids = self.stock_ticker_selector.ticker(date)

                daily_ret = self._compute_ret(date, ids)
                index = pd.MultiIndex.from_tuples([(date, self.policy.ticker)], names=['DateTime', 'ID'])
                ret = pd.Series(daily_ret, index=index, name='收益率')

                # write to db
                self.db_interface.update_df(ret, self.table_name)
                pbar.update()

    def _compute_ret(self, date: dt.datetime, ids: Sequence[str]):
        # pre data
        pre_date = self.calendar.offset(date, -1)
        pre_units = self.units_factor.get_data(dates=pre_date, ids=ids)
        pre_close_data = self.data_reader.close.get_data(dates=pre_date, ids=ids)
        pre_adj = self.data_reader.adj_factor.get_data(dates=pre_date, ids=ids)

        # data
        close_data = self.data_reader.close.get_data(dates=date, ids=ids)
        adj = self.data_reader.adj_factor.get_data(dates=date, ids=ids)

        # computation
        stock_daily_ret = (close_data * adj).values / (pre_close_data * pre_adj).values - 1
        weight = pre_units * pre_close_data
        weight = weight / weight.sum(axis=1).values[0]
        daily_ret = stock_daily_ret.dot(weight.T.values)[0][0]
        return daily_ret


class AccountingDateCacheCompositor(FactorCompositor):
    def __init__(self, db_interface):
        super().__init__(db_interface)

    def update(self):
        dir_name = os.path.join(Path.home(), '.AShareData')
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        output_loc = os.path.join(dir_name, constants.ACCOUNTING_DATE_CACHE_NAME)
        if os.path.exists(output_loc):
            with zipfile.ZipFile(output_loc, 'r', compression=zipfile.ZIP_DEFLATED) as zf:
                with zf.open('cache.pkl') as f:
                    cache = pickle.load(f)
        else:
            cache = {'DateTime': {}, 'cache_date': SortedDict()}
        table_name = '合并资产负债表'
        column_name = '货币资金'

        all_ticker = self.db_interface.get_all_id(table_name)
        with tqdm(all_ticker) as pbar:
            for ticker in all_ticker:
                if ticker not in cache['DateTime'].keys():
                    cache_date = dt.datetime(1900, 1, 1)
                else:
                    cache_date = cache['DateTime'][ticker]
                db_date = self.db_interface.get_latest_timestamp(table_name, ('ID', ticker))
                if db_date > cache_date:
                    ticker_data = self.db_interface.read_table(table_name, column_name, ids=[ticker])
                    index = ticker_data.index
                    new_dates_index = index[index.get_level_values('DateTime') > cache_date]
                    new_dates = pd.to_datetime(new_dates_index.get_level_values('DateTime').unique())
                    for date in new_dates:
                        info = ticker_data.loc[index.get_level_values('DateTime') <= date, :]
                        newest_entry = info.tail(1)
                        report_date = pd.to_datetime(newest_entry.index.get_level_values('报告期')[0])
                        date = self.calendar.offset(date, 0)

                        # q1
                        qoq_date = DateUtils.ReportingDate.qoq_date(report_date)
                        qoq_relevant_entry = info.loc[info.index.get_level_values('报告期') == qoq_date].tail(1)
                        q1 = None if qoq_relevant_entry.empty else qoq_relevant_entry.index.values[0]
                        # q2
                        qoq2_date = DateUtils.ReportingDate.qoq_date(qoq_date)
                        qoq2_relevant_entry = info.loc[info.index.get_level_values('报告期') == qoq2_date].tail(1)
                        q2 = None if qoq2_relevant_entry.empty else qoq2_relevant_entry.index.values[0]
                        # q4
                        yoy_date = DateUtils.ReportingDate.yoy_date(report_date)
                        yoy_relevant_entry = info.loc[info.index.get_level_values('报告期') == yoy_date].tail(1)
                        q4 = None if yoy_relevant_entry.empty else yoy_relevant_entry.index.values[0]
                        # q5
                        qoq_date = DateUtils.ReportingDate.qoq_date(report_date)
                        qoq_relevant_entry = info.loc[info.index.get_level_values('报告期') == qoq_date].tail(1)
                        q5 = None if qoq_relevant_entry.empty else qoq_relevant_entry.index.values[0]

                        # yearly
                        pre_yearly_date = DateUtils.ReportingDate.yearly_dates_offset(report_date)
                        pre_yearly_relevant_entry = info.loc[
                            info.index.get_level_values('报告期') == pre_yearly_date].tail(1)
                        y1 = None if pre_yearly_relevant_entry.empty else pre_yearly_relevant_entry.index.values[0]
                        # 3 yrs ago
                        pre_3_yearly_date = DateUtils.ReportingDate.yearly_dates_offset(report_date, 3)
                        pre_3_yearly_relevant_entry = info.loc[
                            info.index.get_level_values('报告期') == pre_3_yearly_date].tail(1)
                        y3 = None if pre_3_yearly_relevant_entry.empty else pre_3_yearly_relevant_entry.index.values[0]
                        # 5 yrs ago
                        pre_5_yearly_date = DateUtils.ReportingDate.yearly_dates_offset(report_date, 5)
                        pre_5_yearly_relevant_entry = info.loc[
                            info.index.get_level_values('报告期') == pre_5_yearly_date].tail(1)
                        y5 = None if pre_5_yearly_relevant_entry.empty else pre_5_yearly_relevant_entry.index.values[0]

                        # cache_date
                        cache['cache_date'][(ticker, date)] = \
                            utils.DateCache(newest_entry.index.values[0], q1, q2, y1, q4, q5, y3, y5)

                    cache['DateTime'][ticker] = new_dates[-1]

                pbar.update()

        with zipfile.ZipFile(output_loc, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('cache.pkl', pickle.dumps(cache))
