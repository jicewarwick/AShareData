import datetime as dt

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import utils
from ..ashare_data_reader import AShareDataReader
from ..data_source.data_source import DataSource
from ..database_interface import DBInterface
from ..factor import CompactFactor
from ..tickers import FundTickers, StockTickerSelector


class FactorCompositor(DataSource):
    def __init__(self, db_interface: DBInterface = None):
        """
        Factor Compositor

        This class composite factors from raw market/financial info

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.data_reader = AShareDataReader(db_interface)

    def update(self):
        """更新数据"""
        raise NotImplementedError()


class IndexCompositor(FactorCompositor):
    def __init__(self, index_composition_policy: utils.StockIndexCompositionPolicy, db_interface: DBInterface = None):
        """自建指数收益计算器"""
        super().__init__(db_interface)
        self.table_name = '自合成指数'
        self.policy = index_composition_policy
        self.weight = None
        if index_composition_policy.unit_base:
            self.weight = (CompactFactor(index_composition_policy.unit_base, self.db_interface)
                           * self.data_reader.stock_close).weight()
        self.stock_ticker_selector = StockTickerSelector(self.policy.stock_selection_policy, self.db_interface)

    def update(self):
        """ 更新市场收益率 """
        price_table = '股票日行情'

        start_date = self.db_interface.get_latest_timestamp(self.table_name, self.policy.start_date,
                                                            column_condition=('ID', self.policy.ticker))
        end_date = self.db_interface.get_latest_timestamp(price_table)
        dates = self.calendar.select_dates(start_date, end_date, inclusive=(False, True))

        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'{date}')
                ids = self.stock_ticker_selector.ticker(date)

                if ids:
                    t_dates = [(self.calendar.offset(date, -1)), date]
                    if self.weight:
                        rets = (self.data_reader.forward_return * self.weight).sum().get_data(dates=t_dates, ids=ids)
                    else:
                        rets = self.data_reader.stock_return.mean(along='DateTime').get_data(dates=t_dates, ids=ids)
                    index = pd.MultiIndex.from_tuples([(date, self.policy.ticker)], names=['DateTime', 'ID'])
                    ret = pd.Series(rets.values[0], index=index, name='收益率')

                    self.db_interface.update_df(ret, self.table_name)
                pbar.update()


class IndexUpdater(object):
    def __init__(self, config_loc=None, db_interface: DBInterface = None):
        """ 指数更新器

        :param config_loc: 配置文件路径. 默认指数位于 ``./data/自编指数配置.xlsx``. 自定义配置可参考此文件
        :param db_interface: DBInterface
        """
        super().__init__()
        self.db_interface = db_interface
        records = utils.load_excel('自编指数配置.xlsx', config_loc)
        self.policies = {}
        for record in records:
            self.policies[record['name']] = utils.StockIndexCompositionPolicy.from_dict(record)

    def update(self):
        with tqdm(self.policies) as pbar:
            for policy in self.policies.values():
                pbar.set_description(f'更新{policy.name}')
                IndexCompositor(policy, db_interface=self.db_interface).update()
                pbar.update()


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
                        pre_price = pre_data.loc[(slice(None), target_stocks), '最高价'] * adj_factor.loc[
                            (pre_date, target_stocks)]
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


class FundAdjFactorCompositor(FactorCompositor):
    def __init__(self, db_interface: DBInterface = None):
        """
        计算基金的复权因子

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.fund_tickers = FundTickers(self.db_interface)
        self.target_table_name = '复权因子'
        self.div_table_name = '公募基金分红'
        self.cache_entry = '基金复权因子'

    def compute_adj_factor(self, ticker):
        list_date = self.fund_tickers.get_list_date(ticker)
        index = pd.MultiIndex.from_tuples([(list_date, ticker)], names=('DateTime', 'ID'))
        list_date_adj_factor = pd.Series(1, index=index, name=self.target_table_name)
        self.db_interface.update_df(list_date_adj_factor, self.target_table_name)

        div_info = self.db_interface.read_table(self.div_table_name, ids=ticker)
        if div_info.empty:
            return
        div_dates = div_info.index.get_level_values('DateTime').tolist()
        pre_date = [self.calendar.offset(it, -1) for it in div_dates]

        if ticker.endswith('.OF'):
            price_table_name, col_name = '场外基金净值', '单位净值'
        else:
            price_table_name, col_name = '场内基金日行情', '收盘价'
        pre_price_data = self.db_interface.read_table(price_table_name, col_name, dates=pre_date, ids=ticker)
        if pre_price_data.shape[0] != div_info.shape[0]:
            price_data = self.db_interface.read_table(price_table_name, col_name, ids=ticker)
            tmp = pd.concat([price_data.shift(1), div_info], axis=1)
            pre_price_data = tmp.dropna().iloc[:, 0]
        else:
            pre_price_data.index = div_info.index
        adj_factor = (pre_price_data / (pre_price_data - div_info)).cumprod()
        adj_factor.name = self.target_table_name
        self.db_interface.update_df(adj_factor, self.target_table_name)

    def update(self):
        update_time = self.db_interface.get_cache_date(self.cache_entry)
        if update_time is None:
            tickers = self.fund_tickers.all_ticker()
        else:
            next_day = self.calendar.offset(update_time, 1)
            fund_div_table = self.db_interface.read_table(self.div_table_name, start_date=next_day)
            tickers = fund_div_table.index.get_level_values('ID').unique().tolist()
        with tqdm(tickers) as pbar:
            for ticker in tickers:
                pbar.set_description(f'更新 {ticker} 的复权因子')
                self.compute_adj_factor(ticker)
                pbar.update()
        timestamp = self.db_interface.get_latest_timestamp(self.div_table_name)
        self.db_interface.update_cache_date(self.cache_entry, timestamp)


class NegativeBookEquityListingCompositor(FactorCompositor):
    def __init__(self, db_interface: DBInterface = None):
        """标识负净资产股票

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.table_name = '负净资产股票'

    def update(self):
        data = self.db_interface.read_table('合并资产负债表', '股东权益合计(不含少数股东权益)')
        storage = []
        for _, group in data.groupby('ID'):
            if any(group < 0):
                tmp = group.groupby('DateTime').tail(1) < 0
                t = tmp.iloc[tmp.argmax():].droplevel('报告期')
                t2 = t[np.concatenate(([True], t.values[:-1] != t.values[1:]))]
                if any(t2):
                    storage.append(t2)

        ret = pd.concat(storage)
        ret.name = '负净资产股票'
        self.db_interface.purge_table(self.table_name)
        self.db_interface.insert_df(ret, self.table_name)


class MarketSummaryCompositor(FactorCompositor):
    def __init__(self, policy_name: str = '全市场', db_interface: DBInterface = None):
        """市场汇总

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.table_name = '市场汇总'
        self.init_date = dt.datetime(2001, 1, 1)
        records = utils.load_excel('自编指数配置.xlsx')
        policies = {}
        for record in records:
            policies[record['name']] = utils.StockIndexCompositionPolicy.from_dict(record)
        stock_selection_policy = policies[policy_name]
        self.ticker = stock_selection_policy.ticker
        self.stock_ticker_selector = StockTickerSelector(stock_selection_policy.stock_selection_policy,
                                                         self.db_interface)
        self.total_share = CompactFactor('A股总股本', self.db_interface)
        self.float_share = CompactFactor('A股流通股本', self.db_interface)
        self.free_float_share = CompactFactor('自由流通股本', self.db_interface)

    def update(self):
        price_table_name = '股票日行情'

        start_date = self.db_interface.get_latest_timestamp(self.table_name, self.init_date)
        end_date = self.db_interface.get_latest_timestamp(price_table_name, dt.date(1990, 12, 10))
        dates = self.calendar.select_dates(start_date, end_date, (False, True))
        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'更新市场汇总: {date}')
                tickers = self.stock_ticker_selector.ticker(date)
                daily_info = self.db_interface.read_table(price_table_name, ['收盘价', '成交额'], ids=tickers, dates=date)
                total_share = self.total_share.get_data(dates=date, ids=tickers)
                float_share = self.float_share.get_data(dates=date, ids=tickers)
                free_float_share = self.free_float_share.get_data(dates=date, ids=tickers)

                vol = daily_info['成交额'].sum()
                float_val = daily_info['收盘价'].dot(float_share)
                free_float_val = daily_info['收盘价'].dot(free_float_share)
                total_df = daily_info['收盘价'] * total_share
                total_val = total_df.sum()

                earning_ttm = self.data_reader.earning_ttm.get_data(ids=tickers, dates=date)
                earning_data = pd.concat([total_df, earning_ttm], axis=1).dropna()
                pe_val = earning_data.iloc[:, 0].sum() / earning_data.iloc[:, 1].sum()

                book_val = self.data_reader.book_val.get_data(ids=tickers, dates=date)
                book_data = pd.concat([total_df, book_val], axis=1).dropna()
                pb_val = book_data.iloc[:, 0].sum() / book_data.iloc[:, 1].sum()

                sum_dict = {'DateTime': date, 'ID': self.ticker, '成交额': vol, '总市值': total_val, '流通市值': float_val,
                            '自由流通市值': free_float_val, '市盈率TTM': pe_val, '市净率': pb_val}

                df = pd.DataFrame(sum_dict, index=['DateTime']).set_index(['DateTime', 'ID'])
                self.db_interface.insert_df(df, self.table_name)

                pbar.update()
