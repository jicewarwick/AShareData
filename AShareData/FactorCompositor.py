import datetime as dt
import logging
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from tqdm import tqdm

from . import utils
from .AShareDataReader import AShareDataReader
from .data_source.DataSource import DataSource
from .DBInterface import DBInterface
from .Factor import CompactFactor, Factor, FactorBase, IndustryFactor
from .Tickers import FundTickers, StockTickerSelector


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

        div_info = self.db_interface.read_table(div_table_name, ids=[ticker])
        if div_info.empty:
            return
        div_dates = div_info.index.get_level_values('DateTime').tolist()
        after_date = [self.calendar.offset(it, 1) for it in div_dates]

        if ticker.endswith('.OF'):
            price_table_name, col_name = '场外基金净值', '单位净值'
        else:
            price_table_name, col_name = '场内基金日行情', '收盘价'
        price_data = self.db_interface.read_table(price_table_name, col_name, dates=div_dates, ids=[ticker])
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


class IndexCompositor(FactorCompositor):
    def __init__(self, index_composition_policy: utils.StockIndexCompositionPolicy, db_interface: DBInterface = None):
        """自建指数收益计算器"""
        super().__init__(db_interface)
        self.table_name = '自合成指数'
        self.policy = index_composition_policy
        self.units_factor = CompactFactor(index_composition_policy.unit_base, self.db_interface)
        self.stock_ticker_selector = StockTickerSelector(self.policy.stock_selection_policy, self.db_interface)

    def update(self):
        """ 更新市场收益率 """
        price_table = '股票日行情'

        start_date = self._check_db_timestamp(self.table_name, self.policy.start_date,
                                              column_condition=('ID', self.policy.ticker))
        end_date = self.db_interface.get_latest_timestamp(price_table)
        dates = self.calendar.select_dates(start_date, end_date)

        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'{date}')
                ids = self.stock_ticker_selector.ticker(date)

                daily_ret = self.data_reader.weighted_return(date=date, ids=ids, weight_base=self.units_factor)
                index = pd.MultiIndex.from_tuples([(date, self.policy.ticker)], names=['DateTime', 'ID'])
                ret = pd.Series(daily_ret, index=index, name='收益率')

                self.db_interface.update_df(ret, self.table_name)
                pbar.update()


class IndexUpdater(object):
    def __init__(self, config_loc=None, db_interface: DBInterface = None):
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


@dataclass
class FactorPortfolioPolicy:
    """
    因子收益率

    :param name: 名称
    :param bins: 分层数
    :param weight: 权重，默认为 ``None`` （等权）
    :param stock_selection_policy: 股票选取范围
    :param factor: 因子
    :param factor_need_shift: 因子是否需要延迟一个周期以避免未来函数
    :param industry: 行业分类因子，默认为 ``None`` （不进行行业中性）
    :param start_date: 开始日期
    """
    name: str = None
    bins: Sequence[int] = None
    weight: Factor = None
    stock_selection_policy: utils.StockSelectionPolicy = None
    factor: FactorBase = None
    factor_need_shift: bool = False
    industry: IndustryFactor = None
    start_date: dt.datetime = None


class FactorPortfolio(FactorCompositor):
    def __init__(self, factor_portfolio_policy: FactorPortfolioPolicy, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.policy = factor_portfolio_policy
        self.stock_ticker_selector = StockTickerSelector(factor_portfolio_policy.stock_selection_policy, db_interface)
        self.factor_name = self.policy.factor.display_factor_name
        self.ret_name = self.data_reader.stock_return.display_factor_name
        self.industry_category = self.policy.industry.display_factor_name
        self.cap_name = self.policy.weight.display_factor_name

    def update(self):
        table_name = '因子分组收益率'
        identifying_ticker = f'{self.factor_name}_NN_G1inG5'
        start_date = self._check_db_timestamp(table_name, self.policy.start_date,
                                              column_condition=('ID', identifying_ticker))
        end_date = self.db_interface.get_latest_timestamp('股票日行情')
        dates = self.calendar.select_dates(start_date, end_date, inclusive=(False, True))

        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'更新{date}的{self.factor_name}的因子收益率')
                pre_date = self.calendar.offset(date, -1)
                ids = self.stock_ticker_selector.ticker(date)

                pct_return = self.data_reader.stock_return.get_data(start_date=pre_date, end_date=date, ids=ids)
                factor_date = pre_date if self.policy.factor_need_shift else date
                factor_data = self.policy.factor.get_data(dates=[factor_date])
                industry = self.policy.industry.get_data(dates=[date])
                cap = self.policy.weight.get_data(dates=[pre_date])

                storage = [pct_return.droplevel('DateTime'), factor_data.droplevel('DateTime'),
                           industry.droplevel('DateTime'), cap.droplevel('DateTime')]
                data = pd.concat(storage, axis=1).dropna()

                def split_group(x: pd.Series) -> pd.Series:
                    labels = [f'G{i + 1}inG{num_bin}' for i in range(num_bin)]
                    return pd.qcut(x.loc[:, self.factor_name], q=num_bin, labels=labels, duplicates='drop')

                def fill_index(res: pd.Series) -> pd.Series:
                    index_name = [f'{self.factor_name}_{i}{w}_{it}' for it in res.index]
                    index = pd.MultiIndex.from_product([[date], index_name])
                    res.index = index
                    return res

                storage = []
                i = 'I'
                # use industry info:
                for num_bin in self.policy.bins:
                    industry_data = data.copy()
                    group = industry_data.groupby(self.industry_category).apply(split_group)
                    group.name = 'group'
                    industry_data = pd.concat([industry_data, group.droplevel(self.industry_category)], axis=1)

                    # unweighted
                    w = 'N'
                    res = industry_data.groupby('group')[self.ret_name].mean()
                    storage.append(fill_index(res))

                    # cap weighted
                    w = 'W'
                    res = industry_data.groupby('group').apply(
                        lambda x: x.loc[:, self.ret_name].dot(x.loc[:, self.cap_name] / x.loc[:, self.cap_name].sum()))
                    storage.append(fill_index(res))

                i = 'N'
                # without industry
                for num_bin in self.policy.bins:
                    non_industry_info = data.copy()
                    non_industry_info['group'] = split_group(non_industry_info)

                    # unweighted
                    w = 'N'
                    res = non_industry_info.groupby('group')[self.ret_name].mean()
                    storage.append(fill_index(res))

                    # cap weighted
                    w = 'W'
                    res = non_industry_info.groupby('group').apply(
                        lambda x: x.loc[:, self.ret_name].dot(x.loc[:, self.cap_name] / x.loc[:, self.cap_name].sum()))
                    storage.append(fill_index(res))

                full_data = pd.concat(storage)
                full_data.index.names = ('DateTime', 'ID')
                full_data.name = '收益率'
                self.db_interface.insert_df(full_data, table_name)
                pbar.update()
