import pandas as pd
from tqdm import tqdm

from .factor_compositor import FactorCompositor
from .. import utils
from ..database_interface import DBInterface
from ..factor import CompactFactor
from ..tickers import StockTickerSelector


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
