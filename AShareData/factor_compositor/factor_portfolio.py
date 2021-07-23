import datetime as dt
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from tqdm import tqdm

from .factor_compositor import FactorCompositor
from .. import utils
from ..database_interface import DBInterface
from ..factor import Factor, FactorBase, IndustryFactor
from ..tickers import StockTickerSelector


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
        self.factor_name = self.policy.factor.name
        self.ret_name = self.data_reader.stock_return.name
        self.industry_category = self.policy.industry.name
        self.cap_name = self.policy.weight.name

    def update(self):
        table_name = '因子分组收益率'
        identifying_ticker = f'{self.factor_name}_NN_G1inG5'
        start_date = self.db_interface.get_latest_timestamp(table_name, self.policy.start_date,
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
                factor_data = self.policy.factor.get_data(dates=factor_date)
                industry = self.policy.industry.get_data(dates=date)
                cap = self.policy.weight.get_data(dates=pre_date)

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
