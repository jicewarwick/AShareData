import datetime as dt
from typing import List

import pandas as pd
from tqdm import tqdm

from ..database_interface import DBInterface
from ..factor_compositor.factor_compositor import FactorCompositor


class FinancialModel(object):
    def __init__(self, model_name: str, factor_names: List[str]):
        """Base class for Financial Models

        Should include model parameters in the object, accompanied by a subclass of ModelFactorCompositor to compute factor returns

        PS: excess market return is implied and do not be specified in the ``factor_names``

        :param model_name: Financial model Name
        :param factor_names: Factor names specified in the model
        """
        self.model_name = model_name
        self.factor_names = factor_names.copy()

    def get_db_factor_names(self, rebalance_schedule: str = 'D', computing_schedule: str = 'D'):
        """ Naming schemes used when calculating using different rebalancing schedule and computing schedule. Combination of ('D', 'D'), ('M', 'D'), ('M', 'M') are valid

        :param rebalance_schedule: 'D' or 'M', portfolio is rebalanced Daily('D') or at the end of each Month('M')
        :param computing_schedule: 'D' or 'M', portfolio return is computed Daily('D') or Monthly('M')
        :return:
        """
        return [f'{it}_{rebalance_schedule}{computing_schedule}' for it in self.factor_names]


class ModelFactorCompositor(FactorCompositor):
    def __init__(self, model, db_interface: DBInterface):
        """ Model Factor Return Compositor

        Compute factor returns specified by ``model``

        :param model: Financial model
        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.model = model
        self.factor_names = model.factor_names
        self.db_table_name = '模型因子收益率'
        self.start_date = None

    def update(self):
        self.update_daily_rebalanced_portfolio()
        self.update_monthly_rebalanced_portfolio_return()

    def update_monthly_rebalanced_portfolio_return(self):
        eg_factor_name = f'{self.factor_names[-1]}_MD'
        start_date = self.db_interface.get_latest_timestamp(self.db_table_name, self.start_date,
                                                            column_condition=('ID', eg_factor_name))
        end_date = self.db_interface.get_latest_timestamp('股票日行情')
        dates = self.data_reader.calendar.select_dates(start_date, end_date, inclusive=(False, True))

        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'更新 {self.model.model_name} 因子收益率: {date}')
                rebalance_date = self.calendar.pre_month_end(date.year, date.month)

                pre_date = self.data_reader.calendar.offset(date, -1)
                factor_df = self.compute_factor_return(rebalance_date, pre_date, date, 'M', 'D')
                self.db_interface.insert_df(factor_df, self.db_table_name)

                next_date = self.data_reader.calendar.offset(date, -1)
                if next_date.month != date.month:
                    factor_df = self.compute_factor_return(rebalance_date, rebalance_date, date, 'M', 'M')
                    month_beg_date = self.calendar.month_begin(date.year, date.month)
                    factor_df.index = pd.MultiIndex.from_product(
                        [[month_beg_date], factor_df.index.get_level_values('ID')], names=('DateTime', 'ID'))
                    self.db_interface.insert_df(factor_df, self.db_table_name)
                pbar.update()

    def update_daily_rebalanced_portfolio(self):
        eg_factor_name = f'{self.factor_names[-1]}_DD'
        start_date = self.db_interface.get_latest_timestamp(self.db_table_name, self.start_date,
                                                            column_condition=('ID', eg_factor_name))
        end_date = self.db_interface.get_latest_timestamp('股票日行情')
        dates = self.data_reader.calendar.select_dates(start_date, end_date, inclusive=(False, True))

        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'更新 {self.model.model_name} 因子日收益率: {date}')
                pre_date = self.data_reader.calendar.offset(date, -1)
                factor_df = self.compute_factor_return(pre_date, pre_date, date, 'D', 'D')
                self.db_interface.insert_df(factor_df, self.db_table_name)
                pbar.update()

    def compute_factor_return(self, balance_date: dt.datetime, pre_date: dt.datetime, date: dt.datetime,
                              rebalance_marker: str, period_marker: str) -> pd.Series:
        raise NotImplementedError()
