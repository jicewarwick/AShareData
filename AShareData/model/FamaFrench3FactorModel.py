import datetime as dt
from typing import Sequence, Union

import pandas as pd
from tqdm import tqdm

from .model import FinancialModel
from ..AShareDataReader import AShareDataReader
from ..config import get_db_interface
from ..DBInterface import DBInterface
from ..factor_compositor import FactorCompositor
from ..Tickers import StockTickerSelector
from ..utils import StockSelectionPolicy


class FamaFrench3FactorModel(FinancialModel):
    def __init__(self, db_interface: DBInterface = None):
        super().__init__('Fama French 3 factor model', ['FF3_RM', 'FF3_SMB', 'FF3_HML'])
        self.db_interface = db_interface if db_interface else get_db_interface()

        self.data_reader = AShareDataReader(self.db_interface)
        self.cap = self.data_reader.stock_free_floating_market_cap
        self.bm = self.data_reader.bm
        self.returns = self.data_reader.stock_return

        policy = StockSelectionPolicy(ignore_negative_book_value_stock=True, ignore_st=True, ignore_pause=True,
                                      ignore_new_stock_period=244)
        self.ticker_selector = StockTickerSelector(policy, self.db_interface)
        self.start_date = dt.datetime(2010, 1, 3)
        self.hml_threshold = [0, 0.3, 0.7, 1]
        self.smb_threshold = [0, 0.5, 1]
        self.db_table_name = '模型因子日收益率'

    def get_factor_return(self, dates: Union[dt.datetime, Sequence[dt.datetime]] = None):
        return self.db_interface.read_table(self.db_table_name, dates=dates, ids=self.factor_names)

    def compute_factor_loading(self, dates: dt.datetime, ids: str):
        pass


class FamaFrench3FactorCompositor(FactorCompositor):
    def __init__(self, model: FamaFrench3FactorModel):
        super().__init__(model.db_interface)
        self.model = model

    def update(self):
        pass

    def update_daily(self):
        pass

    def update_monthly(self):
        pass

    def update_daily_factor_return(self):
        start_date = self.db_interface.get_latest_timestamp(self.model.db_table_name, self.model.start_date,
                                                            column_condition=('ID', self.model.factor_names[0]))
        end_date = self.db_interface.get_latest_timestamp('股票日行情')
        dates = self.data_reader.calendar.select_dates(start_date, end_date, inclusive=(False, True))

        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'更新 {self.model.model_name} 因子日收益率: {date}')
                factor_df = self.compute_daily_factor_return(date)
                self.db_interface.insert_df(factor_df, self.model.db_table_name)
                pbar.update()

    def compute_daily_factor_return(self, date) -> pd.Series:
        def cap_weighted_return(x):
            return x[returns.name].dot(x[cap.name]) / x[cap.name].sum()

        pre_date = self.data_reader.calendar.offset(date, -1)
        tickers = self.model.ticker_selector.ticker(date)
        bm = self.model.bm.get_data(ids=tickers, dates=pre_date)
        cap = self.model.cap.get_data(ids=tickers, dates=pre_date)
        returns = self.model.returns.get_data(ids=tickers, dates=[pre_date, date])
        df = pd.concat([returns, bm, cap], axis=1).dropna()
        df['G_SMB'] = pd.qcut(df[self.model.cap.name], self.model.smb_threshold, labels=['small', 'big'])
        df['G_HML'] = pd.qcut(df[self.model.bm.name], self.model.hml_threshold, labels=['low', 'mid', 'high'])
        rets = df.groupby(['G_SMB', 'G_HML']).apply(cap_weighted_return)
        tmp = rets.groupby('G_SMB').mean()
        smb = tmp.loc['small'] - tmp.loc['big']
        tmp = rets.groupby('G_HML').mean()
        hml = tmp.loc['high'] - tmp.loc['low']
        market_return = cap_weighted_return(df)
        index = pd.MultiIndex.from_product([[date], self.model.factor_names], names=('DateTime', 'ID'))
        factor_df = pd.Series([market_return, smb, hml], index=index, name='收益率')
        return factor_df
