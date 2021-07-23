import datetime as dt

import pandas as pd

from .model import FinancialModel, ModelFactorCompositor
from ..database_interface import DBInterface
from ..tickers import StockTickerSelector
from ..utils import StockSelectionPolicy


class FamaFrench3FactorModel(FinancialModel):
    def __init__(self):
        """Fama French 3 factor model(1992)"""
        super().__init__('Fama French 3 factor model', ['FF3_SMB', 'FF3_HML'])

        self.stock_selection_policy = StockSelectionPolicy(ignore_negative_book_value_stock=True,
                                                           ignore_st=True, ignore_pause=True,
                                                           ignore_new_stock_period=244)
        self.hml_threshold = [0, 0.3, 0.7, 1]
        self.smb_threshold = [0, 0.5, 1]


class SMBandHMLCompositor(ModelFactorCompositor):
    def __init__(self, model: FamaFrench3FactorModel = None, db_interface: DBInterface = None):
        """Compute SMB and HML in Fama French 3 factor model"""
        model = model if model else FamaFrench3FactorModel()
        super().__init__(model, db_interface)

        self.start_date = dt.datetime(2007, 1, 4)
        self.ticker_selector = StockTickerSelector(model.stock_selection_policy, self.db_interface)

        self.cap = self.data_reader.stock_free_floating_market_cap
        self.bm = self.data_reader.bm
        self.returns = self.data_reader.stock_return

    def compute_factor_return(self, balance_date: dt.datetime, pre_date: dt.datetime, date: dt.datetime,
                              rebalance_marker: str, period_marker: str) -> pd.Series:
        def cap_weighted_return(x):
            return x[returns.name].dot(x[cap.name]) / x[cap.name].sum()

        # data
        tickers = self.ticker_selector.ticker(date)
        bm = self.bm.get_data(ids=tickers, dates=balance_date).droplevel('DateTime')
        cap = self.cap.get_data(ids=tickers, dates=balance_date).droplevel('DateTime')
        returns = self.returns.get_data(ids=tickers, dates=[pre_date, date]).droplevel('DateTime')
        df = pd.concat([returns, bm, cap], axis=1).dropna()

        # grouping
        df['G_SMB'] = pd.qcut(df[self.cap.name], self.model.smb_threshold, labels=['small', 'big'])
        df['G_HML'] = pd.qcut(df[self.bm.name], self.model.hml_threshold, labels=['low', 'mid', 'high'])
        rets = df.groupby(['G_SMB', 'G_HML']).apply(cap_weighted_return)
        tmp = rets.groupby('G_SMB').mean()
        smb = tmp.loc['small'] - tmp.loc['big']
        tmp = rets.groupby('G_HML').mean()
        hml = tmp.loc['high'] - tmp.loc['low']

        # formatting result
        factor_names = [f'{factor_name}_{rebalance_marker}{period_marker}' for factor_name in self.factor_names]
        index_date = pre_date if period_marker == 'M' else date
        index = pd.MultiIndex.from_product([[index_date], factor_names], names=('DateTime', 'ID'))
        factor_df = pd.Series([smb, hml], index=index, name='收益率')
        return factor_df
