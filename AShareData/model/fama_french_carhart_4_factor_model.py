import datetime as dt

import pandas as pd

from .model import FinancialModel, ModelFactorCompositor
from ..database_interface import DBInterface
from ..tickers import StockTickerSelector
from ..utils import StockSelectionPolicy


class FamaFrenchCarhart4FactorModel(FinancialModel):
    def __init__(self):
        """Fama French Carhart 4 factor model(1997)"""
        super().__init__('Fama French Carhart 4 factor model', ['FF3_SMB', 'FF3_HML', 'FFC4_UMD'])

        self.stock_selection_policy = StockSelectionPolicy(ignore_negative_book_value_stock=True,
                                                           ignore_st=True, ignore_pause=True,
                                                           ignore_new_stock_period=244)
        self.hml_threshold = [0, 0.3, 0.7, 1]
        self.smb_threshold = [0, 0.5, 1]
        self.umd_threshold = [0, 0.3, 0.7, 1]
        self.offset_1 = 22
        self.offset_2 = 22*12


class UMDCompositor(ModelFactorCompositor):
    def __init__(self, model: FamaFrenchCarhart4FactorModel = None, db_interface: DBInterface = None):
        """Compute UMD/MOM in Fama French Carhart 4 factor model"""
        model = model if model else FamaFrenchCarhart4FactorModel()
        super().__init__(model, db_interface)
        self.factor_names = ['Carhart_UMD']

        self.start_date = dt.datetime(2007, 1, 4)
        self.ticker_selector = StockTickerSelector(model.stock_selection_policy, self.db_interface)

        self.cap = self.data_reader.stock_free_floating_market_cap
        self.returns = self.data_reader.stock_return

    def compute_factor_return(self, balance_date: dt.datetime, pre_date: dt.datetime, date: dt.datetime,
                              rebalance_marker: str, period_marker: str) -> pd.Series:
        # data
        tm1 = self.calendar.offset(balance_date, -self.model.offset_1)
        tm12 = self.calendar.offset(balance_date, -self.model.offset_2)
        tickers = self.ticker_selector.ticker(date)
        tm1_ticker = self.ticker_selector.ticker(tm1)
        tm12_ticker = self.ticker_selector.ticker(tm12)
        tickers = sorted(list(set(tickers) & set(tm1_ticker) & set(tm12_ticker)))
        p1 = self.data_reader.hfq_close.get_data(ids=tickers, dates=tm1)
        p12 = self.data_reader.hfq_close.get_data(ids=tickers, dates=tm12)
        pct_diff = p12.droplevel('DateTime') / p1.droplevel('DateTime')
        cap = self.cap.get_data(ids=tickers, dates=balance_date).droplevel('DateTime')
        returns = self.returns.get_data(ids=tickers, dates=[pre_date, date]).droplevel('DateTime')
        df = pd.concat([returns, cap, pct_diff], axis=1).dropna()

        # grouping
        df['G_SMB'] = pd.qcut(df[cap.name], self.model.smb_threshold, labels=['small', 'big'])
        df['G_UMD'] = pd.qcut(df[pct_diff.name], self.model.umd_threshold, labels=['up', 'mid', 'down'])
        rets = df.groupby(['G_SMB', 'G_UMD'])[returns.name].mean()
        tmp = rets.groupby('G_UMD').mean()
        umd = tmp.loc['up'] - tmp.loc['down']

        # formatting result
        factor_names = [f'{factor_name}_{rebalance_marker}{period_marker}' for factor_name in self.factor_names]
        index_date = pre_date if period_marker == 'M' else date
        index = pd.MultiIndex.from_product([[index_date], factor_names], names=('DateTime', 'ID'))
        factor_df = pd.Series(umd, index=index, name='收益率')
        return factor_df
