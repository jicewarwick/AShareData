import datetime as dt

import pandas as pd

from ..ashare_data_reader import AShareDataReader
from ..config import get_db_interface
from ..database_interface import DBInterface
from ..model.model import FinancialModel
from ..factor import ContinuousFactor


class FundNAVAnalysis(object):
    def __init__(self, ticker: str, db_interface: DBInterface = None):
        self.db_interface = db_interface if db_interface else get_db_interface()
        self.data_reader = AShareDataReader(self.db_interface)
        self.ticker = ticker
        db_name = '场外基金净值' if ticker.endswith('OF') else '场内基金日行情'
        self.nav_data = ContinuousFactor(db_name, '单位净值').bind_params(ids=self.ticker)

    def compute_correlation(self, index_code: str, period: int = 60) -> float:
        index_return_factor = self.data_reader.get_index_return_factor(index_code)
        start_date = self.data_reader.calendar.offset(dt.date.today(), -period)
        nav_chg = self.nav_data.get_data(start_date=start_date).pct_change()
        index_ret = index_return_factor.get_data(start_date=start_date)
        corr = nav_chg.corr(index_ret)
        return corr

    def compute_exposure(self, model: FinancialModel, period: int = 60):
        pass

    def get_latest_published_portfolio_holding(self) -> pd.DataFrame:
        data = self.db_interface.read_table('公募基金持仓', ids=self.ticker)
        latest = data.loc[data.index.get_level_values('DateTime') == data.index.get_level_values('DateTime').max(), :]
        return latest.sort_values('占股票市值比', ascending=False)
