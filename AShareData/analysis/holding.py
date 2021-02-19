import datetime as dt

import pandas as pd

from .. import utils
from ..AShareDataReader import AShareDataReader
from ..config import get_db_interface
from ..DBInterface import DBInterface


class IndustryComparison(object):
    def __init__(self, index: str, industry_provider: str, industry_level: int, db_interface: DBInterface = None):
        if db_interface is None:
            db_interface = get_db_interface()
        self.data_reader = AShareDataReader(db_interface)
        self.industry_info = self.data_reader.industry(industry_provider, industry_level)
        self.index = index

    def holding_comparison(self, holding: pd.Series):
        holding_ratio = self.portfolio_weight(holding)
        return self.industry_ratio_comparison(holding_ratio)

    def industry_ratio_comparison(self, holding_ratio: pd.Series):
        date = holding_ratio.index.get_level_values('DateTime').unique()[0]

        industry_info = self.industry_info.get_data(dates=date)
        index_comp = self.data_reader.index_constitute.get_data(index_ticker=self.index, date=date)

        holding_industry = self._industry_ratio(holding_ratio, industry_info) * 100
        index_industry = self._industry_ratio(index_comp, industry_info)

        diff_df = pd.concat([holding_industry, index_industry], axis=1, sort=True).fillna(0)

        return diff_df.iloc[:, 0] - diff_df.iloc[:, 1]

    def portfolio_weight(self, holding: pd.Series):
        date = holding.index.get_level_values('DateTime').unique()[0]

        price_info = self.data_reader.stock_close.get_data(dates=date)
        price_info.name = 'close'
        tmp = pd.concat([holding, price_info], axis=1).dropna()
        cap = tmp['quantity'] * tmp['close']
        ratio = cap / cap.sum()
        ratio.name = 'weight'
        return ratio

    @staticmethod
    def _industry_ratio(ratio: pd.Series, industry_info: pd.Series):
        tmp = pd.concat([ratio, industry_info], axis=1).dropna()
        return tmp.groupby(industry_info.name).sum().iloc[:, 0]

    @staticmethod
    def import_holding(holding_loc, date: dt.datetime):
        holding = pd.read_excel(holding_loc).rename({'证券代码': 'ID', '数量': 'quantity'}, axis=1)
        holding['ID'] = holding.ID.apply(utils.format_stock_ticker)
        holding['DateTime'] = date
        holding.set_index(['DateTime', 'ID'], inplace=True)
        return holding
