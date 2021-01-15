import datetime as dt

import pandas as pd

from .. import AShareDataReader, DateUtils, DBInterface, utils
from ..config import get_db_interface


class IndustryComparison(object):
    def __init__(self, index: str, industry_provider: str, industry_level: int, db_interface: DBInterface = None):
        if not db_interface:
            db_interface = get_db_interface()
        self.data_reader = AShareDataReader(db_interface)
        self.industry_info = self.data_reader.industry(industry_provider, industry_level)
        self.index = index

    def holding_comparison(self, holding: pd.Series):
        holding_ratio = self._holding_to_ratio(holding)
        return self.industry_ratio_comparison(holding_ratio)

    def industry_ratio_comparison(self, holding_ratio: pd.Series):
        date = holding_ratio.index.get_level_values('DateTime').unique()[0]

        industry_info = self.industry_info.get_data(dates=date).stack()
        industry_info.name = 'industry'

        index_comp = self.data_reader.index_constitute.get_data(index_ticker=self.index, date=date)

        holding_industry = self._industry_ratio(holding_ratio, industry_info) * 100
        index_industry = self._industry_ratio(index_comp, industry_info)

        diff_df = pd.concat([holding_industry, index_industry], axis=1, sort=True).fillna(0)

        return diff_df.iloc[:, 0] - diff_df.iloc[:, 1]

    def _holding_to_ratio(self, holding: pd.Series):
        date = holding.index.get_level_values('DateTime').unique()[0]

        price_info = self.data_reader.stock_close.get_data(dates=[date]).stack()
        price_info.name = 'close'
        tmp = holding.join(price_info, how='inner')
        cap = tmp['quantity'] * tmp['close']
        ratio = cap / cap.sum()
        ratio.name = 'ratio'
        return ratio

    @staticmethod
    def _industry_ratio(ratio: pd.Series, industry_info: pd.Series):
        tmp = pd.concat([ratio, industry_info], join='inner', axis=1)
        return tmp.groupby('industry').sum().iloc[:, 0]

    @staticmethod
    def import_holding(holding_loc, date: dt.datetime):
        holding = pd.read_excel(holding_loc).rename({'证券代码': 'ID', '数量': 'quantity'}, axis=1)
        holding['ID'] = holding.ID.apply(utils.format_stock_ticker)
        holding['DateTime'] = date
        holding.set_index(['DateTime', 'ID'], inplace=True)
        return holding


class BetaComputer(object):
    def __init__(self, benchmark: str, db_interface: DBInterface = None):
        if not db_interface:
            db_interface = get_db_interface()
        self.data_reader = AShareDataReader(db_interface)
        self.benchmark = benchmark

    def historical_method(self, ticker: str, date: DateUtils.DateType, duration: int):
        pass
