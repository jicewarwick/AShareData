import datetime as dt
from typing import Dict, List

from . import utils
from .DBInterface import DBInterface


class Tickers(object):
    def __init__(self, db_interface: DBInterface):
        self.db_interface = db_interface

    def all_ticker(self) -> List[str]:
        """ return ALL ticker for the asset class"""
        raise NotImplementedError()

    def ticker(self, date: utils.DateType = dt.date.today()) -> List[str]:
        """ return tickers that are alive on `date`"""
        raise NotImplementedError()

    def list_date(self) -> Dict[str, dt.datetime]:
        """ return the list date of all tickers"""
        raise NotImplementedError()


class StockTickers(Tickers):
    def __init__(self, db_interface: DBInterface):
        super().__init__(db_interface)
        self.cache = db_interface.read_table('股票上市退市').reset_index()

    def all_ticker(self) -> List[str]:
        return sorted(self.cache.ID.unique().tolist())

    @utils.format_input_dates
    def ticker(self, date: utils.DateType = dt.date.today()) -> List[str]:
        """Get stocks still listed at ``date``"""
        stock_ticker_df = self.cache.loc[self.cache.DateTime <= date]
        tmp = stock_ticker_df.groupby('ID').tail(1)
        return sorted(tmp.loc[tmp['上市状态'] == 1, 'ID'].tolist())

    def list_date(self) -> Dict[str, dt.datetime]:
        first_list_info = self.cache.groupby('ID').head(1)
        return dict(zip(first_list_info.ID, first_list_info.DateTime))


class StockIndustryTicker(Tickers):
    def __init__(self, db_interface: DBInterface, provider: str):
        super().__init__(db_interface)
        self.provider = provider

    def all_ticker(self):
        pass

    @utils.format_input_dates
    def ticker(self, date: utils.DateType = dt.date.today()) -> List[str]:
        pass

    def list_date(self) -> Dict[str, dt.datetime]:
        pass


class FutureTickers(Tickers):
    def __init__(self, db_interface: DBInterface):
        super().__init__(db_interface)
        self.cache = db_interface.read_table('期货合约', ['合约上市日期', '最后交易日']).reset_index()

    def all_ticker(self) -> List[str]:
        return self.cache.ID.tolist()

    @utils.format_input_dates
    def ticker(self, date: utils.DateType = dt.date.today()) -> List[str]:
        ticker_df = self.cache.loc[(self.cache['合约上市日期'] <= date) & (self.cache['最后交易日'] >= date), :]
        return sorted(ticker_df.ID.tolist())

    def list_date(self) -> Dict[str, dt.datetime]:
        return dict(zip(self.cache.ID, self.cache['合约上市日期']))


class OptionTickers(Tickers):
    def __init__(self, db_interface: DBInterface):
        super().__init__(db_interface)
        self.cache = db_interface.read_table('期权合约', ['上市日期', '行权日期']).reset_index()

    def all_ticker(self) -> List[str]:
        return self.cache.ID.tolist()

    @utils.format_input_dates
    def ticker(self, date: utils.DateType = dt.date.today()) -> List[str]:
        ticker_df = self.cache.loc[(self.cache['上市日期'] <= date) & (self.cache['行权日期'] > date), :]
        return sorted(ticker_df.ID.tolist())

    def list_date(self) -> Dict[str, dt.datetime]:
        return dict(zip(self.cache.ID, self.cache['上市日期']))


class ETFTickers(Tickers):
    def __init__(self, db_interface: DBInterface):
        super().__init__(db_interface)
        self.cache = db_interface.read_table('etf上市日期').reset_index()

    def all_ticker(self) -> List[str]:
        return self.cache.ID.tolist()

    @utils.format_input_dates
    def ticker(self, date: utils.DateType = dt.date.today()) -> List[str]:
        ticker_df = self.cache.loc[(self.cache['DateTime'] <= date) & (self.cache['DateTime'] > date), :]
        return sorted(ticker_df.ID.tolist())

    def list_date(self) -> Dict[str, dt.datetime]:
        return dict(zip(self.cache.ID, self.cache.DateTime))
