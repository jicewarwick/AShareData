import datetime as dt
from functools import cached_property
from typing import Dict, List, Sequence

from . import DateUtils
from .Factor import CompactFactor, CompactRecordFactor, OnTheRecordFactor
from .utils import StockSelectionPolicy

from . import utils
from .DBInterface import DBInterface
from .Factor import IndustryFactor


class Tickers(object):
    def __init__(self, db_interface: DBInterface):
        self.db_interface = db_interface

    def all_ticker(self) -> List[str]:
        """ return ALL ticker for the asset class"""
        raise NotImplementedError()

    def ticker(self, date: DateUtils.DateType = dt.date.today()) -> List[str]:
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

    @DateUtils.format_input_dates
    def ticker(self, date: DateUtils.DateType = dt.date.today()) -> List[str]:
        """Get stocks still listed at ``date``"""
        stock_ticker_df = self.cache.loc[self.cache.DateTime <= date]
        tmp = stock_ticker_df.groupby('ID').tail(1)
        return sorted(tmp.loc[tmp['上市状态'] == 1, 'ID'].tolist())

    def list_date(self) -> Dict[str, dt.datetime]:
        first_list_info = self.cache.groupby('ID').head(1)
        return dict(zip(first_list_info.ID, first_list_info.DateTime))


class FutureTickers(Tickers):
    def __init__(self, db_interface: DBInterface):
        super().__init__(db_interface)
        self.cache = db_interface.read_table('期货合约', ['合约上市日期', '最后交易日']).reset_index()

    def all_ticker(self) -> List[str]:
        return self.cache.ID.tolist()

    @DateUtils.format_input_dates
    def ticker(self, date: DateUtils.DateType = dt.date.today()) -> List[str]:
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

    @DateUtils.format_input_dates
    def ticker(self, date: DateUtils.DateType = dt.date.today()) -> List[str]:
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

    @DateUtils.format_input_dates
    def ticker(self, date: DateUtils.DateType = dt.date.today()) -> List[str]:
        ticker_df = self.cache.loc[(self.cache['DateTime'] <= date) & (self.cache['DateTime'] > date), :]
        return sorted(ticker_df.ID.tolist())

    def list_date(self) -> Dict[str, dt.datetime]:
        return dict(zip(self.cache.ID, self.cache.DateTime))


class StockTickerSelector(object):
    def __init__(self, db_interface: DBInterface, policy: StockSelectionPolicy):
        super().__init__()
        self.db_interface = db_interface
        self.stock_ticker = StockTickers(self.db_interface)
        self.policy = policy

    @cached_property
    def calendar(self):
        return DateUtils.TradingCalendar(self.db_interface)

    @cached_property
    def paused_stock_selector(self):
        return OnTheRecordFactor(self.db_interface, '股票停牌')

    @cached_property
    def const_limit_selector(self):
        return OnTheRecordFactor(self.db_interface, '一字涨跌停')

    @cached_property
    def risk_warned_stock_selector(self):
        tmp = CompactFactor(self.db_interface, '证券名称')
        tmp.data = tmp.data.map(lambda x: 'PT' in x or 'ST' in x or '退' in x)
        return CompactRecordFactor(tmp, '风险警示股')

    @cached_property
    def industry_info(self):
        if self.policy.industry:
            return IndustryFactor(self.db_interface, self.policy.industry_provider, self.policy.industry_level)

    @DateUtils.format_input_dates
    def ticker(self, date: DateUtils.DateType, ids: Sequence[str] = None) -> List[str]:
        if not ids:
            ids = set(self.stock_ticker.ticker(date))
        if self.policy.ignore_new_stock_period:
            start_ref_date = self.calendar.offset(date, -self.policy.ignore_new_stock_period.days)
            ids = set(self.stock_ticker.ticker(start_ref_date)) & ids
        if self.industry_info and self.policy.industry:
            ids = ids & set(self.industry_info.list_constitutes(self.policy.industry))
        if self.policy.ignore_const_limit:
            ids = ids - set(self.const_limit_selector.get_data(date))

        if self.policy.ignore_pause:
            ids = ids - set(self.paused_stock_selector.get_data(date))
        elif self.policy.select_pause:
            ids = ids & set(self.paused_stock_selector.get_data(date))

        if self.policy.select_st:
            ids = ids & set(self.risk_warned_stock_selector.get_data(date))
        elif self.policy.ignore_st:
            ids = ids - set(self.risk_warned_stock_selector.get_data(date=date))

        ids = list(ids)
        return ids
