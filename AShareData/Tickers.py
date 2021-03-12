import datetime as dt
from functools import cached_property
from itertools import product
from typing import Dict, List, Sequence, Union

import pandas as pd

from . import DateUtils
from .config import get_db_interface
from .DBInterface import DBInterface
from .Factor import CompactFactor, CompactRecordFactor, IndustryFactor, OnTheRecordFactor
from .utils import StockSelectionPolicy, TickerSelector


class TickersBase(object):
    """证券代码基类"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        self.db_interface = db_interface if db_interface else get_db_interface()
        self.cache = None

    def all_ticker(self) -> List[str]:
        """ return ALL ticker for the asset class"""
        return sorted(self.cache.ID.unique().tolist())

    @DateUtils.dtlize_input_dates
    def ticker(self, date: DateUtils.DateType = None) -> List[str]:
        """ return tickers that are alive on `date`, `date` default to today"""
        if date is None:
            date = dt.datetime.today()
        stock_ticker_df = self.cache.loc[self.cache.DateTime <= date]
        tmp = stock_ticker_df.groupby('ID').tail(1)
        return sorted(tmp.loc[tmp['上市状态'] == 1, 'ID'].tolist())

    def list_date(self) -> Dict[str, dt.datetime]:
        """ return the list date of all tickers"""
        first_list_info = self.cache.groupby('ID').head(1)
        return dict(zip(first_list_info.ID, first_list_info.DateTime))

    def get_list_date(self, ticker: str):
        """ return the list date of a ticker"""
        info = self.cache.loc[self.cache.ID == ticker, :]
        return info.DateTime.iloc[0]

    def new_ticker(self, start_date: dt.datetime, end_date: dt.datetime = None) -> List[str]:
        if end_date is None:
            end_date = dt.datetime.today()
        if start_date is None:
            start_date = dt.datetime(1990, 12, 10)
        u_data = self.cache.loc[(start_date <= self.cache.DateTime) & (self.cache.DateTime <= end_date), :]
        tmp = u_data.groupby('ID').tail(1)
        return sorted(tmp.loc[tmp['上市状态'] == 1, 'ID'].tolist())


class DiscreteTickers(TickersBase):
    """细类证券代码基类"""

    def __init__(self, asset_type: str, db_interface: DBInterface = None) -> None:
        super().__init__(db_interface)
        self.cache = self.db_interface.read_table('证券代码', text_statement=f'证券类型="{asset_type}"').reset_index()


class StockTickers(DiscreteTickers):
    """股票代码"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('A股股票', db_interface)


class ConvertibleBondTickers(DiscreteTickers):
    """可转债代码"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('可转债', db_interface)


class FutureTickers(DiscreteTickers):
    """期货合约代码"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('期货', db_interface)


class ETFOptionTickers(DiscreteTickers):
    """期权合约代码"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('ETF期权', db_interface)


class IndexOptionTickers(DiscreteTickers):
    """指数期权合约代码"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('指数期权', db_interface)


class FutureOptionTickers(DiscreteTickers):
    """商品期权合约代码"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('商品期权', db_interface)


class ExchangeStockETFTickers(DiscreteTickers):
    """场内股票ETF基金代码"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('股票型ETF基金', db_interface)


class BondETFTickers(DiscreteTickers):
    """债券ETF基金代码"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('债券型ETF基金', db_interface)


class ConglomerateTickers(TickersBase):
    """聚合类证券代码基类"""

    def __init__(self, sql_statement: str, db_interface: DBInterface = None) -> None:
        super().__init__(db_interface)
        self.cache = self.db_interface.read_table('证券代码', text_statement=sql_statement).reset_index()


class OptionTickers(ConglomerateTickers):
    """期权"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('证券类型 like "%期权"', db_interface)


class FundTickers(ConglomerateTickers):
    """基金"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('证券类型 like "%基金"', db_interface)


class ETFTickers(ConglomerateTickers):
    """ETF"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('证券类型 like "%ETF%基金"', db_interface)


class ExchangeFundTickers(ConglomerateTickers):
    """场内基金"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('证券类型 like "%基金" AND 证券类型 not like "%场外基金"', db_interface)


class OTCFundTickers(ConglomerateTickers):
    """场外基金"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('证券类型 like "%场外基金"', db_interface)


class StockFundTickers(ConglomerateTickers):
    """股票型基金"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('证券类型 like "股票型%基金"', db_interface)


class StockOTCFundTickers(ConglomerateTickers):
    """股票型场外基金"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('证券类型 like "股票型%场外基金"', db_interface)


class InvestmentStyleFundTickers(ConglomerateTickers):
    def __init__(self, asset_type_statement: str, style_statement: str, db_interface: DBInterface = None) -> None:
        super().__init__(asset_type_statement, db_interface)
        support_data = self.db_interface.read_table('基金列表', '投资风格', text_statement=style_statement)
        self.cache = self.cache.loc[self.cache.ID.isin(support_data.index)]


class EnhancedIndexFund(InvestmentStyleFundTickers):
    """指数增强场外基金"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('证券类型 like "股票型%场外基金"', '投资风格 = "增强指数型"', db_interface)


class IndexFund(InvestmentStyleFundTickers):
    """指数场外基金"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('证券类型 like "股票型%场外基金"', '投资风格 = "被动指数型"', db_interface)


class ActiveManagedOTCStockFundTickers(InvestmentStyleFundTickers):
    """主动管理型场外基金"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('(证券类型 like "股票型%场外基金") OR (证券类型 like "混合型%场外基金")', '(投资风格 != "被动指数型") AND (投资风格 != "增强指数型")',
                         db_interface)


class StockTickerSelector(TickerSelector):
    """股票代码选择器"""

    def __init__(self, policy: StockSelectionPolicy, db_interface: DBInterface = None) -> None:
        """
        :param db_interface: BDInterface
        :param policy: 选股条件
        """
        super().__init__()
        self.db_interface = db_interface if db_interface else get_db_interface()
        self.stock_ticker = StockTickers(self.db_interface)
        self.policy = policy

    @cached_property
    def calendar(self):
        return DateUtils.TradingCalendar(self.db_interface)

    @cached_property
    def paused_stock_selector(self):
        return OnTheRecordFactor('股票停牌', self.db_interface)

    @cached_property
    def const_limit_selector(self):
        return OnTheRecordFactor('一字涨跌停', self.db_interface)

    @cached_property
    def risk_warned_stock_selector(self):
        tmp = CompactFactor('证券名称', self.db_interface)
        ids = tmp.data.index.get_level_values('ID')
        tmp.data = tmp.data.loc[ids.str.endswith('.SH') | ids.str.endswith('.SZ')]
        tmp.data = tmp.data.map(lambda x: 'PT' in x or 'ST' in x or '退' in x)
        return CompactRecordFactor(tmp, '风险警示股')

    @cached_property
    def industry_info(self):
        if self.policy.industry:
            return IndustryFactor(self.policy.industry_provider, self.policy.industry_level, self.db_interface)

    @DateUtils.dtlize_input_dates
    def ticker(self, date: DateUtils.DateType, ids: Sequence[str] = None) -> List[str]:
        """ select stocks that matched selection policy on `date`(amongst `ids`)

        :param date: query date
        :param ids: tickers to select from
        :return: list of ticker that satisfy the stock selection policy
        """
        if ids is None:
            ids = set(self.stock_ticker.ticker(date))

        if self.policy.ignore_new_stock_period or self.policy.select_new_stock_period:
            start_date, end_date = None, None
            if self.policy.ignore_new_stock_period:
                end_date = self.calendar.offset(date, -self.policy.ignore_new_stock_period)
            if self.policy.select_new_stock_period:
                start_date = self.calendar.offset(date, -self.policy.select_new_stock_period - 1)
            ids = set(self.stock_ticker.new_ticker(start_date=start_date, end_date=end_date)) & ids

        if self.industry_info and self.policy.industry:
            ids = ids & set(self.industry_info.list_constitutes(date=date, industry=self.policy.industry))
        if self.policy.ignore_const_limit:
            ids = ids - set(self.const_limit_selector.get_data(date))

        if self.policy.ignore_pause:
            ids = ids - set(self.paused_stock_selector.get_data(date))
        elif self.policy.select_pause:
            ids = ids & set(self.paused_stock_selector.get_data(date))
        if self.policy.max_pause_days:
            pause_days, period_length = self.policy.max_pause_days
            start_date = self.calendar.offset(date, -period_length)
            end_date = self.calendar.offset(date, -1)
            pause_counts = self.paused_stock_selector.get_counts(start_date=start_date, end_date=end_date)
            pause_counts = pause_counts.loc[pause_counts > pause_days]
            ids = ids - set(pause_counts.index.get_level_values('ID'))

        if self.policy.select_st:
            ids = ids & set(self.risk_warned_stock_selector.get_data(date))
            if self.policy.st_defer_period:
                start_date = self.calendar.offset(date, -self.policy.st_defer_period - 1)
                ids = ids & set(self.risk_warned_stock_selector.get_data(start_date))
        if self.policy.ignore_st:
            ids = ids - set(self.risk_warned_stock_selector.get_data(date))

        ids = sorted(list(ids))
        return ids

    def generate_index(self, start_date: DateUtils.DateType = None, end_date: DateUtils.DateType = None,
                       dates: Union[DateUtils.DateType, Sequence[DateUtils.DateType]] = None) -> pd.MultiIndex:
        storage = []
        if dates is None:
            dates = self.calendar.select_dates(start_date, end_date)
        for date in dates:
            ids = self.ticker(date)
            storage.extend(list(product([date], ids)))
        return pd.MultiIndex.from_tuples(storage, names=['DateTime', 'ID'])
