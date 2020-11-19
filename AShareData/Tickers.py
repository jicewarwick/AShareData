import datetime as dt
from typing import Dict, List, Sequence

from cached_property import cached_property

from . import DateUtils
from .DBInterface import DBInterface
from .Factor import CompactFactor, CompactRecordFactor, IndustryFactor, OnTheRecordFactor
from .utils import StockSelectionPolicy


class TickersBase(object):
    """证券代码基类"""

    def __init__(self, db_interface: DBInterface) -> None:
        self.db_interface = db_interface
        self.cache = None

    def all_ticker(self) -> List[str]:
        """ return ALL ticker for the asset class"""
        return sorted(self.cache.ID.unique().tolist())

    @DateUtils.dtlize_input_dates
    def ticker(self, date: DateUtils.DateType = dt.datetime.today()) -> List[str]:
        """ return tickers that are alive on `date`"""
        stock_ticker_df = self.cache.loc[self.cache.DateTime <= date]
        tmp = stock_ticker_df.groupby('ID').tail(1)
        return sorted(tmp.loc[tmp['上市状态'] == 1, 'ID'].tolist())

    def list_date(self) -> Dict[str, dt.datetime]:
        """ return the list date of all tickers"""
        first_list_info = self.cache.groupby('ID').head(1)
        return dict(zip(first_list_info.ID, first_list_info.DateTime))


class DiscreteTickers(TickersBase):
    """细类证券代码基类"""

    def __init__(self, db_interface: DBInterface, asset_type: str) -> None:
        super().__init__(db_interface)
        self.cache = db_interface.read_table('证券代码', text_statement=f'证券类型="{asset_type}"').reset_index()


class StockTickers(DiscreteTickers):
    """股票代码"""

    def __init__(self, db_interface: DBInterface) -> None:
        super().__init__(db_interface, 'A股股票')


class FutureTickers(DiscreteTickers):
    """期货合约代码"""

    def __init__(self, db_interface: DBInterface) -> None:
        super().__init__(db_interface, '期货')


class OptionTickers(DiscreteTickers):
    """期权合约代码"""

    def __init__(self, db_interface: DBInterface, asset_type: str) -> None:
        super().__init__(db_interface, asset_type)


class ETFOptionTickers(OptionTickers):
    """期权合约代码"""

    def __init__(self, db_interface: DBInterface) -> None:
        super().__init__(db_interface, 'ETF期权')


class IndexOptionTickers(DiscreteTickers):
    """指数期权合约代码"""

    def __init__(self, db_interface: DBInterface) -> None:
        super().__init__(db_interface, '指数期权')


class FutureOptionTickers(DiscreteTickers):
    """商品期权合约代码"""

    def __init__(self, db_interface: DBInterface) -> None:
        super().__init__(db_interface, '商品期权')


class StockETFTickers(DiscreteTickers):
    """股票ETF基金代码"""

    def __init__(self, db_interface: DBInterface) -> None:
        super().__init__(db_interface, '股票ETF基金')


class BondETFTickers(DiscreteTickers):
    """债券ETF基金代码"""

    def __init__(self, db_interface: DBInterface) -> None:
        super().__init__(db_interface, '债券ETF基金')


class ConglomerateTickers(TickersBase):
    """聚合类证券代码基类"""

    def __init__(self, db_interface: DBInterface, criteria: str) -> None:
        super().__init__(db_interface)
        self.cache = db_interface.read_table('证券代码', text_statement=f'证券类型 like "{criteria}"').reset_index()


class ETFTickers(ConglomerateTickers):
    """ETF"""

    def __init__(self, db_interface: DBInterface) -> None:
        super().__init__(db_interface, '%ETF基金')


class OptionTickers(ConglomerateTickers):
    """期权"""

    def __init__(self, db_interface: DBInterface) -> None:
        super().__init__(db_interface, '%期权')


class StockTickerSelector(object):
    """股票代码选择器"""

    def __init__(self, db_interface: DBInterface, policy: StockSelectionPolicy) -> None:
        """
        :param db_interface: BDInterface
        :param policy: 选股条件
        """
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

    @DateUtils.dtlize_input_dates
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
            ids = ids - set(self.risk_warned_stock_selector.get_data(date))

        ids = list(ids)
        return ids
