import datetime as dt
from functools import cached_property, reduce
from itertools import product
from typing import Dict, List, Sequence, Union

import pandas as pd
from dateutil.relativedelta import relativedelta

from . import date_utils
from .config import get_db_interface
from .database_interface import DBInterface
from .factor import CompactFactor, CompactRecordFactor, IndustryFactor, OnTheRecordFactor
from .utils import Singleton, StockSelectionPolicy, TickerSelector


class FundInfo(Singleton):
    def __init__(self, db_interface: DBInterface = None):
        super().__init__()
        if db_interface is None:
            db_interface = get_db_interface()
        self.data = db_interface.read_table('基金列表')


class TickersBase(object):
    """证券代码基类"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        self.db_interface = db_interface if db_interface else get_db_interface()
        self.cache = None

    def all_ticker(self) -> List[str]:
        """ return ALL ticker for the asset class"""
        return sorted(self.cache.ID.unique().tolist())

    @date_utils.dtlize_input_dates
    def ticker(self, date: date_utils.DateType = None) -> List[str]:
        """ return tickers that are alive on `date`, `date` default to today"""
        if date is None:
            date = dt.datetime.today()
        stock_ticker_df = self.cache.loc[self.cache.DateTime <= date]
        tmp = stock_ticker_df.groupby('ID').tail(1)
        return sorted(tmp.loc[tmp['上市状态'] == 1, 'ID'].tolist())

    def alive_tickers(self, dates: Sequence[date_utils.DateType]) -> List[str]:
        """ return tickers that are alive in ALL dates in `dates` """
        ticker_set = reduce(lambda x, y: x & y, [set(self.ticker(date)) for date in dates])
        return sorted(list(ticker_set))

    def list_date(self) -> Dict[str, dt.datetime]:
        """ return the list date of all tickers"""
        first_list_info = self.cache.groupby('ID').head(1)
        return dict(zip(first_list_info.ID, first_list_info.DateTime))

    def get_list_date(self, tickers: Union[str, Sequence[str]]) -> Union[pd.Series, dt.datetime]:
        """ return the list date of a ticker"""
        if isinstance(tickers, str):
            tickers = [tickers]
        info = self.cache.loc[self.cache.ID.isin(tickers) & self.cache['上市状态'] == 1, :].set_index('ID')
        ret = info.DateTime.iloc[0] if info.shape[0] == 1 else info.DateTime
        return ret

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


class StockIndexFutureIndex(FutureTickers):
    """股指期货合约代码"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__(db_interface)
        mask = self.cache.ID.str.startswith('IH') | self.cache.ID.str.startswith('IF') | self.cache.ID.str.startswith(
            'IC')
        self.cache = self.cache.loc[mask, :]


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
        super().__init__('场内基金', db_interface)
        fund_info = FundInfo(db_interface)
        all_tickers = fund_info.data.loc[(fund_info.data['ETF'] == True) & (fund_info.data['投资类型'] == '被动指数型基金'), :]
        self.cache = self.cache.loc[self.cache.ID.isin(all_tickers.index.tolist()), :]


class BondETFTickers(DiscreteTickers):
    """债券ETF基金代码"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('场内基金', db_interface)
        fund_info = FundInfo(db_interface)
        all_tickers = fund_info.data.loc[(fund_info.data['ETF'] == True) & (fund_info.data['投资类型'] == '被动指数型债券基金'), :]
        self.cache = self.cache.loc[self.cache.ID.isin(all_tickers.index.tolist()), :]


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


class ETFTickers(DiscreteTickers):
    """ETF"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('场内基金', db_interface)
        fund_info = FundInfo(db_interface)
        all_tickers = fund_info.data.loc[fund_info.data['ETF'] == True, :]
        self.cache = self.cache.loc[self.cache.ID.isin(all_tickers.index.tolist()), :]


class ExchangeFundTickers(DiscreteTickers):
    """场内基金"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('场内基金', db_interface)


class OTCFundTickers(DiscreteTickers):
    """场外基金"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__('场外基金', db_interface)


class InvestmentStyleFundTicker(DiscreteTickers):
    def __init__(self, investment_type: Sequence[str], otc: bool = False, db_interface: DBInterface = None) -> None:
        """ 某些投资风格的基金

        :param investment_type: [普通股票型基金, 灵活配置型基金, 偏股混合型基金, 平衡混合型基金, 被动指数型基金, 增强指数型基金, 股票多空,
            短期纯债型基金, 中长期纯债型基金, 混合债券型一级基金, 混合债券型二级基金, 偏债混合型基金, 被动指数型债券基金, 增强指数型债券基金,
            商品型基金,
            货币市场型基金,
            国际(QDII)股票型基金, 国际增强指数型基金, (QDII)混合型基金, 国际(QDII)债券型基金, 国际(QDII)另类投资基金,
            REITs]
        :param otc: 选择 OTC 基金代码 或 .SH / .SZ 的基金代码
        :param db_interface: DBInterface
        """
        type_name = '场外基金' if otc else '场内基金'
        super().__init__(type_name, db_interface)
        self.fund_info = FundInfo(db_interface)
        all_tickers = self.fund_info.data.loc[self.fund_info.data['投资类型'].isin(investment_type), :]
        self.cache = self.cache.loc[self.cache.ID.isin(all_tickers.index.tolist()), :]

    def get_next_open_day(self, ids: Union[Sequence[str], str], date: dt.datetime = None):
        if date is None:
            date = dt.datetime.combine(dt.date.today(), dt.time())
        if isinstance(ids, str):
            ids = [ids]
        list_date = self.get_list_date(ids)
        period = self.fund_info.data.loc[ids, '定开时长(月)']
        df = pd.concat([list_date, period], axis=1)
        storage = []
        for ticker, row in df.iterrows():
            if pd.isna(row['定开时长(月)']):
                storage.append(pd.NaT)
                continue
            open_day = row.DateTime
            while open_day < date:
                open_day = open_day + relativedelta(months=row['定开时长(月)'])
            storage.append(open_day)
        return pd.Series(storage, index=df.index)


class StockFundTickers(InvestmentStyleFundTicker):
    """
    股票型基金

    以股票为主要(>=50%)投资标的的基金
    """

    def __init__(self, otc: bool = False, db_interface: DBInterface = None) -> None:
        stock_investment_type = ['偏股混合型基金', '被动指数型基金', '灵活配置型基金', '增强指数型基金', '普通股票型基金', '股票多空', '平衡混合型基金']
        super().__init__(stock_investment_type, otc, db_interface)


class FundWithStocksTickers(InvestmentStyleFundTicker):
    """可以投资股票的基金 """

    def __init__(self, otc: bool = False, db_interface: DBInterface = None) -> None:
        stock_investment_type = ['偏股混合型基金', '被动指数型基金', '灵活配置型基金', '增强指数型基金', '普通股票型基金', '股票多空', '平衡混合型基金', '混合债券型二级基金',
                                 '混合债券型一级基金', '偏债混合型基金']
        super().__init__(stock_investment_type, otc, db_interface)


class EnhancedIndexFund(InvestmentStyleFundTicker):
    """股票指数增强基金"""

    def __init__(self, otc: bool = False, db_interface: DBInterface = None) -> None:
        stock_investment_type = ['增强指数型基金']
        super().__init__(stock_investment_type, otc, db_interface)


class IndexFund(InvestmentStyleFundTicker):
    """指数基金"""

    def __init__(self, otc: bool = False, db_interface: DBInterface = None) -> None:
        stock_investment_type = ['被动指数型基金']
        super().__init__(stock_investment_type, otc, db_interface)


class ActiveManagedStockFundTickers(InvestmentStyleFundTicker):
    """以股票为主要(>=50%)投资标的的主动管理型基金"""

    def __init__(self, otc: bool = False, db_interface: DBInterface = None) -> None:
        stock_investment_type = ['偏股混合型基金', '灵活配置型基金', '增强指数型基金', '普通股票型基金', '股票多空', '平衡混合型基金']
        super().__init__(stock_investment_type, otc, db_interface)


class StockTickerSelector(TickerSelector):
    """股票代码选择器"""

    def __init__(self, policy: StockSelectionPolicy, db_interface: DBInterface = None) -> None:
        """
        :param db_interface: BDInterface
        :param policy: 选股条件
        """
        super().__init__()
        self.db_interface = db_interface if db_interface else get_db_interface()
        self.calendar = date_utils.SHSZTradingCalendar(self.db_interface)
        self.stock_ticker = StockTickers(self.db_interface)
        self.policy = policy

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
    def negative_book_value_stock_selector(self):
        return CompactFactor('负净资产股票', self.db_interface)

    @cached_property
    def industry_info(self):
        if self.policy.industry:
            return IndustryFactor(self.policy.industry_provider, self.policy.industry_level, self.db_interface)

    @date_utils.dtlize_input_dates
    def ticker(self, date: date_utils.DateType, ids: Sequence[str] = None) -> List[str]:
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

        if self.policy.ignore_negative_book_value_stock:
            data = self.negative_book_value_stock_selector.get_data(dates=date)
            ids = ids - set(data.loc[data == True].index.get_level_values('ID').tolist())

        ids = sorted(list(ids))
        return ids

    def generate_index(self, start_date: date_utils.DateType = None, end_date: date_utils.DateType = None,
                       dates: Union[date_utils.DateType, Sequence[date_utils.DateType]] = None) -> pd.MultiIndex:
        storage = []
        if dates is None:
            dates = self.calendar.select_dates(start_date, end_date)
        for date in dates:
            ids = self.ticker(date)
            storage.extend(list(product([date], ids)))
        return pd.MultiIndex.from_tuples(storage, names=['DateTime', 'ID'])
