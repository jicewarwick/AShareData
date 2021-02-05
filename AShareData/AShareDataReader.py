from functools import cached_property, lru_cache

import numpy as np

from . import DateUtils
from .config import generate_db_interface_from_config, get_db_interface
from .DBInterface import DBInterface
from .Factor import BetaFactor, BinaryFactor, CompactFactor, ContinuousFactor, IndexConstitute, IndustryFactor, \
    LatestAccountingFactor, OnTheRecordFactor, TTMAccountingFactor, UnaryFactor
from .Tickers import StockTickers


class AShareDataReader(object):
    """ AShare Data Reader"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        """
        AShare Data Reader

        :param db_interface: DBInterface
        """

        if not db_interface:
            db_interface = get_db_interface()
        self.db_interface = db_interface

    @cached_property
    def calendar(self) -> DateUtils.TradingCalendar:
        """交易日历"""
        return DateUtils.TradingCalendar(self.db_interface)

    @cached_property
    def stocks(self) -> StockTickers:
        """股票列表"""
        return StockTickers(self.db_interface)

    @cached_property
    def sec_name(self) -> CompactFactor:
        """股票名称"""
        return CompactFactor('证券名称', self.db_interface)

    @cached_property
    def adj_factor(self) -> CompactFactor:
        """复权因子"""
        return CompactFactor('复权因子', self.db_interface)

    @cached_property
    def free_a_shares(self) -> CompactFactor:
        """A股流通股本"""
        return CompactFactor('A股流通股本', self.db_interface)

    @cached_property
    def const_limit(self) -> OnTheRecordFactor:
        """一字涨跌停"""
        return OnTheRecordFactor('一字涨跌停', self.db_interface)

    @cached_property
    def stock_close(self) -> ContinuousFactor:
        """收盘价"""
        return ContinuousFactor('股票日行情', '收盘价', self.db_interface)

    @cached_property
    def stock_volume(self) -> ContinuousFactor:
        """成交量"""
        return ContinuousFactor('股票日行情', '成交量', self.db_interface)

    @cached_property
    def total_share(self) -> CompactFactor:
        """总股本"""
        return CompactFactor('总股本', self.db_interface)

    @cached_property
    def floating_share(self) -> CompactFactor:
        """流通股本"""
        return CompactFactor('流通股本', self.db_interface)

    @cached_property
    def free_floating_share(self) -> CompactFactor:
        """自由流通股本"""
        return CompactFactor('自由流通股本', self.db_interface)

    @cached_property
    def stock_market_cap(self) -> BinaryFactor:
        """市值"""
        f = self.total_share * self.stock_close
        f.set_factor_name('股票市值')
        return f

    @cached_property
    def stock_free_floating_market_cap(self) -> BinaryFactor:
        """自由流通市值"""
        f = self.free_floating_share * self.stock_close
        f.set_factor_name('自由流通市值')
        return f

    @cached_property
    def log_cap(self) -> UnaryFactor:
        f = self.stock_market_cap.log()
        f.set_factor_name('市值对数')
        return f

    @cached_property
    def hfq_close(self) -> BinaryFactor:
        f = self.adj_factor * self.stock_close
        f.set_factor_name('后复权收盘价')
        return f

    @cached_property
    def stock_return(self) -> UnaryFactor:
        f = self.hfq_close.pct_change()
        f.set_factor_name('股票收益率')
        return f

    @cached_property
    def forward_return(self) -> UnaryFactor:
        f = self.hfq_close.pct_change_shift(-1)
        f.set_factor_name('股票前瞻收益率')
        return f

    @cached_property
    def log_return(self) -> UnaryFactor:
        f = self.hfq_close.log().diff()
        f.set_factor_name('股票对数收益')
        return f

    @cached_property
    def forward_log_return(self) -> UnaryFactor:
        f = self.hfq_close.log().diff_shift(-1)
        f.set_factor_name('股票前瞻对数收益')
        return f

    @cached_property
    def index_close(self) -> ContinuousFactor:
        return ContinuousFactor('指数日行情', '收盘点位', self.db_interface)

    @cached_property
    def index_return(self) -> UnaryFactor:
        f = self.index_close.pct_change()
        f.set_factor_name('指数收益率')
        return f

    @cached_property
    def index_log_return(self) -> UnaryFactor:
        f = self.index_close.log().diff()
        f.set_factor_name('指数对数收益率')
        return f

    @cached_property
    def index_constitute(self) -> IndexConstitute:
        return IndexConstitute(self.db_interface)

    @lru_cache(5)
    def industry(self, provider: str, level: int) -> IndustryFactor:
        return IndustryFactor(provider, level, self.db_interface)

    @cached_property
    def beta(self) -> BetaFactor:
        return BetaFactor(db_interface=self.db_interface)

    @cached_property
    def book_val(self) -> LatestAccountingFactor:
        f = LatestAccountingFactor('股东权益合计(不含少数股东权益)', self.db_interface)
        f.set_factor_name('股东权益')
        return f

    @cached_property
    def earning_ttm(self) -> TTMAccountingFactor:
        f = TTMAccountingFactor('净利润(不含少数股东损益)', self.db_interface)
        f.set_factor_name('净利润TTM')
        return f

    @cached_property
    def bm(self) -> BinaryFactor:
        f = self.book_val / self.stock_market_cap
        f.set_factor_name('BM')
        return f

    @cached_property
    def pe_ttm(self) -> BinaryFactor:
        f = self.stock_market_cap / self.earning_ttm
        f.set_factor_name('PE_TTM')
        return f

    @cached_property
    def overnight_shibor(self) -> ContinuousFactor:
        f = ContinuousFactor('shibor利率数据', '隔夜', self.db_interface)
        f.set_factor_name('隔夜shibor')
        return f

    @cached_property
    def three_month_shibor(self) -> ContinuousFactor:
        f = ContinuousFactor('shibor利率数据', '3个月', self.db_interface)
        f.set_factor_name('3个月shibor')
        return f

    @cached_property
    def six_month_shibor(self) -> ContinuousFactor:
        f = ContinuousFactor('shibor利率数据', '6个月', self.db_interface)
        f.set_factor_name('6个月shibor')
        return f

    @cached_property
    def one_year_shibor(self) -> ContinuousFactor:
        f = ContinuousFactor('shibor利率数据', '1年', self.db_interface)
        f.set_factor_name('1年shibor')
        return f

    @staticmethod
    def exponential_weight(n: int, half_life: int):
        series = range(-(n - 1), 1)
        return np.exp(np.log(2) * series / half_life)

    @classmethod
    def from_config(cls, json_loc: str):
        db_interface = generate_db_interface_from_config(json_loc)
        return cls(db_interface)
