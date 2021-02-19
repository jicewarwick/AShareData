import datetime as dt
from functools import cached_property, lru_cache
from typing import Sequence

import numpy as np

from . import DateUtils
from .config import generate_db_interface_from_config, get_db_interface
from .DBInterface import DBInterface
from .Factor import BetaFactor, BinaryFactor, CompactFactor, ContinuousFactor, IndexConstitute, IndustryFactor, \
    LatestAccountingFactor, OnTheRecordFactor, TTMAccountingFactor, UnaryFactor
from .Tickers import StockTickers


class AShareDataReader(object):
    def __init__(self, db_interface: DBInterface = None) -> None:
        """
        AShare Data Reader

        :param db_interface: DBInterface
        """

        if db_interface is None:
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
        """股票收盘价"""
        return ContinuousFactor('股票日行情', '收盘价', self.db_interface)

    @cached_property
    def stock_volume(self) -> ContinuousFactor:
        """股票成交量"""
        return ContinuousFactor('股票日行情', '成交量', self.db_interface)

    @cached_property
    def total_share(self) -> CompactFactor:
        """股票总股本"""
        return CompactFactor('总股本', self.db_interface)

    @cached_property
    def floating_share(self) -> CompactFactor:
        """股票流通股本"""
        return CompactFactor('流通股本', self.db_interface)

    @cached_property
    def free_floating_share(self) -> CompactFactor:
        """股票自由流通股本"""
        return CompactFactor('自由流通股本', self.db_interface)

    @cached_property
    def stock_market_cap(self) -> BinaryFactor:
        """股票总市值"""
        f = self.total_share * self.stock_close
        f.set_factor_name('股票市值')
        return f

    @cached_property
    def stock_free_floating_market_cap(self) -> BinaryFactor:
        """股票自由流通市值"""
        f = self.free_floating_share * self.stock_close
        f.set_factor_name('自由流通市值')
        return f

    @cached_property
    def log_cap(self) -> UnaryFactor:
        """股票市值对数"""
        f = self.stock_market_cap.log()
        f.set_factor_name('市值对数')
        return f

    @cached_property
    def hfq_close(self) -> BinaryFactor:
        """股票后复权收盘价"""
        f = self.adj_factor * self.stock_close
        f.set_factor_name('后复权收盘价')
        return f

    @cached_property
    def stock_return(self) -> UnaryFactor:
        """股票收益率"""
        f = self.hfq_close.pct_change()
        f.set_factor_name('股票收益率')
        return f

    @cached_property
    def forward_return(self) -> UnaryFactor:
        """股票前瞻收益率"""
        f = self.hfq_close.pct_change_shift(-1)
        f.set_factor_name('股票前瞻收益率')
        return f

    @cached_property
    def log_return(self) -> UnaryFactor:
        """股票对数收益率"""
        f = self.hfq_close.log().diff()
        f.set_factor_name('股票对数收益')
        return f

    @cached_property
    def forward_log_return(self) -> UnaryFactor:
        """股票前瞻对数收益率"""
        f = self.hfq_close.log().diff_shift(-1)
        f.set_factor_name('股票前瞻对数收益')
        return f

    @cached_property
    def index_close(self) -> ContinuousFactor:
        """指数收盘价"""
        return ContinuousFactor('指数日行情', '收盘点位', self.db_interface)

    @cached_property
    def index_return(self) -> UnaryFactor:
        """指数收益率"""
        f = self.index_close.pct_change()
        f.set_factor_name('指数收益率')
        return f

    @cached_property
    def index_log_return(self) -> UnaryFactor:
        """指数对数收益率"""
        f = self.index_close.log().diff()
        f.set_factor_name('指数对数收益率')
        return f

    @cached_property
    def index_constitute(self) -> IndexConstitute:
        """指数成分股权重"""
        return IndexConstitute(self.db_interface)

    @lru_cache(5)
    def industry(self, provider: str, level: int) -> IndustryFactor:
        """stock industry"""
        return IndustryFactor(provider, level, self.db_interface)

    @cached_property
    def beta(self) -> BetaFactor:
        """stock beat"""
        return BetaFactor(db_interface=self.db_interface)

    @cached_property
    def book_val(self) -> LatestAccountingFactor:
        """Book value"""
        f = LatestAccountingFactor('股东权益合计(不含少数股东权益)', self.db_interface)
        f.set_factor_name('股东权益')
        return f

    @cached_property
    def earning_ttm(self) -> TTMAccountingFactor:
        """Earning Trailing Twelve Month"""
        f = TTMAccountingFactor('净利润(不含少数股东损益)', self.db_interface)
        f.set_factor_name('净利润TTM')
        return f

    @cached_property
    def bm(self) -> BinaryFactor:
        """Book to Market"""
        f = self.book_val / self.stock_market_cap
        f.set_factor_name('BM')
        return f

    @cached_property
    def pe_ttm(self) -> BinaryFactor:
        """Price to Earning Trailing Twelve Month"""
        f = self.stock_market_cap / self.earning_ttm
        f.set_factor_name('PE_TTM')
        return f

    @cached_property
    def overnight_shibor(self) -> ContinuousFactor:
        """隔夜shibor"""
        f = ContinuousFactor('shibor利率数据', '隔夜', self.db_interface)
        f.set_factor_name('隔夜shibor')
        return f

    @cached_property
    def three_month_shibor(self) -> ContinuousFactor:
        """三月期shibor"""
        f = ContinuousFactor('shibor利率数据', '3个月', self.db_interface)
        f.set_factor_name('3个月shibor')
        return f

    @cached_property
    def six_month_shibor(self) -> ContinuousFactor:
        """6月期shibor"""
        f = ContinuousFactor('shibor利率数据', '6个月', self.db_interface)
        f.set_factor_name('6个月shibor')
        return f

    @cached_property
    def one_year_shibor(self) -> ContinuousFactor:
        """一年期shibor"""
        f = ContinuousFactor('shibor利率数据', '1年', self.db_interface)
        f.set_factor_name('1年shibor')
        return f

    def weighted_return(self, date: dt.datetime, ids: Sequence[str], weight_base: CompactFactor = None,
                        pre_date: dt.datetime = None) -> float:
        if pre_date is None:
            pre_date = self.calendar.offset(date, -1)
        pre_close_data = self.stock_close.get_data(dates=pre_date, ids=ids)
        pre_adj = self.adj_factor.get_data(dates=pre_date, ids=ids)

        hfq_close = self.hfq_close.get_data(dates=date, ids=ids)
        stock_return = hfq_close.values / (pre_close_data * pre_adj).values - 1

        if weight_base is None:
            daily_ret = stock_return.mean()
        else:
            pre_units = weight_base.get_data(dates=pre_date, ids=ids)
            cap = pre_units * pre_close_data
            weight = cap / cap.sum()
            daily_ret = stock_return.dot(weight.T.values)

        return daily_ret

    @staticmethod
    def exponential_weight(n: int, half_life: int):
        series = range(-(n - 1), 1)
        return np.exp(np.log(2) * series / half_life)

    @classmethod
    def from_config(cls, json_loc: str):
        """根据 ``config_loc`` 的适配信息生成 ``AShareDataReader`` 实例"""
        db_interface = generate_db_interface_from_config(json_loc)
        return cls(db_interface)
