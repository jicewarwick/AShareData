from functools import cached_property, lru_cache

import numpy as np

from . import DateUtils
from .config import generate_db_interface_from_config, get_db_interface
from .DBInterface import DBInterface
from .Factor import BetaFactor, BinaryFactor, CompactFactor, ContinuousFactor, FactorBase, IndexConstitute, \
    IndustryFactor, InterestRateFactor, LatestAccountingFactor, OnTheRecordFactor, TTMAccountingFactor, UnaryFactor
from .Tickers import StockTickers


class AShareDataReader(object):
    def __init__(self, db_interface: DBInterface = None) -> None:
        """
        AShare Data Reader

        :param db_interface: DBInterface
        """

        self.db_interface = db_interface if db_interface else get_db_interface()
        self.calendar = DateUtils.SHSZTradingCalendar(self.db_interface)

    @cached_property
    def stocks(self) -> StockTickers:
        """股票列表"""
        return StockTickers(self.db_interface)

    @cached_property
    def sec_name(self) -> CompactFactor:
        """证券名称"""
        return CompactFactor('证券名称', self.db_interface)

    @cached_property
    def adj_factor(self) -> CompactFactor:
        """复权因子"""
        return CompactFactor('复权因子', self.db_interface)

    @cached_property
    def float_a_shares(self) -> CompactFactor:
        """A股流通股本"""
        return CompactFactor('A股流通股本', self.db_interface)

    @cached_property
    def const_limit(self) -> OnTheRecordFactor:
        """一字涨跌停"""
        return OnTheRecordFactor('一字涨跌停', self.db_interface)

    @cached_property
    def stock_open(self) -> ContinuousFactor:
        """股票开盘价"""
        return ContinuousFactor('股票日行情', '开盘价', self.db_interface)

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
    def free_floating_share(self) -> CompactFactor:
        """股票自由流通股本"""
        return CompactFactor('自由流通股本', self.db_interface)

    @cached_property
    def stock_market_cap(self) -> BinaryFactor:
        """股票总市值"""
        return (self.total_share * self.stock_close).set_factor_name('股票市值')

    @cached_property
    def stock_free_floating_market_cap(self) -> BinaryFactor:
        """股票自由流通市值"""
        return (self.free_floating_share * self.stock_close).set_factor_name('自由流通市值')

    @cached_property
    def free_floating_cap_weight(self) -> UnaryFactor:
        """自由流通市值权重"""
        return self.stock_free_floating_market_cap.weight().set_factor_name('自由流通市值权重')

    @cached_property
    def log_cap(self) -> UnaryFactor:
        """股票市值对数"""
        return self.stock_market_cap.log().set_factor_name('市值对数')

    @cached_property
    def hfq_close(self) -> BinaryFactor:
        """股票后复权收盘价"""
        return (self.adj_factor * self.stock_close).set_factor_name('后复权收盘价')

    @cached_property
    def stock_return(self) -> UnaryFactor:
        """股票收益率"""
        return self.hfq_close.pct_change().set_factor_name('股票收益率')

    @cached_property
    def forward_return(self) -> UnaryFactor:
        """股票前瞻收益率"""
        return self.hfq_close.pct_change_shift(-1).set_factor_name('股票前瞻收益率')

    @cached_property
    def log_return(self) -> UnaryFactor:
        """股票对数收益率"""
        return self.hfq_close.log().diff().set_factor_name('股票对数收益')

    @cached_property
    def forward_log_return(self) -> UnaryFactor:
        """股票前瞻对数收益率"""
        return self.hfq_close.log().diff_shift(-1).set_factor_name('股票前瞻对数收益')

    @cached_property
    def index_close(self) -> ContinuousFactor:
        """指数收盘价"""
        return ContinuousFactor('指数日行情', '收盘点位', self.db_interface)

    @cached_property
    def index_return(self) -> UnaryFactor:
        """指数收益率"""
        return self.index_close.pct_change().set_factor_name('指数收益率')

    @cached_property
    def user_constructed_index_return(self) -> ContinuousFactor:
        """自合成指数收益率"""
        return ContinuousFactor('自合成指数', '收益率', self.db_interface)

    @cached_property
    def market_return(self) -> ContinuousFactor:
        """全市场收益率"""
        return ContinuousFactor('自合成指数', '收益率', self.db_interface).bind_params(ids='全市场.IND')

    @cached_property
    def model_factor_return(self) -> ContinuousFactor:
        """模型因子收益率"""
        return ContinuousFactor('模型因子收益率', '收益率', self.db_interface)

    @cached_property
    def index_log_return(self) -> UnaryFactor:
        """指数对数收益率"""
        return self.index_close.log().diff().set_factor_name('指数对数收益率')

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
        return LatestAccountingFactor('股东权益合计(不含少数股东权益)', self.db_interface).set_factor_name('股东权益')

    @cached_property
    def earning_ttm(self) -> TTMAccountingFactor:
        """Earning Trailing Twelve Month"""
        return TTMAccountingFactor('净利润(不含少数股东损益)', self.db_interface).set_factor_name('净利润TTM')

    @cached_property
    def bm(self) -> BinaryFactor:
        """Book to Market"""
        return (self.book_val / self.stock_market_cap).set_factor_name('BM')

    @cached_property
    def bm_after_close(self) -> BinaryFactor:
        """After market Book to Market value"""
        return (self.book_val.shift(-1) / self.stock_market_cap).set_factor_name('BM')

    @cached_property
    def pb(self) -> BinaryFactor:
        """Price to Book"""
        return (self.stock_market_cap / self.book_val).set_factor_name('PB')

    @cached_property
    def cb_close(self) -> ContinuousFactor:
        """可转债收盘价"""
        return ContinuousFactor('可转债日行情', '收盘价', self.db_interface)

    @cached_property
    def cb_total_val(self) -> ContinuousFactor:
        """可转债未转股余额"""
        return ContinuousFactor('可转债日行情', '未转股余额', self.db_interface)

    @cached_property
    def cb_convert_price(self) -> CompactFactor:
        """可转债转股价"""
        return CompactFactor('可转债转股价').set_factor_name('转股价')

    # TODO
    @cached_property
    def pb_after_close(self) -> BinaryFactor:
        """After market Price to Book"""
        return (self.stock_market_cap / self.book_val.shift(-1)).set_factor_name('BM')

    @cached_property
    def pe_ttm(self) -> BinaryFactor:
        """Price to Earning Trailing Twelve Month"""
        return (self.stock_market_cap / self.earning_ttm).set_factor_name('PE_TTM')

    @cached_property
    def overnight_shibor(self) -> InterestRateFactor:
        """隔夜shibor"""
        return InterestRateFactor('shibor利率数据', '隔夜', self.db_interface).set_factor_name('隔夜shibor')

    @cached_property
    def three_month_shibor(self) -> InterestRateFactor:
        """三月期shibor"""
        return InterestRateFactor('shibor利率数据', '3个月', self.db_interface).set_factor_name('3个月shibor')

    @cached_property
    def six_month_shibor(self) -> InterestRateFactor:
        """6月期shibor"""
        return InterestRateFactor('shibor利率数据', '6个月', self.db_interface).set_factor_name('6个月shibor')

    @cached_property
    def one_year_shibor(self) -> InterestRateFactor:
        """一年期shibor"""
        return InterestRateFactor('shibor利率数据', '1年', self.db_interface).set_factor_name('1年shibor')

    @cached_property
    def model_factor_return(self):
        return ContinuousFactor('模型因子收益率', '收益率', self.db_interface)

    def get_index_return_factor(self, ticker: str) -> FactorBase:
        factor = ContinuousFactor('自合成指数', '收益率') if ticker.endswith('.IND') else self.index_return
        return factor.bind_params(ids=ticker)

    @staticmethod
    def exponential_weight(n: int, half_life: int):
        series = range(-(n - 1), 1)
        return np.exp(np.log(2) * series / half_life)

    @classmethod
    def from_config(cls, json_loc: str):
        """根据 ``config_loc`` 的适配信息生成 ``AShareDataReader`` 实例"""
        db_interface = generate_db_interface_from_config(json_loc)
        return cls(db_interface)
