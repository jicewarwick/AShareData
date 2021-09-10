from functools import cached_property, lru_cache

from . import date_utils
from .config import generate_db_interface_from_config, get_db_interface
from .database_interface import DBInterface
from .factor import BetaFactor, BinaryFactor, CompactFactor, ContinuousFactor, FactorBase, IndexConstitute, \
    IndustryFactor, InterestRateFactor, LatestAccountingFactor, LatestUpdateFactor, OnTheRecordFactor, \
    TTMAccountingFactor, UnaryFactor
from .tickers import StockTickers


class DataReader(object):
    def __init__(self, db_interface: DBInterface = None) -> None:
        """
        Data Reader Base Class

        :param db_interface: DBInterface
        """

        self.db_interface = db_interface if db_interface else get_db_interface()
        self.calendar = date_utils.SHSZTradingCalendar(self.db_interface)

    @classmethod
    def from_config(cls, json_loc: str):
        """根据 `config_loc` 的适配信息生成 :py:class:`.DataReader` 实例"""
        db_interface = generate_db_interface_from_config(json_loc)
        return cls(db_interface)

    @cached_property
    def sec_name(self) -> CompactFactor:
        """证券名称"""
        return CompactFactor('证券名称', self.db_interface)

    @cached_property
    def latest_sec_name(self) -> LatestUpdateFactor:
        """证券名称"""
        return LatestUpdateFactor('证券名称', '证券名称', self.db_interface)


class DataReaderWithAdjFactor(DataReader):
    adj_factor = None

    def __init__(self, db_interface: DBInterface = None) -> None:
        """
        Data Reader with adj_factor

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        if self.adj_factor is None:
            self.adj_factor = CompactFactor('复权因子', self.db_interface)


class StockDataReader(DataReaderWithAdjFactor):
    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__(db_interface)

    @cached_property
    def stocks(self) -> StockTickers:
        """股票列表"""
        return StockTickers(self.db_interface)

    @cached_property
    def float_a_shares(self) -> CompactFactor:
        """A股流通股本"""
        return CompactFactor('A股流通股本', self.db_interface)

    @cached_property
    def const_limit(self) -> OnTheRecordFactor:
        """一字涨跌停"""
        return OnTheRecordFactor('一字涨跌停', self.db_interface)

    @cached_property
    def open(self) -> ContinuousFactor:
        """股票开盘价"""
        return ContinuousFactor('股票日行情', '开盘价', self.db_interface)

    @cached_property
    def close(self) -> ContinuousFactor:
        """股票收盘价"""
        return ContinuousFactor('股票日行情', '收盘价', self.db_interface)

    @cached_property
    def volume(self) -> ContinuousFactor:
        """股票成交量"""
        return ContinuousFactor('股票日行情', '成交量', self.db_interface)

    @cached_property
    def turnover(self) -> ContinuousFactor:
        """股票成交额"""
        return ContinuousFactor('股票日行情', '成交额', self.db_interface)

    @cached_property
    def turnover_rate(self) -> ContinuousFactor:
        """股票换手率"""
        return (self.turnover / (self.close * self.free_floating_share)).set_factor_name('换手率')

    @cached_property
    def total_share(self) -> CompactFactor:
        """股票总股本"""
        return CompactFactor('总股本', self.db_interface)

    @cached_property
    def free_floating_share(self) -> CompactFactor:
        """股票自由流通股本"""
        return CompactFactor('自由流通股本', self.db_interface)

    @cached_property
    def total_market_cap(self) -> BinaryFactor:
        """股票总市值"""
        return (self.total_share * self.close).set_factor_name('股票市值')

    @cached_property
    def free_floating_market_cap(self) -> BinaryFactor:
        """股票自由流通市值"""
        return (self.free_floating_share * self.close).set_factor_name('自由流通市值')

    @cached_property
    def free_floating_cap_weight(self) -> UnaryFactor:
        """自由流通市值权重"""
        return self.free_floating_market_cap.weight().set_factor_name('自由流通市值权重')

    @cached_property
    def log_cap(self) -> UnaryFactor:
        """股票市值对数"""
        return self.total_market_cap.log().set_factor_name('市值对数')

    @cached_property
    def hfq_close(self) -> BinaryFactor:
        """股票后复权收盘价"""
        return (self.adj_factor * self.close).set_factor_name('后复权收盘价')

    @cached_property
    def returns(self) -> UnaryFactor:
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
    def market_return(self) -> ContinuousFactor:
        """全市场收益率"""
        return ContinuousFactor('自合成指数', '收益率', self.db_interface).bind_params(ids='全市场.IND')

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
    def total_debt(self) -> LatestAccountingFactor:
        """Total Debt"""
        return LatestAccountingFactor('负债合计', self.db_interface)

    @cached_property
    def total_asset(self) -> LatestAccountingFactor:
        """Total Asset"""
        return LatestAccountingFactor('资产总计', self.db_interface)

    @cached_property
    def earning_ttm(self) -> TTMAccountingFactor:
        """Earning Trailing Twelve Month"""
        return TTMAccountingFactor('净利润(不含少数股东损益)', self.db_interface).set_factor_name('净利润TTM')

    @cached_property
    def bm(self) -> BinaryFactor:
        """Book to Market"""
        return (self.book_val / self.total_market_cap).set_factor_name('BM')

    @cached_property
    def bm_after_close(self) -> BinaryFactor:
        """After market Book to Market value"""
        return (self.book_val.shift(-1) / self.total_market_cap).set_factor_name('BM')

    @cached_property
    def pb(self) -> BinaryFactor:
        """Price to Book"""
        return (self.total_market_cap / self.book_val).set_factor_name('PB')

    @cached_property
    def bp(self) -> BinaryFactor:
        """Book to Price"""
        return (self.book_val / self.total_market_cap).set_factor_name('BP')

    # TODO
    @cached_property
    def pb_after_close(self) -> BinaryFactor:
        """After market Price to Book"""
        return (self.total_market_cap / self.book_val.shift(-1)).set_factor_name('BM')

    @cached_property
    def pe_ttm(self) -> BinaryFactor:
        """Price to Earning Trailing Twelve Month"""
        return (self.total_market_cap / self.earning_ttm).set_factor_name('PE_TTM')

    @cached_property
    def debt_to_asset(self) -> BinaryFactor:
        return (self.total_debt / self.total_asset).set_factor_name('资产负债率')


class ConvertibleBondDataReader(DataReader):
    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__(db_interface)

    @cached_property
    def close(self) -> ContinuousFactor:
        """可转债收盘价"""
        return ContinuousFactor('可转债日行情', '收盘价', self.db_interface)

    @cached_property
    def cb_total_val(self) -> ContinuousFactor:
        """可转债未转股余额"""
        return ContinuousFactor('可转债日行情', '未转股余额', self.db_interface)

    @cached_property
    def convert_price(self) -> CompactFactor:
        """可转债转股价"""
        return CompactFactor('可转债转股价').set_factor_name('转股价')


class FutureDataReader(DataReader):
    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__(db_interface)

    @cached_property
    def close(self) -> ContinuousFactor:
        """期货收盘价"""
        return ContinuousFactor('期货日行情', '收盘价', self.db_interface)


class FundDataReader(DataReaderWithAdjFactor):
    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__(db_interface)


class OTCFundDataReader(FundDataReader):
    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__(db_interface)

    @cached_property
    def nav(self) -> ContinuousFactor:
        """场外基金单位净值"""
        return ContinuousFactor('场外基金净值', '单位净值', self.db_interface)

    @cached_property
    def hfq_nav(self) -> BinaryFactor:
        """场外基金后复权净值"""
        return (self.nav * self.adj_factor).set_factor_name('基金后复权净值')


class ExchangeFundDataReader(FundDataReader):
    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__(db_interface)

    @cached_property
    def nav(self) -> ContinuousFactor:
        """场内基金单位净值"""
        return ContinuousFactor('场内基金日行情', '单位净值', self.db_interface)

    @cached_property
    def hfq_nav(self) -> BinaryFactor:
        """场内基金后复权净值"""
        return (self.nav * self.adj_factor).set_factor_name('基金后复权净值')


class SHIBORDataReader(DataReader):
    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__(db_interface)

    @cached_property
    def overnight(self) -> InterestRateFactor:
        """隔夜shibor"""
        return InterestRateFactor('shibor利率数据', '隔夜', self.db_interface).set_factor_name('隔夜shibor')

    @cached_property
    def three_month(self) -> InterestRateFactor:
        """三月期shibor"""
        return InterestRateFactor('shibor利率数据', '3个月', self.db_interface).set_factor_name('3个月shibor')

    @cached_property
    def six_month(self) -> InterestRateFactor:
        """6月期shibor"""
        return InterestRateFactor('shibor利率数据', '6个月', self.db_interface).set_factor_name('6个月shibor')

    @cached_property
    def one_year(self) -> InterestRateFactor:
        """一年期shibor"""
        return InterestRateFactor('shibor利率数据', '1年', self.db_interface).set_factor_name('1年shibor')


class IndexDataReader(DataReader):
    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__(db_interface)

    @cached_property
    def close(self) -> ContinuousFactor:
        """指数收盘价"""
        return ContinuousFactor('指数日行情', '收盘点位', self.db_interface)

    @cached_property
    def returns(self) -> UnaryFactor:
        """指数收益率"""
        return self.close.pct_change().set_factor_name('指数收益率')

    @cached_property
    def log_return(self) -> UnaryFactor:
        """指数对数收益率"""
        return self.close.log().diff().set_factor_name('指数对数收益率')

    @cached_property
    def constitute(self) -> IndexConstitute:
        """指数成分股权重"""
        return IndexConstitute(self.db_interface)

    @cached_property
    def user_constructed_index_return(self) -> ContinuousFactor:
        """自合成指数收益率"""
        return ContinuousFactor('自合成指数', '收益率', self.db_interface)

    def get_index_return_factor(self, ticker: str) -> FactorBase:
        factor = ContinuousFactor('自合成指数', '收益率') if ticker.endswith('.IND') else self.returns
        return factor.bind_params(ids=ticker)


class ModelDataReader(DataReader):
    def __init__(self, db_interface: DBInterface = None) -> None:
        super().__init__(db_interface)

    @cached_property
    def model_factor_return(self):
        return ContinuousFactor('模型因子收益率', '收益率', self.db_interface)
