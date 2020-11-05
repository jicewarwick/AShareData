from .Factor import *
from .Tickers import StockTickers


class AShareDataReader(object):
    """ AShare Data Reader"""

    def __init__(self, db_interface: DBInterface) -> None:
        """
        AShare Data Reader

        :param db_interface: DBInterface
        """
        self.db_interface = db_interface

    @cached_property
    def calendar(self) -> DateUtils.TradingCalendar:
        """Trading Calendar"""
        return DateUtils.TradingCalendar(self.db_interface)

    @cached_property
    def stocks(self) -> StockTickers:
        return StockTickers(self.db_interface)

    @cached_property
    def sec_name(self):
        return CompactFactor(self.db_interface, '证券名称')

    @cached_property
    def adj_factor(self) -> CompactFactor:
        return CompactFactor(self.db_interface, '复权因子')

    @cached_property
    def free_a_shares(self) -> CompactFactor:
        return CompactFactor(self.db_interface, 'A股流通股本')

    @cached_property
    def const_limit(self) -> OnTheRecordFactor:
        return OnTheRecordFactor(self.db_interface, '一字涨跌停')

    @cached_property
    def close(self) -> ContinuousFactor:
        return ContinuousFactor(self.db_interface, '股票日行情', '收盘价')
