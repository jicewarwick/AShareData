import datetime as dt
from typing import List

from cached_property import cached_property

from . import utils
from .DBInterface import DBInterface, get_stocks
from .TradingCalendar import TradingCalendar


class DataSource(object):
    """Data Source Base Class"""

    def __init__(self, db_interface: DBInterface):
        self.db_interface = db_interface

    @cached_property
    def all_stocks(self) -> List[str]:
        """获取所有股票列表"""
        return get_stocks(self.db_interface)

    @cached_property
    def calendar(self) -> TradingCalendar:
        """获取交易日历"""
        return TradingCalendar(self.db_interface)

    def _check_db_timestamp(self, table_name: str, default_timestamp: utils.DateType) -> dt.datetime:
        latest_time = self.db_interface.get_latest_timestamp(table_name)
        if latest_time is None:
            latest_time = utils.date_type2datetime(default_timestamp)
        return latest_time
