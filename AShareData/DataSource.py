import datetime as dt
from functools import cached_property

from . import DateUtils
from .DBInterface import DBInterface


class DataSource(object):
    """Data Source Base Class"""

    def __init__(self, db_interface: DBInterface) -> None:
        self.db_interface = db_interface

    @cached_property
    def calendar(self) -> DateUtils.TradingCalendar:
        """交易日历"""
        return DateUtils.TradingCalendar(self.db_interface)

    def _check_db_timestamp(self, table_name: str, default_timestamp: DateUtils.DateType,
                            column_condition: (str, str) = None) -> dt.datetime:
        latest_time = self.db_interface.get_latest_timestamp(table_name, column_condition)
        if latest_time is None:
            latest_time = DateUtils.date_type2datetime(default_timestamp)
        return latest_time
