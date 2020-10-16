import datetime as dt
from typing import Callable, List, Sequence, Tuple

from .DBInterface import DBInterface
from .utils import date_type2datetime, DateType, format_input_dates


class TradingCalendar(object):
    """Trading Calendar Class"""

    def __init__(self, db_interface: DBInterface):
        calendar_df = db_interface.read_table('交易日历')
        self.calendar = calendar_df['交易日期'].dt.to_pydatetime().tolist()

    def is_trading_date(self, date: DateType):
        """return if ``date`` is a trading date"""
        return date_type2datetime(date) in self.calendar

    @staticmethod
    def _select_dates(calendar: Sequence[dt.datetime],
                      start_date: DateType = None, end_date: DateType = None,
                      func: Callable[[dt.datetime, dt.datetime, dt.datetime], bool] = None) -> List[dt.datetime]:
        calendar_len = len(calendar)
        i = 0
        for i in range(calendar_len):
            if calendar[i] >= start_date:
                break

        storage = []
        while i < calendar_len:
            if calendar[i] <= end_date:
                if func(calendar[i - 1], calendar[i], calendar[i + 1]):
                    storage.append(calendar[i])
                i = i + 1
            else:
                break
        return storage

    @format_input_dates
    def select_dates(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get list of all trading days during[``start_date``, ``end_date``]"""
        if not start_date:
            start_date = self.calendar[0]
        if not end_date:
            end_date = dt.datetime.now()
        return self._select_dates(self.calendar, start_date, end_date, lambda pre, curr, next_: True)

    @format_input_dates
    def first_day_of_week(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get first trading day of the months during[``start_date``, ``end_date``]"""
        return self._select_dates(self.calendar, start_date, end_date,
                                  lambda pre, curr, next_: pre.isocalendar()[1] != curr.isocalendar()[1])

    @format_input_dates
    def last_day_of_week(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get last trading day of the months during[``start_date``, ``end_date``]"""
        return self._select_dates(self.calendar, start_date, end_date,
                                  lambda pre, curr, next_: curr.isocalendar()[1] != next_.isocalendar()[1])

    @format_input_dates
    def first_day_of_month(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get first trading day of the months during[``start_date``, ``end_date``]"""
        return self._select_dates(self.calendar, start_date, end_date, lambda pre, curr, next_: pre.month != curr.month)

    @format_input_dates
    def last_day_of_month(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get last trading day of the months during[``start_date``, ``end_date``]"""
        return self._select_dates(self.calendar, start_date, end_date,
                                  lambda pre, curr, next_: curr.month != next_.month)

    @format_input_dates
    def last_day_of_year(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get last trading day of the year during[``start_date``, ``end_date``]"""
        return self._select_dates(self.calendar, start_date, end_date, lambda pre, curr, next_: curr.year != next_.year)

    @format_input_dates
    def offset(self, date: DateType, days: int) -> dt.datetime:
        """offset ``date`` by number of days

        Push days forward if ``days`` is positive and backward if ``days`` is negative.

        Note: ``date`` has to be a trading day
        """
        return self.calendar[self.calendar.index(date) + days]

    @format_input_dates
    def middle(self, start_date: DateType, end_date: DateType) -> dt.datetime:
        """Get middle of the trading period[``start_date``, ``end_date``]"""
        return self.calendar[int((self.calendar.index(start_date) + self.calendar.index(end_date)) / 2.0)]

    @format_input_dates
    def days_count(self, start_date: DateType, end_date: DateType) -> int:
        """Count number of trading days during [``start_date``, ``end_date``]"""
        return self.calendar.index(end_date) - self.calendar.index(start_date)

    def yesterday(self) -> dt.datetime:
        return self.offset(dt.date.today(), -1)

    def split_to_chunks(self, start_date: DateType, end_date: DateType, chunk_size: int) \
            -> List[Tuple[dt.datetime, dt.datetime]]:
        all_dates = self.select_dates(start_date, end_date)
        res = []
        for i in range(0, len(all_dates), chunk_size):
            tmp = all_dates[i:i + chunk_size]
            res.append((tmp[0], tmp[-1]))
        return res
