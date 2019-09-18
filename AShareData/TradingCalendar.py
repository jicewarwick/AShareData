import datetime as dt
from typing import Callable, List, Sequence

from AShareData.DBInterface import DBInterface
from AShareData.utils import date_type2datetime, DateType


class TradingCalendar(object):
    def __init__(self, db_interface: DBInterface):
        calendar_df = db_interface.read_table('交易日历')
        self.calendar = calendar_df['交易日期'].dt.to_pydatetime().tolist()

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

    def select_dates(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        start_date = date_type2datetime(start_date)
        end_date = date_type2datetime(end_date) if end_date else dt.datetime.now()
        return self._select_dates(self.calendar, start_date, end_date, lambda pre, curr, next_: True)

    def first_day_of_month(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        start_date, end_date = date_type2datetime(start_date), date_type2datetime(end_date)
        return self._select_dates(self.calendar, start_date, end_date, lambda pre, curr, next_: pre.month != curr.month)

    def last_day_of_month(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        start_date, end_date = date_type2datetime(start_date), date_type2datetime(end_date)
        return self._select_dates(self.calendar, start_date, end_date,
                                  lambda pre, curr, next_: curr.month != next_.month)

    def last_day_of_year(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        start_date, end_date = date_type2datetime(start_date), date_type2datetime(end_date)
        return self._select_dates(self.calendar, start_date, end_date, lambda pre, curr, next_: curr.year != next_.year)

    def offset(self, date: DateType, days: int) -> dt.datetime:
        date = date_type2datetime(date)
        return self.calendar[self.calendar.index(date) + days]

    def middle(self, start_date: DateType, end_date: DateType) -> dt.datetime:
        start_date, end_date = date_type2datetime(start_date), date_type2datetime(end_date)
        return self.calendar[int((self.calendar.index(start_date) + self.calendar.index(end_date)) / 2.0)]

    def days_count(self, start_date: DateType, end_date: DateType) -> int:
        start_date, end_date = date_type2datetime(start_date), date_type2datetime(end_date)
        return self.calendar.index(end_date) - self.calendar.index(start_date)
