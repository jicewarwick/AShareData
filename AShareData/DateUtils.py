import bisect
import datetime as dt
import inspect
from functools import wraps
from typing import Callable, List, Optional, Sequence, Tuple, Union

from .DBInterface import DBInterface

DateType = Union[str, dt.datetime, dt.date]


def date_type2str(date: DateType, delimiter: str = '') -> Optional[str]:
    if date is not None:
        formatter = delimiter.join(['%Y', '%m', '%d'])
        return date.strftime(formatter) if not isinstance(date, str) else date


def date_type2datetime(date: Union[str, dt.date, dt.datetime, Sequence]) \
        -> Union[dt.datetime, Sequence[dt.datetime], None]:
    if isinstance(date, Sequence):
        return [_date_type2datetime(it) for it in date]
    else:
        return _date_type2datetime(date)


def _date_type2datetime(date: DateType) -> Optional[dt.datetime]:
    if isinstance(date, dt.datetime):
        return date
    if isinstance(date, dt.date):
        return dt.datetime.combine(date, dt.time())
    if isinstance(date, str) & (date not in ['', 'nan']):
        date.replace('/', '')
        date.replace('-', '')
        return dt.datetime.strptime(date, '%Y%m%d')


def format_input_dates(func):
    @wraps(func)
    def inner(*args, **kwargs):
        signature = inspect.signature(func)
        for arg, (arg_name, _) in zip(args, signature.parameters.items()):
            if arg in ['start_date', 'end_date', 'dates', 'date', 'report_period']:
                kwargs[arg_name] = date_type2datetime(arg)
            else:
                kwargs[arg_name] = arg

        for it in kwargs.keys():
            if it in ['start_date', 'end_date', 'dates', 'date', 'report_period']:
                kwargs[it] = date_type2datetime(kwargs[it])
        return func(**kwargs)

    return inner


class TradingCalendar(object):
    """Trading Calendar Class"""

    def __init__(self, db_interface: DBInterface):
        calendar_df = db_interface.read_table('交易日历')
        self.calendar = calendar_df['交易日期'].dt.to_pydatetime().tolist()

    @format_input_dates
    def is_trading_date(self, date: DateType):
        """return if ``date`` is a trading date"""
        return date in self.calendar

    def _select_dates(self, start_date: dt.datetime = None, end_date: dt.datetime = None,
                      func: Callable[[dt.datetime, dt.datetime, dt.datetime], bool] = None) -> List[dt.datetime]:
        i = bisect.bisect_left(self.calendar, start_date)
        j = bisect.bisect_right(self.calendar, end_date)
        if self.calendar[j] == end_date:
            j = j + 1

        if func:
            storage = []
            while i < j:
                if func(self.calendar[i - 1], self.calendar[i], self.calendar[i + 1]):
                    storage.append(self.calendar[i])
                i = i + 1
            return storage
        else:
            return self.calendar[i:j]

    @format_input_dates
    def select_dates(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get list of all trading days during[``start_date``, ``end_date``]"""
        if not start_date:
            start_date = self.calendar[0]
        if not end_date:
            end_date = dt.datetime.now()
        return self._select_dates(start_date, end_date, lambda pre, curr, next_: True)

    @format_input_dates
    def first_day_of_week(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get first trading day of the months during[``start_date``, ``end_date``]"""
        return self._select_dates(start_date, end_date,
                                  lambda pre, curr, next_: pre.isocalendar()[1] != curr.isocalendar()[1])

    @format_input_dates
    def last_day_of_week(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get last trading day of the months during[``start_date``, ``end_date``]"""
        return self._select_dates(start_date, end_date,
                                  lambda pre, curr, next_: curr.isocalendar()[1] != next_.isocalendar()[1])

    @format_input_dates
    def first_day_of_month(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get first trading day of the months during[``start_date``, ``end_date``]"""
        return self._select_dates(start_date, end_date, lambda pre, curr, next_: pre.month != curr.month)

    @format_input_dates
    def last_day_of_month(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get last trading day of the months during[``start_date``, ``end_date``]"""
        return self._select_dates(start_date, end_date,
                                  lambda pre, curr, next_: curr.month != next_.month)

    @format_input_dates
    def last_day_of_year(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get last trading day of the year during[``start_date``, ``end_date``]"""
        return self._select_dates(start_date, end_date, lambda pre, curr, next_: curr.year != next_.year)

    @format_input_dates
    def offset(self, date: DateType, days: int) -> dt.datetime:
        """offset ``date`` by number of days

        Push days forward if ``days`` is positive and backward if ``days`` is negative.

        Note: ``date`` has to be a trading day
        """
        loc = bisect.bisect_left(self.calendar, date)
        if self.calendar[loc] != date and days > 0:
            days = days - 1
        return self.calendar[loc + days]

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


class ReportingDate(object):
    @staticmethod
    @format_input_dates
    def yoy_date(date: DateType) -> dt.datetime:
        return dt.datetime(date.year - 1, date.month, date.day)

    @staticmethod
    @format_input_dates
    def qoq_date(date: DateType):
        qoq_dict = {3: dt.datetime(date.year - 1, 12, 31),
                    6: dt.datetime(date.year, 3, 31),
                    9: dt.datetime(date.year, 6, 30),
                    12: dt.datetime(date.year, 9, 30)}
        return qoq_dict[date.month]

    @staticmethod
    @format_input_dates
    def pre_yearly_dates(date: DateType) -> dt.datetime:
        return dt.datetime(date.year - 1, 12, 31)

    @staticmethod
    @format_input_dates
    def ttm_dates(date: DateType) -> (dt.datetime, Optional[dt.datetime]):
        p1 = ReportingDate.yoy_date(date)
        p2 = ReportingDate.pre_yearly_dates(date) if date.month != 12 else None
        return p1, p2
