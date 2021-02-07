import bisect
import datetime as dt
import inspect
from functools import wraps
from typing import Callable, List, Optional, Sequence, Tuple, Union

from .config import get_db_interface
from .DBInterface import DBInterface

DateType = Union[str, dt.datetime, dt.date]


def date_type2str(date: DateType, delimiter: str = '') -> Optional[str]:
    if date is not None:
        formatter = delimiter.join(['%Y', '%m', '%d'])
        return date.strftime(formatter) if not isinstance(date, str) else date


def strlize_input_dates(func):
    @wraps(func)
    def inner(*args, **kwargs):
        signature = inspect.signature(func)
        for arg, (arg_name, _) in zip(args, signature.parameters.items()):
            if arg in ['start_date', 'end_date', 'dates', 'date', 'report_period']:
                kwargs[arg_name] = date_type2str(arg)
            else:
                kwargs[arg_name] = arg

        for it in kwargs.keys():
            if it in ['start_date', 'end_date', 'dates', 'date', 'report_period']:
                kwargs[it] = date_type2str(kwargs[it])
        return func(**kwargs)

    return inner


def date_type2datetime(date: Union[DateType, Sequence]) -> Optional[Union[dt.datetime, Sequence[dt.datetime]]]:
    if isinstance(date, str):
        return _date_type2datetime(date)
    elif isinstance(date, Sequence):
        return [_date_type2datetime(it) for it in date]
    else:
        return _date_type2datetime(date)


def _date_type2datetime(date: DateType) -> Optional[dt.datetime]:
    if isinstance(date, dt.datetime):
        return date
    if isinstance(date, dt.date):
        return dt.datetime.combine(date, dt.time())
    if isinstance(date, str) & (date not in ['', 'nan']):
        date = date.replace('/', '')
        date = date.replace('-', '')
        return dt.datetime.strptime(date, '%Y%m%d')


def dtlize_input_dates(func):
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


class TradingCalendarBase(object):
    def __init__(self):
        self.calendar = None

    @dtlize_input_dates
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
            for k in range(i, j):
                if func(self.calendar[k - 1], self.calendar[k], self.calendar[k + 1]):
                    storage.append(self.calendar[k])
            return storage
        else:
            return self.calendar[i:j]

    @dtlize_input_dates
    def select_dates(self, start_date: DateType = None, end_date: DateType = None, inclusive=(True, True)) \
            -> List[dt.datetime]:
        """Get list of all trading days during[``start_date``, ``end_date``] inclusive"""
        if start_date is None:
            start_date = self.calendar[0]
        if end_date is None:
            end_date = dt.datetime.now()
        dates = self._select_dates(start_date, end_date, lambda pre, curr, next_: True)
        if dates and not inclusive[0]:
            dates = dates[1:]
        if dates and not inclusive[1]:
            dates = dates[:-1]
        return dates

    @dtlize_input_dates
    def first_day_of_week(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get first trading day of weeks between [``start_date``, ``end_date``]"""
        return self._select_dates(start_date, end_date,
                                  lambda pre, curr, next_: pre.isocalendar()[1] != curr.isocalendar()[1])

    @dtlize_input_dates
    def last_day_of_week(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get last trading day of weeks between [``start_date``, ``end_date``]"""
        return self._select_dates(start_date, end_date,
                                  lambda pre, curr, next_: curr.isocalendar()[1] != next_.isocalendar()[1])

    @dtlize_input_dates
    def first_day_of_month(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get first trading day of months between [``start_date``, ``end_date``]"""
        return self._select_dates(start_date, end_date, lambda pre, curr, next_: pre.month != curr.month)

    @dtlize_input_dates
    def last_day_of_month(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get last trading day of months between [``start_date``, ``end_date``]"""
        return self._select_dates(start_date, end_date,
                                  lambda pre, curr, next_: curr.month != next_.month)

    @dtlize_input_dates
    def last_day_of_year(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        """Get last trading day of the year between [``start_date``, ``end_date``]"""
        return self._select_dates(start_date, end_date, lambda pre, curr, next_: curr.year != next_.year)

    @dtlize_input_dates
    def offset(self, date: DateType, days: int) -> dt.datetime:
        """offset ``date`` by number of days

        Push days forward if ``days`` is positive and backward if ``days`` is negative.
        If `days = 0` and date is a trading day, date is returned
        If `days = 0` and date is not a trading day, the next trading day is returned
        """
        loc = bisect.bisect_left(self.calendar, date)
        if self.calendar[loc] != date and days > 0:
            days = days - 1
        return self.calendar[loc + days]

    @dtlize_input_dates
    def middle(self, start_date: DateType, end_date: DateType) -> dt.datetime:
        """Get middle of the trading period[``start_date``, ``end_date``]"""
        return self.calendar[int((self.calendar.index(start_date) + self.calendar.index(end_date)) / 2.0)]

    @dtlize_input_dates
    def days_count(self, start_date: DateType, end_date: DateType) -> int:
        """Count number of trading days during [``start_date``, ``end_date``]
        Note: ``end_date`` need to be a trading day
        """
        i = bisect.bisect_left(self.calendar, start_date)
        if self.calendar[i] != start_date:
            i = i - 1
        j = self.calendar.index(end_date)

        return j - i

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


class TradingCalendar(TradingCalendarBase):
    """A Share Trading Calendar"""

    def __init__(self, db_interface: DBInterface = None):
        super().__init__()
        if db_interface is None:
            db_interface = get_db_interface()
        calendar_df = db_interface.read_table('交易日历')
        self.calendar = sorted(calendar_df['交易日期'].dt.to_pydatetime().tolist())


class HKTradingCalendar(TradingCalendarBase):
    """A Share Trading Calendar"""

    def __init__(self, db_interface: DBInterface = None):
        super().__init__()
        if db_interface is None:
            db_interface = get_db_interface()
        calendar_df = db_interface.read_table('港股交易日历')
        self.calendar = sorted(calendar_df['交易日期'].dt.to_pydatetime().tolist())


class ReportingDate(object):
    @staticmethod
    @dtlize_input_dates
    def yoy_date(date: DateType) -> dt.datetime:
        """
        返回去年同期的报告期
        :param date: 报告期
        :return: 去年同期的报告期
        """
        return dt.datetime(date.year - 1, date.month, date.day)

    @staticmethod
    @dtlize_input_dates
    def yearly_offset(date: DateType, delta: int = 1) -> dt.datetime:
        """
        返回后``delta``年的年报的报告期
        :param date: 报告期
        :param delta: 时长（年）
        :return: 前``delta``个年报的报告期
        """
        return dt.datetime(date.year + delta, 12, 31)

    @staticmethod
    @dtlize_input_dates
    def quarterly_offset(date: DateType, delta: int = 1) -> dt.datetime:
        """
        返回前``delta``个季度后的报告期
        :param date: 报告期
        :param delta: 时长（季度）
        :return: ``delta``个季度后的报告期
        """
        rep = date.year * 12 + date.month + delta * 3 - 1
        month = rep % 12 + 1
        day = 31 if month == 3 or month == 12 else 30
        return dt.datetime(rep // 12, month, day)

    @classmethod
    def offset(cls, report_date, offset_str: str):
        delta = -int(offset_str[1:])
        if offset_str[0] == 'q':
            return cls.quarterly_offset(report_date, delta)
        elif offset_str[0] == 'y':
            return cls.yearly_offset(report_date, delta)
        else:
            raise ValueError(f'Illegal offset_str: {offset_str}')

    @staticmethod
    def get_latest_report_date(date: Union[dt.date, dt.datetime] = None) -> List[dt.datetime]:
        """

        上市公司季报披露时间:
        一季报：4月1日——4月30日。
        二季报（中报）：7月1日——8月30日。
        三季报：10月1日——10月31日。
        四季报（年报）：1月1日——4月30日。

        :return: 最新财报的报告期
        """
        if date is None:
            date = dt.date.today()
        year = date.year
        if date.month < 5:
            return [dt.datetime(year, 3, 30), dt.datetime(year - 1, 12, 31)]
        elif date.month < 9:
            return [dt.datetime(year, 6, 31)]
        else:
            return [dt.datetime(year, 9, 30)]
