import datetime as dt
from typing import Union, Optional, Sequence

DateType = Union[str, dt.datetime, dt.date]


def date_type2str(date: DateType) -> str:
    return date.strftime('%Y%m%d') if not isinstance(date, str) else date


def date_type2datetime(date: str) -> Optional[dt.datetime]:
    if isinstance(date, dt.datetime):
        return date
    if isinstance(date, dt.date):
        return dt.datetime.combine(date, dt.time())
    if isinstance(date, str) & (date not in ['', 'nan']):
        return dt.datetime.strptime(date, '%Y%m%d')


def select_dates(date_list: Sequence[dt.datetime],
                 start_date: DateType = None, end_date: DateType = None) -> Sequence[dt.datetime]:
    if start_date:
        start_date = date_type2datetime(start_date)
        date_list = [it for it in date_list if it >= start_date]

    end_date = date_type2datetime(end_date) if end_date else dt.datetime.now()
    date_list = [it for it in date_list if it <= end_date]
    return date_list
