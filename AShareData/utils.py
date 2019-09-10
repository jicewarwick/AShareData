import datetime as dt
import json
from typing import Union, Optional, List

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine.url import URL

DateType = Union[str, dt.datetime, dt.date]


def date_type2str(date: DateType, delimiter: str = '') -> str:
    formatter = delimiter.join(['%Y', '%m', '%d'])
    return date.strftime(formatter) if not isinstance(date, str) else date


def date_type2datetime(date: str) -> Optional[dt.datetime]:
    if isinstance(date, dt.datetime):
        return date
    if isinstance(date, dt.date):
        return dt.datetime.combine(date, dt.time())
    if isinstance(date, str) & (date not in ['', 'nan']):
        return dt.datetime.strptime(date, '%Y%m%d')


def stock_code2ts_code(stock_code: Union[int, str]) -> str:
    stock_code = int(stock_code)
    return f'{stock_code:06}.SH' if stock_code >= 600000 else f'{stock_code:06}.SZ'


def ts_code2stock_code(ts_code: str) -> str:
    return ts_code.split()[0]


def prepare_engine(config_loc: str) -> sa.engine.Engine:
    with open(config_loc, 'r') as f:
        config = json.load(f)
    url = URL(drivername=config['driver'], host=config['host'], port=config['port'], database=config['database'],
              username=config['username'], password=config['password'],
              query={'charset': 'utf8mb4'})
    return sa.create_engine(url)


def _prepare_example_json(config_loc, example_config_loc) -> None:
    with open(config_loc, 'r') as f:
        config = json.load(f)
    for key in config.keys():
        config[key] = '********' if isinstance(config[key], str) else 0
    with open(example_config_loc, 'w') as fh:
        json.dump(config, fh, indent=4)

# _prepare_example_json('data.json', 'config_example.json')


def get_calendar(engine: sa.engine) -> List[dt.datetime]:
    calendar_df = pd.read_sql_table('交易日历', engine)
    return calendar_df['交易日期'].dt.to_pydatetime().tolist()


def get_stocks(engine: sa.engine) -> List[str]:
    stock_list_df = pd.read_sql_table('股票上市退市', engine)
    return sorted(stock_list_df['ID'].unique().tolist())


class TradingCalendar(object):
    def __init__(self, calendar: List[dt.datetime]):
        self.calendar = calendar

    def select_dates(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        calendar = self.calendar.copy()
        if start_date:
            start_date = date_type2datetime(start_date)
            calendar = [it for it in calendar if it >= start_date]

        end_date = date_type2datetime(end_date) if end_date else dt.datetime.now()
        return [it for it in calendar if it <= end_date]

    def get_first_day_of_month(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        calendar = self.select_dates(start_date, end_date)
        storage = []
        for i in range(len(calendar) - 1):
            if calendar[i].month != calendar[i + 1].month:
                storage.append(calendar[i + 1])
        return storage

    def get_last_day_of_month(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        calendar = self.select_dates(start_date, end_date)
        storage = []
        for i in range(len(calendar) - 1):
            if calendar[i].month != calendar[i + 1].month:
                storage.append(calendar[i])
        return storage

    def offset(self, date: DateType, days: int) -> dt.datetime:
        date = date_type2datetime(date)
        return self.calendar[self.calendar.index(date) + days]

    def middle(self, start_date: DateType, end_date: DateType) -> dt.datetime:
        start_date, end_date = date_type2datetime(start_date), date_type2datetime(end_date)
        return self.calendar[int((self.calendar.index(start_date) + self.calendar.index(end_date)) / 2.0)]

    def days_count(self, start_date: DateType, end_date: DateType) -> int:
        start_date, end_date = date_type2datetime(start_date), date_type2datetime(end_date)
        ind = 1 if start_date <= end_date else -1
        return ind * abs(self.calendar.index(end_date) - self.calendar.index(start_date) + 1)
