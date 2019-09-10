import datetime as dt
import json
from typing import Union, Optional, Sequence, List

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


def select_dates(date_list: Sequence[dt.datetime],
                 start_date: DateType = None, end_date: DateType = None) -> Sequence[dt.datetime]:
    if start_date:
        start_date = date_type2datetime(start_date)
        date_list = [it for it in date_list if it >= start_date]

    end_date = date_type2datetime(end_date) if end_date else dt.datetime.now()
    date_list = [it for it in date_list if it <= end_date]
    return date_list


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


def trading_days_offset(calendar: Sequence[dt.datetime], date: DateType, days: int) -> dt.datetime:
    date = date_type2datetime(date)
    return calendar[calendar.index(date) + days]


def get_calendar(engine: sa.engine) -> List[dt.datetime]:
    calendar_df = pd.read_sql_table('交易日历', engine)
    return calendar_df['交易日期'].dt.to_pydatetime().tolist()


def get_stocks(engine: sa.engine) -> List[str]:
    stock_list_df = pd.read_sql_table('股票上市退市', engine)
    return sorted(stock_list_df['ID'].unique().tolist())


def get_last_trading_day_of_month(calendar: Sequence[dt.datetime],
                                  start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
    pass
