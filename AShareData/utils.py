import datetime as dt
import json
from typing import List, Optional, Union

from AShareData.DBInterface import DBInterface

DateType = Union[str, dt.datetime, dt.date]


def get_stocks(db_interface: DBInterface) -> List[str]:
    stock_list_df = db_interface.read_table('股票上市退市')
    return sorted(stock_list_df['ID'].unique().tolist())


def date_type2str(date: DateType, delimiter: str = '') -> Optional[str]:
    if date is not None:
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


def _prepare_example_json(config_loc, example_config_loc) -> None:
    with open(config_loc, 'r') as f:
        config = json.load(f)
    for key in config.keys():
        config[key] = '********' if isinstance(config[key], str) else 0
    with open(example_config_loc, 'w') as fh:
        json.dump(config, fh, indent=4)

# _prepare_example_json('data.json', 'config_example.json')
