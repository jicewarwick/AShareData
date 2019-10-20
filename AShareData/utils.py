import datetime as dt
import json
from importlib.resources import open_text
from typing import Any, Dict, Optional, Union

import pandas as pd

DateType = Union[str, dt.datetime, dt.date]


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


def compute_diff(input_data: pd.DataFrame, db_data: pd.DataFrame) -> pd.DataFrame:
    db_data = db_data.unstack().ffill().tail(1).stack()
    tmp_data = pd.concat([input_data, db_data]).unstack().droplevel(None, axis=1)
    tmp_data = tmp_data.where(tmp_data.notnull(), None)
    diff = (tmp_data != tmp_data.shift())
    diff_stock = diff.iloc[-1, :]
    diff_stock = diff_stock.loc[diff_stock].index.tolist()
    return input_data.loc[(slice(None), diff_stock), :]


def load_param(default_loc: str, param_json_loc: str) -> Dict[str, Any]:
    if param_json_loc is None:
        f = open_text('AShareData.data', default_loc)
    else:
        f = open(param_json_loc, 'r', encoding='utf-8')
    with f:
        param = json.load(f)
        return param
