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
        date.replace('/', '')
        date.replace('-', '')
        return dt.datetime.strptime(date, '%Y%m%d')


def _prepare_example_json(config_loc, example_config_loc) -> None:
    with open(config_loc, 'r') as f:
        config = json.load(f)
    for key in config.keys():
        config[key] = '********' if isinstance(config[key], str) else 0
    with open(example_config_loc, 'w') as fh:
        json.dump(config, fh, indent=4)


def compute_diff(input_data: pd.Series, db_data: pd.Series) -> Optional[pd.Series]:
    if db_data.empty:
        return input_data

    db_data = db_data.groupby('ID').tail(1)
    combined_data = pd.concat([db_data.droplevel('DateTime'), input_data.droplevel('DateTime')], axis=1, sort=True)
    stocks = combined_data.index[combined_data.iloc[:, 0] != combined_data.iloc[:, 1]]
    return input_data.loc[slice(None), stocks, :]


def load_param(default_loc: str, param_json_loc: str) -> Dict[str, Any]:
    if param_json_loc is None:
        f = open_text('AShareData.data', default_loc)
    else:
        f = open(param_json_loc, 'r', encoding='utf-8')
    with f:
        param = json.load(f)
        return param
