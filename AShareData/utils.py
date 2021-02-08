import datetime as dt
import json
import sys
import tempfile
from dataclasses import dataclass
from importlib.resources import open_text
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from . import constants


class NullPrinter(object):
    def __init__(self):
        self._stdout = None
        self._std_error = None
        self._temp_file = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._std_error = sys.stderr
        self._temp_file = tempfile.TemporaryFile(mode='w')
        sys.stdout = self._temp_file
        sys.stderr = self._temp_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._temp_file.close()
        sys.stdout = self._stdout
        sys.stderr = self._std_error


def load_param(default_loc: str, param_json_loc: str = None) -> Dict[str, Any]:
    if param_json_loc is None:
        f = open_text('AShareData.data', default_loc)
    else:
        f = open(param_json_loc, 'r', encoding='utf-8')
    with f:
        param = json.load(f)
        return param


def chunk_list(l: list, n: int):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def format_stock_ticker(ticker: Union[str, int]) -> str:
    if isinstance(ticker, str):
        ticker = int(ticker)
    if ticker < 600000:
        return f'{ticker:06d}.SZ'
    else:
        return f'{ticker:06d}.SH'


def format_czc_ticker(ticker: str) -> str:
    c = ticker[1] if ticker[1].isnumeric() else ticker[2]
    ticker = ticker.replace(c, '', 1)
    return ticker


def full_czc_ticker(ticker: str) -> str:
    c = 1 if ticker[1].isnumeric() else 2
    ticker = ticker[:c] + '2' + ticker[c:]
    return ticker


class SecuritySelectionPolicy:
    pass


@dataclass
class StockSelectionPolicy(SecuritySelectionPolicy):
    """ 股票筛选条件 """
    industry_provider: str = None  # 股票行业分类标准
    industry_level: int = None  # 股票行业分类标准
    industry: str = None  # 股票所在行业

    ignore_new_stock_period: int = None  # 新股纳入市场收益计算的时间(交易日天数)
    select_new_stock_period: int = None  # 仅选取新上市的股票, 可与 ignore_new_stock_period 搭配使用

    select_st: bool = False  # 仅选取 风险警告股, 即 PT, ST, SST, *ST, (即将)退市股 等
    ignore_st: bool = False  # 排除 风险警告股

    select_pause: bool = False  # 选取停牌股
    ignore_pause: bool = False  # 排除停牌股
    max_pause_days: Tuple[int, int] = None  # (i, n): 在前n个交易日中最大停牌天数不大于i

    ignore_const_limit: bool = False  # 排除一字板股票

    def __post_init__(self):
        if self.industry_provider:
            assert self.industry_provider in constants.INDUSTRY_DATA_PROVIDER, '非法行业分类机构!'
            assert self.industry_level <= constants.INDUSTRY_LEVEL[self.industry_provider], '非法行业分类级别!'


@dataclass
class StockIndexCompositionPolicy:
    """ 自建指数信息 """
    ticker: str = None  # 新建指数入库代码. 建议以`.IND`结尾, 代表自合成指数
    name: str = None  # 指数名称
    unit_base: str = None  # 股本指标
    stock_selection_policy: StockSelectionPolicy = None  # 股票筛选条件
    start_date: dt.datetime = None  # 指数开始日期

    def __post_init__(self):
        assert self.unit_base in ['自由流通股本', '总股本', 'A股流通股本', 'A股总股本'], '非法股本字段!'


class TickerSelector(object):
    def __init__(self):
        super().__init__()

    def generate_index(self, *args, **kwargs) -> pd.MultiIndex:
        raise NotImplementedError()

    def ticker(self, *args, **kwargs) -> List[str]:
        raise NotImplementedError()


def generate_factor_bin_names(factor_name: str, weight: bool = True, industry_neutral: bool = True, bins: int = 10):
    i = 'I' if industry_neutral else 'N'
    w = 'W' if weight else 'N'
    return [f'{factor_name}_{i}{w}_G{it}inG{bins}' for it in range(1, bins + 1)]


def decompose_bin_names(factor_bin_name):
    tmp = factor_bin_name.split('_')
    composition_info = tmp[1]
    group_info = tmp[-1].split('in')

    return {
        'factor_name': tmp[0],
        'industry_neutral': composition_info[0] == 'I',
        'cap_weight': composition_info[1] == 'W',
        'group': group_info[0],
        'total_group': group_info[-1]
    }
