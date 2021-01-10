import datetime as dt
import json
import sys
import tempfile
from collections import namedtuple
from dataclasses import dataclass
from importlib.resources import open_text
from typing import Any, Dict, Union

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


@dataclass
class StockSelectionPolicy:
    """
    股票筛选条件:
        :param name: 名称
        :param industry_provider: 股票行业分类标准
        :param industry_level: 股票行业分类标准
        :param industry: 股票所在行业
        :param ignore_st: 排除 风险警告股, 包括 PT, ST, SST, \*ST, (即将)退市股 等
        :param ignore_new_stock_period: 新股纳入市场收益计算的时间
        :param ignore_pause: 排除停牌股
        :param ignore_const_limit: 排除一字板股票
    """
    industry_provider: str = None
    industry_level: int = None
    industry: str = None
    ignore_new_stock_period: dt.timedelta = None
    select_st: bool = False
    ignore_st: bool = False
    select_pause: bool = False
    ignore_pause: bool = False
    ignore_const_limit: bool = False

    def __post_init__(self):
        if self.industry_provider:
            assert self.industry_provider in constants.INDUSTRY_DATA_PROVIDER, '非法行业分类机构!'
            assert self.industry_level <= constants.INDUSTRY_LEVEL[self.industry_provider], '非法行业分类级别!'


@dataclass
class IndexCompositionPolicy:
    """
    自建指数信息
        :param ticker: 新建指数入库代码. 建议以`.IND`结尾, 代表自合成指数
        :param stock_selection_policy: 股票筛选条件
    """
    ticker: str = None
    name: str = None
    unit_base: str = None
    stock_selection_policy: StockSelectionPolicy = None
    start_date: dt.datetime = None

    def __post_init__(self):
        assert self.unit_base in ['自由流通股本', '总股本', 'A股流通股本', 'A股总股本'], '非法股本字段!'


DateCache = namedtuple('DateCache', ['q0', 'q1', 'q2', 'y1', 'q4', 'q5', 'y3', 'y5'])
