import datetime as dt
import json
from collections import namedtuple
from dataclasses import dataclass
from importlib.resources import open_text
from typing import Any, Dict

from . import constants


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
