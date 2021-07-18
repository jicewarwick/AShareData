import datetime as dt
import json
import sys
import tempfile
from dataclasses import dataclass
from importlib.resources import open_text, read_binary
from typing import Any, Dict, List, Optional, Tuple, Union

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


def load_excel(default_loc: str, param_json_loc: str = None) -> List[Dict[str, Any]]:
    if param_json_loc is None:
        f = read_binary('AShareData.data', default_loc)
        df = pd.read_excel(f)
    else:
        df = pd.read_excel(param_json_loc)
    for col in df.columns:
        df[col] = df[col].where(df[col].notnull(), other=None)
    return df.to_dict('records')


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


def split_hs_ticker(ticker: str) -> Optional[Tuple[int, str]]:
    try:
        ticker_num, market = ticker.split('.')
    except (ValueError, AttributeError):
        return None
    if market not in ['SH', 'SZ']:
        return None
    try:
        ticker_num = int(ticker_num)
    except ValueError:
        return None
    return ticker_num, market


def is_main_board_stock(ticker: str) -> bool:
    """判断是否为沪深主板股票

    :param ticker: 股票代码, 如 `000001.SZ`
    """
    return get_stock_board_name(ticker) == '主板'


def get_stock_board_name(ticker: str) -> str:
    """获取股票所在版块(主板, 中小板, 创业板, 科创板), 其他返回 `非股票`

        :param ticker: 股票代码, 如 `000001.SZ`
    """
    ret = split_hs_ticker(ticker)
    if ret is None:
        return '非股票'
    ticker_num, market = ret
    if (0 < ticker_num < 2000 and market == 'SZ') or (600000 <= ticker_num < 606000 and market == 'SH'):
        return '主板'
    elif 2000 < ticker_num < 4000 and market == 'SZ':
        return '中小板'
    elif 300000 < ticker_num < 301000 and market == 'SZ':
        return '创业板'
    elif 688000 < ticker_num < 690000 and market == 'SH':
        return '科创板'
    else:
        return '非股票'


class SecuritySelectionPolicy:
    pass


@dataclass
class StockSelectionPolicy(SecuritySelectionPolicy):
    """ 股票筛选条件

    :param industry_provider: 股票行业分类标准
    :param industry_level: 股票行业分类标准
    :param industry: 股票所在行业

    :param ignore_new_stock_period: 新股纳入市场收益计算的时间(交易日天数)
    :param select_new_stock_period: 仅选取新上市的股票, 可与 ``ignore_new_stock_period`` 搭配使用

    :param ignore_st: 排除 风险警告股
    :param select_st: 仅选取 风险警告股, 包括 PT, ST, SST, \*ST, (即将)退市股 等
    :param st_defer_period: 新ST纳入计算的时间(交易日天数), 配合 ``select_st`` 使用

    :param select_pause: 选取停牌股
    :param ignore_pause: 排除停牌股
    :param max_pause_days: (i, n): 在前n个交易日中最大停牌天数不大于i

    :param ignore_const_limit: 排除一字板股票
    :param ignore_negative_book_value_stock: 排除净资产为负的股票
    """
    industry_provider: str = None
    industry_level: int = None
    industry: str = None

    ignore_new_stock_period: int = None
    select_new_stock_period: int = None

    ignore_st: bool = False
    select_st: bool = False
    st_defer_period: int = 10

    select_pause: bool = False
    ignore_pause: bool = False
    max_pause_days: Tuple[int, int] = None

    ignore_const_limit: bool = False
    ignore_negative_book_value_stock: bool = False

    def __post_init__(self):
        if self.ignore_new_stock_period:
            self.ignore_new_stock_period = int(self.ignore_new_stock_period)
        if self.select_new_stock_period:
            self.select_new_stock_period = int(self.select_new_stock_period)
        if self.industry_provider:
            if self.industry_provider not in constants.INDUSTRY_DATA_PROVIDER:
                raise ValueError('非法行业分类机构!')
            if not (0 < self.industry_level <= constants.INDUSTRY_LEVEL[self.industry_provider]):
                raise ValueError('非法行业分类级别!')
            self.industry_level = int(self.industry_level)
        if self.ignore_st & self.select_st:
            raise ValueError('不能同时选择ST股票和忽略ST股票')


@dataclass
class StockIndexCompositionPolicy:
    """ 自建指数信息

    :param ticker: 新建指数入库代码. 建议以`.IND`结尾, 代表自合成指数
    :param name: 指数名称
    :param unit_base: 股本指标
    :param stock_selection_policy: 股票筛选条件
    :param start_date: 指数开始日期
    """
    ticker: str = None
    name: str = None
    unit_base: str = None
    stock_selection_policy: StockSelectionPolicy = None
    start_date: dt.datetime = None

    def __post_init__(self):
        if self.unit_base and self.unit_base not in ['自由流通股本', '总股本', 'A股流通股本', 'A股总股本']:
            raise ValueError('非法股本字段!')

    @classmethod
    def from_dict(cls, info: Dict):
        info = info.copy()
        ticker = info.pop('ticker')
        name = info.pop('name')
        unit_base = info.pop('unit_base')
        start_date = info.pop('start_date')
        stock_selection_policy = StockSelectionPolicy(**info)
        return cls(ticker=ticker, name=name, unit_base=unit_base, stock_selection_policy=stock_selection_policy,
                   start_date=start_date)


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
