from typing import Tuple, Union

import empyrical
import pandas as pd

from .. import date_utils
from ..factor import ContinuousFactor


@date_utils.dtlize_input_dates
def aggregate_returns(target: ContinuousFactor, convert_to: str, benchmark_factor: ContinuousFactor = None,
                      start_date: date_utils.DateType = None, end_date: date_utils.DateType = None
                      ) -> Union[pd.Series, pd.DataFrame]:
    """ 按 年/月/周 统计收益

    :param target: 标的收益率
    :param convert_to: 周期, 可为 ``yearly`` (年), ``monthly`` (月), ``weekly`` (周)
    :param benchmark_factor: 基准收益率
    :param start_date: 开始时间
    :param end_date: 结束时间
    :return: 各个周期的收益率. 若指定基准则还会计算各周期差值列( ``diff`` )
    """

    def _agg_ret(factor):
        target_returns = factor.get_data(start_date=start_date, end_date=end_date).unstack().iloc[:, 0]
        agg_target_ret = empyrical.aggregate_returns(target_returns, convert_to)
        return agg_target_ret

    ret = _agg_ret(target)
    if benchmark_factor:
        agg_benchmark_return = _agg_ret(benchmark_factor)
        ret = pd.concat([ret, agg_benchmark_return], axis=1)
        ret['diff'] = ret.iloc[:, 0] - ret.iloc[:, 1]
    return ret


def locate_max_drawdown(returns: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp, float]:
    """ 寻找最大回撤周期

    :param returns: 收益序列, 已时间为 ``index``
    :return: (最大回撤开始时间, 最大回撤结束时间, 最大回撤比例)
    """
    if len(returns) < 1:
        raise ValueError('returns is empty.')

    cumulative = empyrical.cum_returns(returns, starting_value=100)
    max_return = cumulative.cummax()
    drawdown = cumulative.sub(max_return).div(max_return)
    val = drawdown.min()
    end = drawdown.index[drawdown.argmin()]
    start = drawdown.loc[(drawdown == 0) & (drawdown.index <= end)].index[-1]
    return start, end, val
