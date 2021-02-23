import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from AShareData import DateUtils, utils
from AShareData.config import get_db_interface
from AShareData.DBInterface import DBInterface
from AShareData.Factor import ContinuousFactor

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_factor_return(factor_name: str, weight: bool = True, industry_neutral: bool = True, bins: int = 5,
                       start_date: DateUtils.DateType = None, end_date: DateUtils.DateType = None,
                       db_interface: DBInterface = None) -> plt.Figure:
    if db_interface is None:
        db_interface = get_db_interface()

    ids = utils.generate_factor_bin_names(factor_name, weight=weight, industry_neutral=industry_neutral, bins=bins)
    data = db_interface.read_table('因子分组收益率', ids=ids, start_date=start_date, end_date=end_date)
    df = (data.unstack() + 1).cumprod()
    bin_names_info = [utils.decompose_bin_names(it) for it in df.columns]
    diff_series = df[ids[0]] - df[ids[-1]]

    df.columns = [it['group'] for it in bin_names_info]
    diff_series.name = f'{utils.decompose_bin_names(ids[0])["group"]}-{utils.decompose_bin_names(ids[-1])["group"]}'

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex='col')
    df.plot(ax=axes[0])
    industry_neutral_str = '行业中性' if industry_neutral else '非行业中性'
    weight_str = '市值加权' if weight else '等权'
    axes[0].set_title(f'{factor_name} 分组收益率({industry_neutral_str}, {weight_str})')
    plot_dt = df.index.get_level_values('DateTime')
    axes[0].set_xlim(left=plot_dt[0], right=plot_dt[-1])
    axes[0].grid(True)

    diff_series.plot(ax=axes[1])
    axes[1].grid(True)
    axes[1].legend()

    return fig


def plot_index(index_factor: ContinuousFactor, benchmark_factor: ContinuousFactor = None,
               start_date=None, end_date=None) -> Axes:
    data = index_factor.get_data(start_date=start_date, end_date=end_date)
    if benchmark_factor:
        benchmark_data = benchmark_factor.get_data(start_date=start_date, end_date=end_date)
        data = pd.concat([data, benchmark_data])
    data = data.unstack()
    val = (data + 1).cumprod()

    axes = val.plot()
    axes.set_xlim(left=val.index[0], right=val.index[-1])
    axes.grid(True)
    return axes
