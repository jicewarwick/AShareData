import matplotlib.pyplot as plt

from AShareData.config import *
from AShareData.Factor import *
from AShareData.FactorCompositor import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

config_loc = './config.json'
set_global_config(config_loc)

data_reader = AShareDataReader()

factor_name = 'Beta'
weight = True
industry_neutral = True
bins = 5
start_date = None
end_date = None
db_interface = None


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
