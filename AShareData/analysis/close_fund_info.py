import datetime as dt

import pandas as pd
from dateutil.relativedelta import relativedelta

from ..config import get_db_interface
from ..database_interface import DBInterface
from ..tickers import ExchangeFundTickers, OTCFundTickers


def close_fund_opening_info(date: dt.datetime = None, db_interface: DBInterface = None):
    if date is None:
        date = dt.datetime.combine(dt.date.today(), dt.time())
    if db_interface is None:
        db_interface = get_db_interface()
    exchange_fund_tickers = ExchangeFundTickers(db_interface)
    tickers = exchange_fund_tickers.ticker()

    info = db_interface.read_table('基金列表', ['全名', '定开', '定开时长(月)', '封闭运作转LOF时长(月)', '投资类型'], ids=tickers)
    funds = info.loc[(info['定开'] == 1) | (info['封闭运作转LOF时长(月)'] > 0), :].copy()
    of_ticker = [it.replace('.SH', '.OF').replace('.SZ', '.OF') for it in funds.index.tolist()]
    list_date = OTCFundTickers(db_interface).get_list_date(of_ticker).sort_index()
    list_date.index = funds.index
    list_date.name = '成立日期'
    mask = funds['封闭运作转LOF时长(月)'] > 0
    funds.loc[mask, '定开时长(月)'] = funds['封闭运作转LOF时长(月)'][mask]
    funds = pd.concat([funds, list_date], axis=1)

    tmp = pd.Series([relativedelta(months=it) for it in funds['定开时长(月)']], index=funds.index)
    funds.rename({'成立日期': '上一次开放日'}, axis=1, inplace=True)
    funds['下一次开放日'] = tmp + funds.loc[:, '上一次开放日']
    ind_base = funds['定开'].astype(bool)
    ind = (funds['下一次开放日'] < date) & ind_base
    while any(ind):
        funds.loc[ind, '上一次开放日'] = funds.loc[ind, '下一次开放日']
        funds.loc[ind, '下一次开放日'] = (tmp + funds.loc[:, '上一次开放日']).loc[ind]
        ind = (funds['下一次开放日'] < date) & ind_base

    funds['距离下次开放时间'] = [max((it - date).days, 0) for it in funds['下一次开放日']]

    return funds.loc[:, ['全名', '投资类型', '上一次开放日', '下一次开放日', '距离下次开放时间']].sort_values('距离下次开放时间')
