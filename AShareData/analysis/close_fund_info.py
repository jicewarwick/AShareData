import datetime as dt

import pandas as pd
from dateutil.relativedelta import relativedelta

from ..config import get_db_interface
from ..DBInterface import DBInterface


def close_fund_opening_info(date: dt.datetime = None, db_interface: DBInterface = None):
    if date is None:
        date = dt.datetime.combine(dt.date.today(), dt.time())
    if db_interface is None:
        db_interface = get_db_interface()
    base_info = db_interface.read_table('基金列表', ['成立日期', '退市日期', '到期日期']).reset_index()
    base_info = base_info.loc[((base_info['退市日期'] > date.date()) | base_info['退市日期'].isnull()) &
                              ((base_info['到期日期'] > date.date()) | base_info['到期日期'].isnull()), :]
    tmp1 = base_info.copy()
    tmp1['ID'] = tmp1.ID.str.replace('.OF', '.SZ', regex=False)
    tmp2 = base_info.copy()
    tmp2['ID'] = tmp2.ID.str.replace('.OF', '.SH', regex=False)
    cog = pd.concat([base_info, tmp1, tmp2]).drop_duplicates().set_index('ID').drop(['退市日期', '到期日期'], axis=1)
    cog = cog.loc[~cog.index.str.endswith('.OF'), :]

    po_info = db_interface.read_table('基金拓展信息', ['全名', '定开', '定开时长(月)', '封闭运作转LOF时长(月)'])
    listing_info = pd.concat([cog, po_info], axis=1)
    listing_info = listing_info.dropna(subset=['成立日期', '全名'])
    funds = listing_info.loc[(listing_info['定开'] == 1) | (listing_info['封闭运作转LOF时长(月)'] > 0), :].copy()
    mask = funds['封闭运作转LOF时长(月)'] > 0
    funds.loc[mask, '定开时长(月)'] = funds['封闭运作转LOF时长(月)'][mask]

    tmp = pd.Series([relativedelta(months=it) for it in funds['定开时长(月)']], index=funds.index)
    funds.rename({'成立日期': '上一次开放日'}, axis=1, inplace=True)
    funds['下一次开放日'] = tmp + funds.loc[:, '上一次开放日']
    ind_base = funds['定开'].astype(bool)
    ind = (funds['下一次开放日'] < date.date()) & ind_base
    while any(ind):
        funds.loc[ind, '上一次开放日'] = funds.loc[ind, '下一次开放日']
        funds.loc[ind, '下一次开放日'] = (tmp + funds.loc[:, '上一次开放日']).loc[ind]
        ind = (funds['下一次开放日'] < date.date()) & ind_base

    funds['距离下次开放时间'] = [max((it - date.date()).days, 0) for it in funds['下一次开放日']]

    return funds.loc[:, ['全名', '上一次开放日', '下一次开放日', '距离下次开放时间']].sort_values('距离下次开放时间')
