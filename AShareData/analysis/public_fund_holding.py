import datetime as dt
from typing import Dict

import pandas as pd
from functools import cached_property

from ..ashare_data_reader import AShareDataReader
from ..config import get_db_interface
from ..database_interface import DBInterface


class PublicFundHoldingRecords(object):
    def __init__(self, ticker: str, date: dt.datetime, db_interface: DBInterface = None):
        if db_interface is None:
            db_interface = get_db_interface()
        self.db_interface = db_interface
        self.data_reader = AShareDataReader(db_interface)
        self.ticker = ticker
        self.date = date

    @cached_property
    def cache(self):
        return self.db_interface.read_table('公募基金持仓', ['持有股票数量', '占股票市值比'],
                                            report_period=self.date, constitute_ticker=self.ticker)

    def stock_holding_by_funds(self):
        close = self.data_reader.stock_close.get_data(ids=self.ticker, dates=self.date).values[0]

        data = self.cache.loc[:, ['持有股票数量']].copy().droplevel(['DateTime', 'ConstituteTicker', '报告期'])
        data['市值'] = data['持有股票数量'] * close
        sec_name = self.data_reader.sec_name.get_data(ids=data.index.tolist(), dates=self.date).droplevel('DateTime')
        ret = pd.concat([sec_name, data], axis=1)
        ret = ret.sort_values('持有股票数量', ascending=False)

    def fund_holding_pct(self) -> Dict:
        fund_holding_shares = self.cache['持有股票数量'].sum()

        total_share = self.data_reader.total_share.get_data(ids=self.ticker, dates=self.date)[0]
        float_share = self.data_reader.float_a_shares.get_data(ids=self.ticker, dates=self.date)[0]
        free_float_share = self.data_reader.free_floating_share.get_data(ids=self.ticker, dates=self.date)[0]
        return {'基金持有': fund_holding_shares, '占总股本': fund_holding_shares / total_share,
                '占流通股本': fund_holding_shares / float_share, '占只有流通股本': fund_holding_shares / free_float_share}
