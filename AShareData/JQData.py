import datetime as dt
import logging
import sys
import tempfile
from cached_property import cached_property
from typing import Dict, List, Sequence, Union, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from jqdatasdk import auth, bond, query, get_price, get_call_auction


from . import constants, utils, DateUtils
from .DataSource import DataSource
from .DBInterface import DBInterface
from .Tickers import ETFTickers, FutureTickers, OptionTickers, StockTickers


class JQData(DataSource):
    def __init__(self, db_interface: DBInterface, mobile: str, password: str):
        super().__init__(db_interface)

        with tempfile.TemporaryFile(mode='w') as log_file:
            out = sys.stdout
            out2 = sys.stderr
            sys.stdout = log_file
            sys.stderr = log_file
            try:
                auth(mobile, password)
            except:
                logging.error('JQData Login Error!')
            finally:
                sys.stdout = out
                sys.stderr = out2

        self._factor_param = utils.load_param('jqdata_param.json')

    @cached_property
    def stock_tickers(self):
        return StockTickers(self.db_interface)

    def update_convertible_bond_list(self):
        q = query(bond.BOND_BASIC_INFO).filter(bond.BOND_BASIC_INFO.bond_type == '可转债')
        df = bond.run_query(q)
        exchange = df.exchange.map({'深交所主板': '.SZ', '上交所': '.SH'})
        df.code = df.code + exchange
        renaming_dict = self._factor_param['可转债信息']
        df.company_code = df.company_code.apply(self.jqcode_to_windcode)
        df.list_date = df.list_date.apply(DateUtils.date_type2datetime)
        df.delist_Date = df.delist_Date.apply(DateUtils.date_type2datetime)
        df.company_code = df.company_code.apply(self.jqcode_to_windcode)
        ret = df.loc[:, renaming_dict.keys()].rename(renaming_dict, axis=1).set_index('ID')
        self.db_interface.update_df(ret, '可转债信息')
        print(df)

    def update_stock_minute_data(self):
        date = dt.datetime(2020, 11, 20)
        pre_date = date - dt.timedelta(days=1)
        date_start = date + dt.timedelta(hours=8)
        date_end = date + dt.timedelta(hours=16)
        tickers = self.stock_tickers.ticker(date)
        tickers = ['000001.XSHE']
        data = get_price(tickers, start_date=date_start, end_date=date_end, frequency='1m', fq=None, fill_paused=True)

        auction = get_call_auction(tickers, DateUtils.date_type2str(pre_date, '-'), DateUtils.date_type2str(date, '-'))

    @staticmethod
    def jqcode_to_windcode(jq_code: str) -> Optional[str]:
        if jq_code:
            jq_code = jq_code.replace('.XSHG', '.SH')
            jq_code = jq_code.replace('.XSHE', '.SZ')
            jq_code = jq_code.replace('.CCFX', '.CFE')
            jq_code = jq_code.replace('.XDCE', '.DCE')
            jq_code = jq_code.replace('.XSGE', '.SFE')
            jq_code = jq_code.replace('.XZCE', '.ZCE')
            jq_code = jq_code.replace('.XINE', '.INE')
            return jq_code
