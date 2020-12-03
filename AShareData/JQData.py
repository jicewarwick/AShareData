import datetime as dt
from typing import Mapping, Optional, Union

import pandas as pd
from cached_property import cached_property
from tqdm import tqdm

from . import DateUtils, utils
from .DataSource import DataSource
from .DBInterface import DBInterface
from .Tickers import StockTickers

with utils.NullPrinter():
    import jqdatasdk as jq


class JQData(DataSource):
    def __init__(self, db_interface: DBInterface, mobile: str, password: str):
        super().__init__(db_interface)
        self.mobile = mobile
        self.password = password
        self.is_logged_in = False
        self._factor_param = utils.load_param('jqdata_param.json')

    def connect(self):
        if not self.is_logged_in:
            with utils.NullPrinter():
                jq.auth(self.mobile, self.password)
            if jq.is_auth():
                self.is_logged_in = True
            else:
                raise ValueError('JQDataLoginError: Wrong mobile number or password')

    def __enter__(self):
        self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_logged_in:
            jq.logout()

    @cached_property
    def stock_tickers(self):
        return StockTickers(self.db_interface)

    def update_convertible_bond_list(self):
        q = jq.query(jq.bond.BOND_BASIC_INFO).filter(jq.bond.BOND_BASIC_INFO.bond_type == '可转债')
        df = jq.bond.run_query(q)
        exchange = df.exchange.map({'深交所主板': '.SZ', '上交所': '.SH'})
        df.code = df.code + exchange
        renaming_dict = self._factor_param['可转债信息']
        df.company_code = df.company_code.apply(self.jqcode2windcode)
        df.list_date = df.list_date.apply(DateUtils.date_type2datetime)
        df.delist_Date = df.delist_Date.apply(DateUtils.date_type2datetime)
        df.company_code = df.company_code.apply(self.jqcode2windcode)
        ret = df.loc[:, renaming_dict.keys()].rename(renaming_dict, axis=1).set_index('ID')
        self.db_interface.update_df(ret, '可转债信息')
        print(df)

    def _get_stock_minute_data_first_minute(self, date: dt.datetime):
        renaming_dict = self._factor_param['行情数据']
        diff_columns = ['成交量', '成交额']
        tickers = self.stock_tickers.ticker(date)
        tickers = [self.windcode2jqcode(it) for it in tickers]

        auction_time = date + dt.timedelta(hours=9, minutes=25)
        auction_data = self.db_interface.read_table('股票集合竞价数据', columns=['成交价', '成交量', '成交额'], dates=[auction_time])
        auction_db_data = self._auction_data_to_price_data(auction_data)

        first_minute = date + dt.timedelta(hours=9, minutes=31)
        real_first_minute = date + dt.timedelta(hours=9, minutes=30)
        first_minute_data = jq.get_price(tickers, start_date=first_minute, end_date=first_minute, frequency='1m',
                                         fq=None, fill_paused=True)
        first_minute_data = self._standardize_df(first_minute_data, renaming_dict)
        tmp = first_minute_data.loc[:, diff_columns].droplevel('DateTime').fillna(0) - \
              auction_db_data.loc[:, diff_columns].droplevel('DateTime').fillna(0)
        tmp['DateTime'] = real_first_minute
        tmp.set_index('DateTime', append=True, inplace=True)
        tmp.index = tmp.index.swaplevel()
        first_minute_db_data = pd.concat([first_minute_data.drop(diff_columns, axis=1), tmp], sort=True, axis=1)
        db_data = pd.concat([auction_db_data, first_minute_db_data], sort=True)
        self.db_interface.insert_df(db_data, '股票分钟行情')

    def _get_stock_minute_data_after_first_minute(self, date: dt.datetime):
        renaming_dict = self._factor_param['行情数据']
        tickers = self.stock_tickers.ticker(date)
        tickers = [self.windcode2jqcode(it) for it in tickers]

        t0932 = date + dt.timedelta(hours=9, minutes=32)
        t1458 = date + dt.timedelta(hours=14, minutes=58)
        t1459 = date + dt.timedelta(hours=14, minutes=59)
        t1500 = date + dt.timedelta(hours=15)

        data = jq.get_price(tickers, start_date=t0932, end_date=t1458, frequency='1m', fq=None, fill_paused=True)
        data.time = data.time.apply(lambda x: x - dt.timedelta(minutes=1))
        db_data = self._standardize_df(data, renaming_dict)
        self.db_interface.insert_df(db_data, '股票分钟行情')

        # SZ
        sz_tickers = [it for it in tickers if it.endswith('XSHE')]
        data = jq.get_price(sz_tickers, start_date=t1500, end_date=t1500, frequency='1m', fq=None, fill_paused=True)
        db_data = self._standardize_df(data, renaming_dict)
        self.db_interface.insert_df(db_data, '股票分钟行情')

        # SH
        sh_tickers = [it for it in tickers if it.endswith('XSHG')]
        data = jq.get_price(sh_tickers, start_date=t1459, end_date=t1500, frequency='1m', fq=None, fill_paused=True)
        data = data.loc[data.volume > 0, :]
        if t1459 in data.time.tolist():
            data.time = data.time.apply(lambda x: x - dt.timedelta(minutes=1))
        db_data = self._standardize_df(data, renaming_dict)
        self.db_interface.insert_df(db_data, '股票分钟行情')

    def get_stock_minute(self, date: dt.datetime):
        self._get_stock_minute_data_first_minute(date)
        self._get_stock_minute_data_after_first_minute(date)

    def update_stock_minute(self):
        table_name = '股票分钟行情'
        db_timestamp = self._check_db_timestamp(table_name, dt.datetime(2015, 1, 1))
        start_date = self.calendar.offset(db_timestamp.date(), 1)
        if dt.datetime.now().hour < 17:
            end_date = self.calendar.yesterday()
        else:
            end_date = dt.datetime.today()
        dates = self.calendar.select_dates(start_date, end_date)
        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'更新{date}的{table_name}')
                self.get_stock_minute(date)
                pbar.update()

    def stock_open_auction_data(self, date: DateUtils.DateType):
        table_name = '股票集合竞价数据'
        renaming_dict = self._factor_param[table_name]
        date_str = DateUtils.date_type2str(date, '-')
        tickers = self.stock_tickers.ticker(date)
        tickers = [self.windcode2jqcode(it) for it in tickers]
        data = jq.get_call_auction(tickers, start_date=date_str, end_date=date_str)
        auction_time = date + dt.timedelta(hours=9, minutes=25)
        data.time = auction_time
        db_data = self._standardize_df(data, renaming_dict)
        self.db_interface.insert_df(db_data, table_name)

    def update_option_daily(self):
        pass

    @staticmethod
    def _auction_data_to_price_data(auction_data: pd.DataFrame) -> pd.DataFrame:
        auction_data['开盘价'] = auction_data['成交价']
        auction_data['最高价'] = auction_data['成交价']
        auction_data['最低价'] = auction_data['成交价']
        auction_data['收盘价'] = auction_data['成交价']
        return auction_data.drop('成交价', axis=1)

    @staticmethod
    def _standardize_df(df: pd.DataFrame, parameter_info: Mapping[str, str]) -> Union[pd.Series, pd.DataFrame]:
        dates_columns = [it for it in df.columns if it.endswith('date') | it.endswith('time')]
        for it in dates_columns:
            df[it] = df[it].apply(DateUtils.date_type2datetime)

        df.rename(parameter_info, axis=1, inplace=True)
        if 'ID' in df.columns:
            df.ID = df.ID.apply(JQData.jqcode2windcode)
        index = sorted(list({'DateTime', 'ID', '报告期', 'IndexCode'} & set(df.columns)))
        df = df.set_index(index, drop=True)
        if df.shape[1] == 1:
            df = df.iloc[:, 0]
        return df

    @staticmethod
    def jqcode2windcode(jq_code: str) -> Optional[str]:
        if jq_code:
            jq_code = jq_code.replace('.XSHG', '.SH')
            jq_code = jq_code.replace('.XSHE', '.SZ')
            jq_code = jq_code.replace('.CCFX', '.CFE')
            jq_code = jq_code.replace('.XDCE', '.DCE')
            jq_code = jq_code.replace('.XSGE', '.SHF')
            jq_code = jq_code.replace('.XZCE', '.CZC')
            jq_code = jq_code.replace('.XINE', '.INE')
            return jq_code

    @staticmethod
    def windcode2jqcode(jq_code: str) -> Optional[str]:
        if jq_code:
            jq_code = jq_code.replace('.SH', '.XSHG')
            jq_code = jq_code.replace('.SZ', '.XSHE')
            jq_code = jq_code.replace('.CFE', '.CCFX')
            jq_code = jq_code.replace('.DCE', '.XDCE')
            jq_code = jq_code.replace('.SHF', '.XSGE')
            jq_code = jq_code.replace('.CZC', '.XZCE')
            jq_code = jq_code.replace('.INE', '.XINE')
            return jq_code
