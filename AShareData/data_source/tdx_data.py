import datetime as dt
from collections import OrderedDict
from typing import Sequence

import pandas as pd
from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams
from tqdm import tqdm

from .data_source import DataSource, MinutesDataFunctionMixin
from .. import utils
from ..config import get_global_config
from ..database_interface import DBInterface
from ..tickers import ConvertibleBondTickers, StockTickers


class TDXData(DataSource, MinutesDataFunctionMixin):
    def __init__(self, db_interface: DBInterface = None, host: str = None, port: int = None):
        super().__init__(db_interface)
        if host is None:
            conf = get_global_config()
            host = conf['tdx_server']['host']
            port = conf['tdx_server']['port']
        self.api = TdxHq_API()
        self.host = host
        self.port = port
        self._factor_param = utils.load_param('tdx_param.json')
        self.stock_ticker = StockTickers(db_interface)

    def login(self):
        self.api.connect(self.host, self.port)

    def logout(self):
        self.api.disconnect()

    def update_stock_minute(self):
        """更新股票分钟行情"""
        table_name = '股票分钟行情'
        db_timestamp = self.db_interface.get_latest_timestamp(table_name, dt.datetime(2015, 1, 1))
        start_date = self.calendar.offset(db_timestamp.date(), 1)
        end_date = dt.datetime.today()
        dates = self.calendar.select_dates(start_date, end_date)
        for date in dates:
            self.get_stock_minute(date)

    def get_stock_minute(self, date: dt.datetime) -> None:
        """获取 ``date`` 的股票分钟行情"""
        tickers = self.stock_ticker.ticker(date)
        minute_data = self.get_minute_data(date, tickers)
        auction_time = date + dt.timedelta(hours=9, minutes=25)
        auction_db_data = self.db_interface.read_table('股票集合竞价数据', columns=['成交价', '成交量', '成交额'], dates=auction_time)
        df = self.left_shift_minute_data(minute_data=minute_data, auction_db_data=auction_db_data)

        self.db_interface.insert_df(df, '股票分钟行情')

    def update_convertible_bond_minute(self):
        """更新可转债分钟行情"""
        table_name = '可转债分钟行情'
        cb_tickers = ConvertibleBondTickers(self.db_interface)

        db_timestamp = self.db_interface.get_latest_timestamp(table_name, dt.datetime(1998, 9, 2))
        start_date = self.calendar.offset(db_timestamp.date(), 1)
        end_date = dt.datetime.today()
        dates = self.calendar.select_dates(start_date, end_date)

        for date in dates:
            tickers = cb_tickers.ticker(date)
            minute_data = self.get_minute_data(date, tickers)
            self.db_interface.insert_df(minute_data, table_name)

    def get_minute_data(self, date: dt.datetime, tickers: Sequence[str]) -> pd.DataFrame:
        num_days = self.calendar.days_count(date, dt.date.today())
        start_index = num_days * 60 * 4

        storage = []
        with tqdm(tickers) as pbar:
            for ticker in tickers:
                pbar.set_description(f'下载 {ticker} 在 {date} 的分钟数据')
                code, market = self._split_ticker(ticker)
                data = self.api.get_security_bars(category=TDXParams.KLINE_TYPE_1MIN, market=market, code=code,
                                                  start=start_index, count=240)
                if data:
                    data = self._formatting_data(data, ticker)
                    storage.append(data)
                pbar.update()

        df = pd.concat(storage) if storage else pd.DataFrame()
        return df

    def _formatting_data(self, info: OrderedDict, ticker: str) -> pd.DataFrame:
        df = pd.DataFrame(info)
        df['datetime'] = df['datetime'].apply(self.str2datetime)
        df = df.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1).rename(self._factor_param['行情数据'], axis=1)
        df['ID'] = ticker

        df = df.set_index(['DateTime', 'ID'], drop=True)
        df = df.where(abs(df) > 0.0001, 0)
        return df

    @staticmethod
    def _split_ticker(ticker: str) -> [str, int]:
        code, market_str = ticker.split('.')
        market = TDXParams.MARKET_SZ if market_str == 'SZ' else TDXParams.MARKET_SH
        return code, market

    @staticmethod
    def str2datetime(date: str) -> dt.datetime:
        return dt.datetime.strptime(date, '%Y-%m-%d %H:%M')
