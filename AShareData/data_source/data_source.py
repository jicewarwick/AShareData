import datetime as dt

import pandas as pd

from .. import date_utils
from ..config import get_db_interface
from ..database_interface import DBInterface


class DataSource(object):
    """Data Source Base Class"""

    def __init__(self, db_interface: DBInterface = None) -> None:
        self.db_interface = db_interface if db_interface else get_db_interface()
        self.calendar = date_utils.SHSZTradingCalendar(self.db_interface)

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logout()
        pass

    def login(self):
        pass

    def logout(self):
        pass


class MinutesDataFunctionMixin(object):
    @staticmethod
    def _auction_data_to_price_data(auction_data: pd.DataFrame) -> pd.DataFrame:
        auction_data['开盘价'] = auction_data['成交价']
        auction_data['最高价'] = auction_data['成交价']
        auction_data['最低价'] = auction_data['成交价']
        auction_data['收盘价'] = auction_data['成交价']
        return auction_data.drop('成交价', axis=1)

    @classmethod
    def left_shift_minute_data(cls, minute_data: pd.DataFrame, auction_db_data: pd.DataFrame) -> pd.DataFrame:
        auction_data = cls._auction_data_to_price_data(auction_db_data)

        date = minute_data.index.get_level_values('DateTime')[0].date()
        t0930 = dt.datetime.combine(date, dt.time(9, 30))
        t0931 = dt.datetime.combine(date, dt.time(9, 31))
        t1500 = dt.datetime.combine(date, dt.time(15, 0))

        # morning auction
        diff_columns = ['成交量', '成交额']
        first_min_data = minute_data.loc[minute_data.index.get_level_values('DateTime') == t0931, :]

        tmp = first_min_data.loc[:, diff_columns].droplevel('DateTime').fillna(0) - \
              auction_data.loc[:, diff_columns].droplevel('DateTime').fillna(0)
        tmp['DateTime'] = t0930
        tmp.set_index('DateTime', append=True, inplace=True)
        tmp.index = tmp.index.swaplevel()

        new_index = pd.MultiIndex.from_product([[t0930], first_min_data.index.get_level_values('ID')],
                                               names=['DateTime', 'ID'])
        first_min_data = first_min_data.drop(diff_columns, axis=1)
        first_min_data.index = new_index

        first_minute_db_data = pd.concat([first_min_data, tmp], sort=True, axis=1)

        # mid data
        mid_data = minute_data.reset_index()
        mid_data = mid_data.loc[(mid_data.DateTime < t1500) & (mid_data.DateTime > t0931), :]
        mid_data.DateTime = mid_data.DateTime - dt.timedelta(minutes=1)
        mid_data = mid_data.set_index(['DateTime', 'ID'], drop=True)

        # afternoon auction
        end_data = minute_data.loc[minute_data.index.get_level_values('DateTime') == t1500, :]

        # combine all
        storage = [auction_data, first_minute_db_data, mid_data, end_data]
        ret = pd.concat(storage)
        ret = ret.loc[ret['成交量'] >= 1, :]
        return ret
