import datetime as dt
import logging
import sys
import tempfile
from typing import List

import WindPy
import pandas as pd

from AShareData.utils import DateType, date_type2datetime


class WindWrapper(object):
    def __init__(self):
        self._w = None

    def connect(self):
        with tempfile.TemporaryFile(mode='w') as log_file:
            out = sys.stdout
            out2 = sys.stderr
            sys.stdout = log_file
            sys.stderr = log_file
            try:
                self._w = WindPy.w
                self._w.start()
            except:
                logging.error('Wind API fail to start')
            finally:
                sys.stdout = out
                sys.stderr = out2

    def disconnect(self):
        if self._w:
            self._w.close()

    def is_connected(self):
        return self._w.isconnected()

    @staticmethod
    def _api_error(api_data):
        if isinstance(api_data, tuple):
            error_code = api_data[0]
            has_data = True
        else:
            error_code = api_data.ErrorCode
            data = api_data.Data
            has_data = any(data)

        if (error_code != 0) or (not has_data):
            raise ValueError(f"Failed to get data, ErrorCode: {error_code}, Error Message: {api_data[1].iloc[0, 0]}")

    @staticmethod
    def _standardize_date(date: DateType = None):
        if not date:
            date = dt.date.today()
        if isinstance(date, (dt.date, dt.datetime)):
            date = date.strftime('%Y-%m-%d')
        return date

    # wrap functions
    def wsd(self, *args, **kwargs) -> pd.DataFrame:
        data = self._w.wsd(*args, usedf=True, **kwargs)
        self._api_error(data)
        return data[1]

    def wss(self, *args, **kwargs) -> pd.DataFrame:
        data = self._w.wss(*args, usedf=True, **kwargs)
        self._api_error(data)
        return data[1]

    def wset(self, *args, **kwargs) -> pd.DataFrame:
        data = self._w.wset(*args, usedf=True, **kwargs)
        self._api_error(data)
        df = data[1]

        index_val = sorted(list({'date', 'wind_code'} & set(df.columns)))
        if index_val:
            df.set_index(index_val, drop=True, inplace=True)
        return df

    def tdays(self, *args, **kwargs) -> List[dt.datetime]:
        data = self._w.tdays(*args, **kwargs)
        self._api_error(data)
        return data.Data[0]

    def tdaysoffset(self, *args, **kwargs) -> dt.datetime:
        data = self._w.tdaysoffset(*args, **kwargs)
        self._api_error(data)
        return data.Data[0][0]

    def tdayscount(self, *args, **kwargs) -> int:
        data = self._w.tdayscount(*args, **kwargs)
        self._api_error(data)
        return data.Data[0][0]

    # outright functions
    def get_index_constitute(self, date: DateType = dt.date.today(),
                             index: str = '000300.SH') -> pd.DataFrame:
        date = date_type2datetime(date)
        data = self.wset('indexconstituent', date=date, windcode=index)
        return data
