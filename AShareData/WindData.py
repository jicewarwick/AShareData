import datetime as dt
from typing import Dict, List, Sequence, Union

import numpy as np
import pandas as pd
import WindPy
from cached_property import cached_property
from tqdm import tqdm

from . import constants, DateUtils, utils
from .DataSource import DataSource
from .DBInterface import DBInterface
from .Tickers import ConvertibleBondTickers, ETFOptionTickers, ETFTickers, FutureTickers, IndexOptionTickers, \
    StockTickers


class WindWrapper(object):
    """Wind Wrapper to make wind API easier to use"""

    def __init__(self):
        self._w = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        with utils.NullPrinter():
            self._w = WindPy.w
            self._w.start()

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
    def _standardize_date(date: DateUtils.DateType = None):
        if not date:
            date = dt.date.today()
        if isinstance(date, (dt.date, dt.datetime)):
            date = date.strftime('%Y-%m-%d')
        return date

    @staticmethod
    def _to_df(out: WindPy.w.WindData) -> Union[pd.Series, pd.DataFrame]:
        times = DateUtils.date_type2datetime(out.Times)
        df = pd.DataFrame(out.Data).T
        if len(out.Times) > 1:
            df.index = times
            if len(out.Fields) >= len(out.Codes):
                df.columns = out.Fields
                df['ID'] = out.Codes[0]
                df.set_index('ID', append=True, inplace=True)
            else:
                df.columns = out.Codes
                df = df.stack()
                df.name = out.Fields[0]
        else:
            df.index = out.Codes
            df.columns = out.Fields
            df['DateTime'] = times[0]
            df = df.set_index(['DateTime'], append=True).swaplevel()
        df.index.names = ['DateTime', 'ID']
        if isinstance(df, pd.DataFrame) and (df.shape[1] == 1):
            df = df.iloc[:, 0]
        return df

    # wrapped functions
    def wsd(self, codes: Union[str, List[str]], fields: Union[str, List[str]],
            begin_time: Union[str, dt.datetime] = None,
            end_time: Union[str, dt.datetime] = None,
            options: str = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        data = self._w.wsd(codes, fields, begin_time, end_time, options, **kwargs)
        self._api_error(data)
        return self._to_df(data)

    @DateUtils.dtlize_input_dates
    def wss(self, codes: Union[str, List[str]], fields: Union[str, List[str]], options: str = '',
            date: DateUtils.DateType = None, **kwargs) -> pd.DataFrame:
        if date:
            options = f'tradeDate={date.strftime("%Y%m%d")};' + options
        data = self._w.wss(codes, fields, options, usedf=True, **kwargs)
        self._api_error(data)
        ret_data = data[1]
        if date:
            ret_data.index.names = ['ID']
            ret_data['DateTime'] = date
            ret_data = ret_data.reset_index().set_index(['DateTime', 'ID'])

        return ret_data

    def wsi(self, codes: Union[str, List[str]], fields: Union[str, List[str]],
            begin_time: Union[str, dt.datetime] = None,
            end_time: Union[str, dt.datetime] = None,
            options: str = None) -> pd.DataFrame:
        data = self._w.wsi(codes, fields, begin_time, end_time, options, usedf=True)
        self._api_error(data)
        return data[1]

    def wst(self, codes: Union[str, List[str]], fields: Union[str, List[str]],
            begin_time: Union[str, dt.datetime] = None,
            end_time: Union[str, dt.datetime] = None,
            options: str = None, **kwargs) -> pd.DataFrame:
        data = self._w.wsi(codes, fields, begin_time, end_time, options, usedf=True, **kwargs)
        self._api_error(data)
        return data[1]

    def wset(self, table_name: str, options: str = '', **kwargs) -> pd.DataFrame:
        data = self._w.wset(table_name, options, usedf=True, **kwargs)
        self._api_error(data)
        df = data[1]

        df.rename({'date': 'DateTime', 'wind_code': 'ID'}, axis=1, inplace=True)
        index_val = sorted(list({'DateTime', 'ID'} & set(df.columns)))
        if index_val:
            df.set_index(index_val, drop=True, inplace=True)
        return df

    def wsq(self, codes: Union[str, List[str]], fields: Union[str, List[str]]) -> pd.DataFrame:
        data = self._w.wsq(codes, fields, usedf=True)
        self._api_error(data)
        return data[1]

    # outright functions
    def get_index_constitute(self, date: DateUtils.DateType = dt.date.today(),
                             index: str = '000300.SH') -> pd.DataFrame:
        date = DateUtils.date_type2datetime(date)
        data = self.wset('indexconstituent', date=date, windcode=index)
        return data


class WindData(DataSource):
    """Wind 数据源"""

    def __init__(self, db_interface: DBInterface, param_json_loc: str = None):
        super().__init__(db_interface)

        self._factor_param = utils.load_param('wind_param.json', param_json_loc)
        self.w = WindWrapper()

    def __enter__(self):
        self.w.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.w.disconnect()

    def connect(self):
        self.w.connect()

    @cached_property
    def stock_list(self) -> StockTickers:
        return StockTickers(self.db_interface)

    @cached_property
    def future_list(self) -> FutureTickers:
        return FutureTickers(self.db_interface)

    @cached_property
    def option_list(self) -> IndexOptionTickers:
        return IndexOptionTickers(self.db_interface)

    @cached_property
    def stock_index_option_list(self) -> IndexOptionTickers:
        return IndexOptionTickers(self.db_interface)

    @cached_property
    def etf_option_list(self) -> ETFOptionTickers:
        return ETFOptionTickers(self.db_interface)

    @cached_property
    def etf_list(self):
        return ETFTickers(self.db_interface)

    @cached_property
    def convertible_bond_list(self):
        return ConvertibleBondTickers(self.db_interface)

    #######################################
    # stock funcs
    #######################################
    def get_stock_daily_data(self, date: DateUtils.DateType) -> None:
        """更新每日行情, 写入数据库, 不返回

        行情信息包括: 开高低收量额

        :param date: 交易日期
        :return: None
        """
        table_name = '股票日行情'
        renaming_dict = self._factor_param[table_name]
        price_df = self.w.wss(self.stock_list.ticker(date), list(renaming_dict.keys()), date=date,
                              options='priceAdj=U;cycle=D;unit=1')
        price_df.rename(renaming_dict, axis=1, inplace=True)
        self.db_interface.update_df(price_df, table_name)

    def get_stock_rt_price(self):
        tickers = self.stock_list.ticker(dt.date.today())
        storage = []
        for ticker in utils.chunk_list(tickers, 3000):
            storage.append(self.w.wsq(ticker, 'rt_latest'))
        data = pd.concat(storage)
        data.index.names = ['ID']
        data.columns = ['最新价']
        data['DateTime'] = dt.datetime.now()
        data.set_index('DateTime', append=True, inplace=True)
        self.db_interface.purge_table('股票最新价')
        self.db_interface.insert_df(data, '股票最新价')

    def update_stock_daily_data(self):
        table_name = '股票日行情'
        start_date = self._check_db_timestamp(table_name, dt.date(1990, 12, 10))
        dates = self.calendar.select_dates(start_date, dt.date.today(), inclusive=(False, True))
        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的{table_name}')
                self.get_stock_daily_data(date)
                pbar.update()

    def get_stock_minute_data(self, date: dt.datetime):
        table_name = '股票分钟行情'
        replace_dict = self._factor_param[table_name]

        start_time = dt.datetime.combine(date.date(), dt.time(hour=8))
        end_time = dt.datetime.combine(date.date(), dt.time(hour=16))
        storage = []
        for section in utils.chunk_list(self.stock_list.ticker(date), 100):
            partial_data = self.w.wsi(section, "open,high,low,close,volume,amt", start_time, end_time, "")
            storage.append(partial_data.dropna())
        data = pd.concat(storage)
        data.set_index('windcode', append=True, inplace=True)
        data.index.names = ['DateTime', 'ID']
        data.rename(replace_dict, axis=1, inplace=True)
        self.db_interface.insert_df(data, table_name)

    def update_minutes_data(self) -> None:
        """股票分钟行情更新脚本"""
        table_name = '股票分钟行情'
        latest = self._check_db_timestamp(table_name, dt.datetime.today() - dt.timedelta(days=365 * 3))
        date_range = self.calendar.select_dates(latest.date(), dt.date.today(), inclusive=(False, True))

        with tqdm(date_range) as pbar:
            for date in date_range:
                pbar.set_description(f'更新{date}的{table_name}')
                self.get_stock_minute_data(date)
                pbar.update()

    def update_stock_adj_factor(self):
        def data_func(ticker: str, date: DateUtils.DateType) -> pd.Series:
            data = self.w.wsd(ticker, 'adjfactor', date, date)
            data.name = '复权因子'
            return data

        self.sparse_data_template('复权因子', data_func)

    def update_stock_units(self):
        # 流通股本
        def float_a_func(ticker: str, date: DateUtils.DateType) -> pd.Series:
            data = self.w.wsd(ticker, "float_a_shares", date, date, "unit=1")
            data.name = 'A股流通股本'
            return data

        self.sparse_data_template('A股流通股本', float_a_func)

        # 自由流通股本
        def free_float_a_func(ticker: str, date: DateUtils.DateType) -> pd.Series:
            data = self.w.wsd(ticker, "free_float_shares", date, date, "unit=1")
            data.name = '自由流通股本'
            return data

        self.sparse_data_template('自由流通股本', free_float_a_func)

        # 总股本
        def total_share_func(ticker: str, date: DateUtils.DateType) -> pd.Series:
            data = self.w.wsd(ticker, "total_shares", date, date, "unit=1")
            data.name = '总股本'
            return data

        self.sparse_data_template('总股本', total_share_func)

        # A股总股本
        def total_a_share_func(ticker: str, date: DateUtils.DateType) -> pd.Series:
            data = self.w.wsd(ticker, "share_totala", date, date, "unit=1")
            data.name = 'A股总股本'
            return data

        self.sparse_data_template('A股总股本', total_a_share_func)

    def _update_industry(self, provider: str) -> None:
        """更新行业信息

        :param provider: 行业分类提供商
        """

        def _get_industry_data(ticker: Union[str, List[str]], date: dt.datetime) -> pd.Series:
            wind_data = self.w.wsd(ticker, f'industry_{constants.INDUSTRY_DATA_PROVIDER_CODE_DICT[provider]}',
                                   date, date, industryType=constants.INDUSTRY_LEVEL[provider])
            wind_data.name = f'{provider}行业'
            wind_data = wind_data.str.replace('III|Ⅲ|IV|Ⅳ$', '')
            return wind_data

        table_name = f'{provider}行业'
        query_date = self.calendar.yesterday()
        latest = self.db_interface.get_latest_timestamp(table_name)
        if latest is None:
            latest = DateUtils.date_type2datetime(constants.INDUSTRY_START_DATE[provider])
            initial_data = self.w.wss(self.stock_list.ticker(latest),
                                      f'industry_{constants.INDUSTRY_DATA_PROVIDER_CODE_DICT[provider]}',
                                      date=latest).dropna()
            self.db_interface.insert_df(initial_data, table_name)
        else:
            initial_data = self.db_interface.read_table(table_name).groupby('ID').tail(1)

        new_data = _get_industry_data(ticker=self.stock_list.ticker(), date=query_date).dropna()

        default_start_date = self.stock_list.list_date()
        for ticker, date in default_start_date.items():
            if date < constants.INDUSTRY_START_DATE[provider]:
                default_start_date[ticker] = constants.INDUSTRY_START_DATE[provider]

        self.sparse_data_queryer(_get_industry_data, initial_data, new_data, f'更新{table_name}',
                                 default_start_date=default_start_date)

    def update_industry(self) -> None:
        for provider in constants.INDUSTRY_DATA_PROVIDER:
            self._update_industry(provider)

    def update_pause_stock_info(self):
        table_name = '股票停牌'
        start_date = self._check_db_timestamp(table_name, dt.date(1990, 12, 10)) + dt.timedelta(days=1)
        end_date = self.calendar.yesterday()
        chunks = self.calendar.split_to_chunks(start_date, end_date, 20)

        renaming_dict = self._factor_param[table_name]
        with tqdm(chunks) as pbar:
            pbar.set_description('下载股票停牌数据')
            for range_start, range_end in chunks:
                start_date_str = range_start.strftime("%Y%m%d")
                end_date_str = range_end.strftime("%Y%m%d")
                pbar.set_postfix_str(f'{start_date_str} - {end_date_str}')
                data = self.w.wset("tradesuspend",
                                   f'startdate={start_date_str};enddate={end_date_str};field=date,wind_code,suspend_type,suspend_reason')
                data.rename(renaming_dict, axis=1, inplace=True)
                ind1 = (data['停牌类型'] == '盘中停牌') & (data['停牌原因'].str.startswith('股票价格'))
                ind2 = (data['停牌原因'].str.startswith('盘中'))
                data = data.loc[(~ind1) & (~ind2), :]
                self.db_interface.insert_df(data, table_name)
                pbar.update()

    #######################################
    # convertible bond funcs
    #######################################
    def update_convertible_bond_daily_data(self):
        table_name = '可转债日行情'
        renaming_dict = self._factor_param['股票日行情']
        start_date = self._check_db_timestamp(table_name, dt.datetime(1993, 2, 9))
        dates = self.calendar.select_dates(start_date, dt.date.today(), inclusive=(False, True))

        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的{table_name}')
                tickers = self.convertible_bond_list.ticker(date)
                if tickers:
                    data = self.w.wss(tickers, "open, high, low, close, volume, amt",
                                      date=date, options='priceAdj=U;cycle=D')
                    data.rename(renaming_dict, axis=1, inplace=True)
                    self.db_interface.insert_df(data, table_name)
                pbar.update()

    #######################################
    # future funcs
    #######################################
    def update_future_daily_data(self):
        contract_daily_table_name = '期货日行情'

        start_date = self.db_interface.get_latest_timestamp(contract_daily_table_name)
        dates = self.calendar.select_dates(start_date, dt.date.today(), inclusive=(False, True))

        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的{contract_daily_table_name}')
                data = self.w.wss(self.future_list.ticker(date), "open, high, low, close, settle, volume, amt, oi",
                                  date=date, options='priceAdj=U;cycle=D')
                data.rename(self._factor_param[contract_daily_table_name], axis=1, inplace=True)
                self.db_interface.insert_df(data, contract_daily_table_name)
                pbar.update()

    #######################################
    # option funcs
    #######################################
    def get_stock_option_daily_data(self, date: dt.datetime) -> None:
        contract_daily_table_name = '期权日行情'

        tickers = self.etf_option_list.ticker(date) + self.option_list.ticker(date)
        data = self.w.wss(tickers,
                          "high,open,low,close,volume,amt,oi,delta,gamma,vega,theta,rho",
                          date=date, priceAdj='U', cycle='D')
        data.rename(self._factor_param[contract_daily_table_name], axis=1, inplace=True)
        self.db_interface.insert_df(data, contract_daily_table_name)

    def update_stock_option_daily_data(self) -> None:
        contract_daily_table_name = '期权日行情'

        start_date = self._check_db_timestamp(contract_daily_table_name, dt.datetime(2015, 2, 8))
        dates = self.calendar.select_dates(start_date, dt.date.today(), inclusive=(False, True))

        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的{contract_daily_table_name}')
                self.get_stock_option_daily_data(date)
                pbar.update()

    #######################################
    # index funcs
    #######################################
    def update_target_stock_index_daily(self) -> None:
        table_name = '指数日行情'

        start_date = self.db_interface.get_latest_timestamp(table_name)
        dates = self.calendar.select_dates(start_date, dt.date.today(), inclusive=(False, True))

        indexes = list(constants.STOCK_INDEXES.values())
        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的{table_name}')
                indicators = "open,low,high,close,volume,amt,mkt_cap_ard,total_shares,float_a_shares,free_float_shares,pe_ttm"
                data = self.w.wss(indexes, indicators, date=date, priceAdj='U', cycle='D')
                data.rename(self._factor_param[table_name], axis=1, inplace=True)
                self.db_interface.insert_df(data, table_name)
                pbar.update()

    #######################################
    # helper funcs
    #######################################
    def sparse_data_queryer(self, data_func, start_series: pd.Series = None, end_series: pd.Series = None,
                            desc: str = '', default_start_date: Union[Dict, DateUtils.DateType] = None):
        start_ticker = [] if start_series.empty else start_series.index.get_level_values('ID')
        all_ticker = sorted(list(set(start_ticker) | set(end_series.index.get_level_values('ID'))))

        tmp = start_series.reset_index().set_index('ID').reindex(all_ticker)
        start_series = tmp.reset_index().set_index(['DateTime', 'ID']).iloc[:, 0]

        end_index = pd.MultiIndex.from_product([[end_series.index.get_level_values('DateTime')[0]], all_ticker],
                                               names=['DateTime', 'ID'])
        end_series = end_series.reindex(end_index)

        ind = np.logical_not(start_series.isnull().values & end_series.isnull().values)
        start_series = start_series.loc[ind, :]
        end_series = end_series.loc[ind, :]

        if start_series.dtype == 'float64':
            ind = np.abs(start_series.values - end_series.values) > 0.0001
            ind = ind | start_series.isnull().values | end_series.isnull().values
            ind = ind & (start_series.values != 0)
        else:
            ind = (start_series.values != end_series.values)
        start_series = start_series.loc[ind]
        end_series = end_series.loc[ind, :]

        with tqdm(start_series) as pbar:
            for i in range(start_series.shape[0]):
                new_val = end_series.iloc[i:i + 1]
                old_val = start_series.iloc[i:i + 1]
                if np.isnan(old_val.index.get_level_values('DateTime').values[0]):
                    ticker = old_val.index.get_level_values('ID').values[0]
                    if isinstance(default_start_date, dict):
                        index_date = default_start_date[ticker]
                    else:
                        index_date = DateUtils.date_type2datetime(default_start_date)
                    old_val = data_func(ticker=ticker, date=index_date.date())
                    self.db_interface.update_df(old_val.to_frame(), old_val.name)
                pbar.set_description(f'{desc}: {new_val.index.get_level_values("ID").values[0]}')
                self._binary_data_queryer(data_func, old_val, new_val)
                pbar.update(1)

    def _binary_data_queryer(self, data_func, start_data: pd.Series, end_data: pd.Series) -> None:
        if start_data.dtype == 'float64':
            if all(start_data.notnull()) and all(end_data.notnull()) and abs(
                    start_data.values[0] - end_data.values[0]) < 0.001:
                is_diff = False
            else:
                is_diff = True
        else:
            is_diff = start_data.values[0] != end_data.values[0]
        if is_diff:
            start_date = start_data.index.get_level_values('DateTime')[0]
            end_date = end_data.index.get_level_values('DateTime')[0]
            if self.calendar.days_count(start_date, end_date) < 2:
                self.db_interface.update_df(end_data.to_frame(), end_data.name)
            else:
                ticker = end_data.index.get_level_values('ID')[0]
                mid_date = self.calendar.middle(start_date, end_date)
                mid_data = data_func(ticker=ticker, date=mid_date)
                self._binary_data_queryer(data_func, start_data, mid_data)
                self._binary_data_queryer(data_func, mid_data, end_data)

    def sparse_data_template(self, table_name: str, data_func, ticker: Sequence[str] = None,
                             default_start_date: Union[Dict, DateUtils.DateType] = None):
        if default_start_date is None:
            default_start_date = self.stock_list.list_date()
        if ticker is None:
            ticker = self.stock_list.all_ticker()
        current_data = self.db_interface.read_table(table_name).groupby('ID').tail(1)
        current_data = current_data.loc[current_data.index.get_level_values('ID').isin(ticker), :]
        end_date = self.calendar.yesterday()
        new_data = data_func(ticker=ticker, date=end_date)
        new_data.name = table_name

        self.sparse_data_queryer(data_func, current_data, new_data, f'更新{table_name}',
                                 default_start_date=default_start_date)
