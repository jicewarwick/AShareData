import datetime as dt
import logging
import re
import sys
import tempfile
from typing import List, Sequence, Union

import pandas as pd
import WindPy
from tqdm import tqdm

from . import constants, utils
from .DataSource import DataSource
from .DBInterface import DBInterface, get_stocks
from .TradingCalendar import TradingCalendar


class WindWrapper(object):
    """Wind Wrapper to make wind API easier to use"""

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
    def _standardize_date(date: utils.DateType = None):
        if not date:
            date = dt.date.today()
        if isinstance(date, (dt.date, dt.datetime)):
            date = date.strftime('%Y-%m-%d')
        return date

    @staticmethod
    def _to_df(out: WindPy.w.WindData) -> Union[pd.Series, pd.DataFrame]:
        times = [utils.date_type2datetime(it) for it in out.Times]
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

    # wrap functions
    def wsd(self, security, fields, startDate=None, endDate=None, options=None,
            *args, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        data = self._w.wsd(security, fields, startDate, endDate, options, *args, **kwargs)
        self._api_error(data)
        return self._to_df(data)

    def wss(self, *args, **kwargs) -> pd.DataFrame:
        data = self._w.wss(*args, usedf=True, **kwargs)
        self._api_error(data)
        return data[1]

    def wsi(self, *args, **kwargs) -> pd.DataFrame:
        data = self._w.wsi(*args, usedf=True, **kwargs)
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
    def get_index_constitute(self, date: utils.DateType = dt.date.today(),
                             index: str = '000300.SH') -> pd.DataFrame:
        date = utils.date_type2datetime(date)
        data = self.wset('indexconstituent', date=date, windcode=index)
        return data


class WindData(DataSource):
    """Wind 数据源"""
    _stock_trading_time_cut = [(dt.timedelta(hours=9, minutes=30), dt.timedelta(hours=10)),
                               (dt.timedelta(hours=10, minutes=1), dt.timedelta(hours=10, minutes=30)),
                               (dt.timedelta(hours=10, minutes=31), dt.timedelta(hours=11)),
                               (dt.timedelta(hours=11, minutes=1), dt.timedelta(hours=11, minutes=30)),
                               (dt.timedelta(hours=13), dt.timedelta(hours=13, minutes=30)),
                               (dt.timedelta(hours=13, minutes=31), dt.timedelta(hours=14)),
                               (dt.timedelta(hours=14, minutes=1), dt.timedelta(hours=14, minutes=30)),
                               (dt.timedelta(hours=14, minutes=31), dt.timedelta(hours=15))]

    def __init__(self, db_interface: DBInterface, param_json_loc: str = None):
        super().__init__(db_interface)

        self.calendar = TradingCalendar(db_interface)
        self.stocks = get_stocks(db_interface)
        self._factor_param = utils.load_param('wind_param.json', param_json_loc)
        self.w = WindWrapper()
        self.w.connect()

    def _get_industry_data(self, wind_code: Union[str, Sequence[str]], provider: str, date: dt.datetime) -> pd.Series:
        wind_data = self.w.wsd(wind_code, f'industry_{constants.INDUSTRY_DATA_PROVIDER_CODE_DICT[provider]}',
                               date, date, industryType=constants.INDUSTRY_LEVEL[provider])
        wind_data.name = '行业名称'
        wind_data = wind_data.str.replace('III|Ⅲ|IV|Ⅳ$', '')
        return wind_data

    def _find_industry(self, wind_code: str, provider: str,
                       start_date: utils.DateType, start_data: str,
                       end_date: utils.DateType, end_data: str) -> None:
        if start_data != end_data:
            logging.info(f'{wind_code} 的行业由 {start_date} 的 {start_data} 改为 {end_date} 的 {end_data}')
            while True:
                if self.calendar.days_count(start_date, end_date) < 2:
                    entry = pd.DataFrame({'DateTime': end_date, 'ID': wind_code, '行业名称': end_data}, index=[0])
                    entry.set_index(['DateTime', 'ID'], inplace=True)
                    logging.info(f'插入数据: {wind_code} 于 {end_date} 的行业为 {end_data}')
                    self.db_interface.update_df(entry, f'{provider}行业')
                    break
                mid_date = self.calendar.middle(start_date, end_date)
                logging.debug(f'查询{wind_code} 在 {mid_date}的行业')
                mid_data = self._get_industry_data(wind_code, provider, mid_date).iloc[0]
                if mid_data == start_data:
                    start_date = mid_date
                elif mid_data == end_data:
                    end_date = mid_date
                else:
                    self._find_industry(wind_code, provider, start_date, start_data, mid_date, mid_data)
                    self._find_industry(wind_code, provider, mid_date, mid_data, end_date, end_data)
                    break

    def update_industry(self, provider: str) -> None:
        """更新行业信息

        :param provider:
        """
        table_name = f'{provider}行业'
        query_date = self.calendar.offset(dt.date.today(), -1)
        latest = self.db_interface.get_latest_timestamp(table_name)
        if latest is None:
            latest = utils.date_type2datetime(constants.INDUSTRY_START_DATE[provider])
            initial_data = self._get_industry_data(self.stocks, provider, latest).dropna()
            self.db_interface.insert_df(initial_data, table_name)
        else:
            initial_data = self.db_interface.read_table(table_name)

        new_data = self._get_industry_data(self.stocks, provider, query_date).dropna()

        diff_stock = utils.compute_diff(new_data, initial_data)
        with tqdm(diff_stock) as pbar:
            for (_, stock), new_industry in diff_stock.iteritems():
                pbar.set_description(f'获取{stock}的{provider}行业')
                try:
                    old_industry = initial_data.loc[(slice(None), stock), '行业名称'].values[-1]
                except KeyError:
                    old_industry = None
                self._find_industry(stock, provider, latest, old_industry, query_date, new_industry)
                pbar.update(1)

    def _stock_get_minutes_data(self, start_time: dt.datetime, end_time: dt.datetime) -> None:
        table_name = '股票分钟行情'
        replace_dict = self._factor_param[table_name]

        data = self.w.wsi(self.stocks, "open,high,low,close,volume,amt", start_time, end_time, "")
        data.set_index('windcode', append=True, inplace=True)
        data.index.names = ['DateTime', 'ID']
        data.rename(replace_dict, axis=1, inplace=True)
        self.db_interface.insert_df(data, table_name)

        logging.debug(f'{start_time} - {end_time} 的分钟数据下载完成')

    def get_stock_minutes_data(self, query_date: utils.DateType, start_time: dt.timedelta = None) -> None:
        """从``query_date``的``start_time``开始获取股票分钟数据"""
        query_date = utils.date_type2datetime(query_date)
        assert self.calendar.is_trading_date(query_date.date()), f'{query_date} 非交易日!'

        logging.info(f'开始下载 {utils.date_type2str(query_date, "-")} 的分钟数据')
        for start_delta, end_delta in self._stock_trading_time_cut:
            if start_time is not None:
                if start_time > start_delta:
                    continue
            self._stock_get_minutes_data(query_date + start_delta, query_date + end_delta)

        logging.info(f'{utils.date_type2str(query_date, "-")} 分钟数据下载完成')

    def update_minutes_data(self) -> None:
        """股票分钟行情更新脚本"""
        table_name = '股票分钟行情'
        latest = self.db_interface.get_latest_timestamp(table_name)
        if latest is None:
            pass
        elif latest.hour != 15:
            self.get_stock_minutes_data(latest.date(), latest - dt.datetime.combine(latest.date(), dt.time(0, 0)))

        pre_date = self.calendar.offset(dt.date.today(), -1)
        while True:
            latest = self.calendar.offset(latest.date(), 1)
            if latest >= pre_date:
                break
            try:
                self.get_stock_minutes_data(latest)
            except ValueError as e:
                latest = self.db_interface.get_latest_timestamp(table_name)
                logging.info(f'股票分钟数据已更新至 {latest}')
                print(e)
                break

    def update_index_close(self):
        pass

    def get_future_symbols(self) -> List[str]:
        all_contracts = self.w.wset('sectorconstituent', date=dt.date.today(), sectorid='1000028001000000')
        IDs = all_contracts.index.get_level_values(1).tolist()
        symbols = [re.sub('\d*', '', it) for it in IDs]
        symbols = list(set(symbols))

        # delete simulated symbols
        symbols = [it for it in symbols if '-S' not in it]
        symbols = sorted(symbols)
        pd.DataFrame(symbols).to_sql('期货品种', self.db_interface.engine, if_exists='replace')
        return symbols

    def update_future_contracts_list(self):
        contract_table_name = '期货合约'
        start_date = self.db_interface.get_column(contract_table_name, '合约上市日期')
        if start_date is None:
            start_date = dt.date(1990, 1, 1)
        else:
            start_date = max(start_date)

        symbols_table_name = '期货品种'
        symbols_table = self.db_interface.read_table(symbols_table_name)
        symbols = symbols_table.iloc[:, 1].tolist()
        storage = []
        for symbol in symbols:
            storage.append(self.w.wset('futurecc',
                                       f'startdate={start_date.strftime("%Y-%m-%d")};enddate={dt.date.today().strftime("%Y-%m-%d")}',
                                       wind_code=symbol))
        data = pd.concat(storage)
        info = data.rename(self._factor_param[contract_table_name], axis=1)
        info.index.name = 'ID'
        info = info.drop_duplicates()
        self.db_interface.update_df(info, contract_table_name)

    def update_future_daily_data(self):
        contract_daily_table_name = '期货日行情'
        contract_table_name = '期货合约'
        start_date = self.db_interface.get_latest_timestamp(contract_daily_table_name)
        if start_date is None:
            start_date = self.db_interface.get_column(contract_table_name, '合约上市日期')
            start_date = min(start_date)
        end_date = dt.date.today() - dt.timedelta(days=1)
        dates = self.calendar.select_dates(start_date, end_date)
        if len(dates) <= 1:
            return
        dates = dates[1:]

        contract_table = self.db_interface.read_table(contract_table_name)
        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的{contract_daily_table_name}')
                tmp = contract_table.loc[(contract_table['合约上市日期'] <= date) & (contract_table['最后交易日'] >= date), :]
                contract_id = tmp.index.tolist()
                data = self.w.wss(contract_id, "open, high, low, close, settle, volume, amt, oi",
                                  f'tradeDate={date.strftime("%Y%m%d")};priceAdj=U;cycle=D')
                data.index.names = ['ID']
                data['DateTime'] = date
                data = data.set_index('DateTime', append=True)
                data = data.rename(self._factor_param[contract_daily_table_name], axis=1)
                self.db_interface.insert_df(data, contract_daily_table_name)
                pbar.update()

    def update_stock_option_list(self) -> None:
        contract_table_name = '期权合约'

        start_date = self.db_interface.get_column(contract_table_name, '上市日期')
        if start_date is None:
            start_date = dt.date(1990, 1, 1)
        else:
            start_date = max(start_date)
        end_date = dt.date.today() - dt.timedelta(days=1)
        option_dict = {'510050.SH': 'sse', '510300.SH': 'sse', '510500.SH': 'sse', '159919.SZ': 'szse',
                       '000300.SH': 'cffex'}
        exchange_dict = {'sse': 'SH', 'szse': 'SZ', 'cffex': 'CFE'}
        storage = []
        for underlying, exchange in option_dict.items():
            data = self.w.wset("optioncontractbasicinfo",
                               f"startdate={start_date.strftime('%Y-%m-%d')};enddate={end_date.strftime('%Y-%m-%d')};exchange={exchange};windcode={underlying};status=all;field=wind_code,sec_name,option_mark_code,call_or_put,exercise_price,contract_unit,limit_month,listed_date,exercise_date")
            index = ['.'.join([it, exchange_dict[exchange]]) for it in data.index]
            data.index = index
            storage.append(data)

        all_data = pd.concat(storage)
        all_data = all_data.rename(self._factor_param[contract_table_name], axis=1)
        all_data.index.names = ['ID']

        self.db_interface.update_df(all_data, contract_table_name)

    def update_stock_option_daily_data(self) -> None:
        contract_daily_table_name = '期权日行情'
        contract_table_name = '期权合约'
        start_date = self.db_interface.get_latest_timestamp(contract_daily_table_name)
        if start_date is None:
            contract_table_name = '期权合约'
            start_date = self.db_interface.get_column(contract_table_name, '上市日期')
            start_date = min(start_date)
        end_date = dt.date.today() - dt.timedelta(days=1)
        dates = self.calendar.select_dates(start_date, end_date)
        if len(dates) <= 1:
            return
        dates = dates[1:]

        contract_table = self.db_interface.read_table(contract_table_name)
        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的{contract_daily_table_name}')
                tmp = contract_table.loc[(contract_table['上市日期'] <= date) & (contract_table['行权日期'] >= date), :]
                contract_id = tmp.index.tolist()
                data = self.w.wss(contract_id, "high,open,low,close,volume,amt,oi,delta,gamma,vega,theta,rho",
                                  f'tradeDate={date.strftime("%Y%m%d")};priceAdj=U;cycle=D')
                data.index.names = ['ID']
                data['DateTime'] = date
                data = data.set_index('DateTime', append=True)
                data = data.rename(self._factor_param[contract_daily_table_name], axis=1)
                self.db_interface.insert_df(data, contract_daily_table_name)
                pbar.update()

    def update_target_stock_index_daily(self) -> None:
        table_name = '指数日行情'
        start_date = self.db_interface.get_latest_timestamp(table_name)
        end_date = dt.date.today() - dt.timedelta(days=1)
        dates = self.calendar.select_dates(start_date, end_date)
        if len(dates) <= 1:
            return
        dates = dates[1:]

        indexes = list(constants.STOCK_INDEXES.values())
        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的{table_name}')
                data = self.w.wss(indexes, "open,low,high,close,volume,amt", f'tradeDate={date.strftime("%Y%m%d")};priceAdj=U;cycle=D')
                data = self._standardize_wss_data(data, date, table_name)
                self.db_interface.insert_df(data, table_name)
                pbar.update()

    def update_etf_list(self) -> pd.DataFrame:
        etf_table_name = 'ETF上市日期'
        etfs = self.w.wset("sectorconstituent",
                           f"date={dt.date.today().strftime('%Y%m%d')};sectorid=1000009165000000;field=wind_code,sec_name")
        wind_codes = etfs.index.tolist()
        listed_date = self.w.wss(wind_codes, "fund_etflisteddate")
        etf_info = pd.concat([etfs, listed_date], axis=1)
        etf_info = etf_info.set_index('FUND_ETFLISTEDDATE', append=True)
        etf_info.index.names = ['ID', 'DateTime']
        etf_info.columns = ['证券名称']
        self.db_interface.update_df(etf_info, etf_table_name)
        return etf_info

    def update_etf_daily(self):
        table_name = '场内基金日行情'
        etf_table_name = 'ETF上市日期'
        adj_factor_table_name = '复权因子'
        funds_info = self.db_interface.read_table(etf_table_name).reset_index()

        start_date = self.db_interface.get_latest_timestamp(table_name)
        if start_date is None:
            start_date = funds_info.DateTime.min()

        end_date = dt.date.today() - dt.timedelta(days=1)
        dates = self.calendar.select_dates(start_date, end_date)
        if len(dates) <= 1:
            return
        dates = dates[1:]

        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的ETF日行情')
                wind_codes = funds_info.loc[funds_info.DateTime <= date, 'ID'].tolist()
                data = self.w.wss(wind_codes, "open,low,high,close,volume,amt,nav,unit_total", f'tradeDate={date.strftime("%Y%m%d")};priceAdj=U;cycle=D')
                data = self._standardize_wss_data(data, date, table_name)
                self.db_interface.insert_df(data, table_name)

                adj_data = self.w.wss(wind_codes, "NAV_adj", f'tradeDate={date.strftime("%Y%m%d")};priceAdj=U;cycle=D')
                adj_data = self._standardize_wss_data(adj_data, date, table_name)
                adj_data = adj_data.iloc[:, 0]
                self.db_interface.update_compact_df(adj_data, adj_factor_table_name)

                pbar.update()

    def _standardize_wss_data(self, data: pd.DataFrame, date: utils.DateType, table_name: str) -> pd.DataFrame:
        data.index.names = ['ID']
        data['DateTime'] = date
        data = data.set_index('DateTime', append=True)
        data = data.rename(self._factor_param[table_name], axis=1)
        return data

