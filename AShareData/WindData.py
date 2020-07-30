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
from .DBInterface import DBInterface, get_listed_stocks


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

    # wrapped functions
    def wsd(self, codes: Union[str, List[str]], fields: Union[str, List[str]],
            begin_time: Union[str, dt.datetime] = None,
            end_time: Union[str, dt.datetime] = None,
            options: str = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        data = self._w.wsd(codes, fields, begin_time, end_time, options, **kwargs)
        self._api_error(data)
        return self._to_df(data)

    def wss(self, codes: Union[str, List[str]], fields: Union[str, List[str]], options: str = '',
            trade_date: utils.DateType = None, **kwargs) -> pd.DataFrame:
        if trade_date:
            trade_date = utils.date_type2datetime(trade_date)
            options = f'tradeDate={trade_date.strftime("%Y%m%d")};' + options
        data = self._w.wss(codes, fields, options, usedf=True, **kwargs)
        self._api_error(data)
        ret_data = data[1]
        if trade_date:
            ret_data.index.names = ['ID']
            ret_data['DateTime'] = trade_date
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

        index_val = sorted(list({'date', 'wind_code'} & set(df.columns)))
        if index_val:
            df.set_index(index_val, drop=True, inplace=True)
        return df

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

        self._factor_param = utils.load_param('wind_param.json', param_json_loc)
        self.w = WindWrapper()
        self.w.connect()

    def update_routine(self):
        # stock
        self.update_stock_daily_data()
        self.update_adj_factor()

        # future
        self.update_future_contracts_list()
        self.update_future_daily_data()

        # option
        self.update_stock_option_list()
        self.update_stock_option_daily_data()

        # index
        self.update_target_stock_index_daily()

        # etf
        self.update_etf_list()
        self.update_etf_daily()
        return

    #######################################
    # stock funcs
    #######################################
    def get_stock_daily_data(self, trade_date: utils.DateType = None, start_date: utils.DateType = None,
                             end_date: utils.DateType = dt.datetime.now()) -> None:
        """更新每日行情, 写入数据库, 不返回

        行情信息包括: 开高低收, 量额, 复权因子, 股本

        :param trade_date: 交易日期
        :param start_date: 开始日期
        :param end_date: 结束日期

        交易日期查询一天, 开始结束日期查询区间. 二选一

        :return: None
        """
        if (not trade_date) & (not start_date):
            raise ValueError('trade_date 和 start_date 必填一个!')
        dates = [trade_date] if trade_date else self.calendar.select_dates(start_date, end_date)

        renaming_dict = self._factor_param['股票日线行情']

        with tqdm(dates) as pbar:
            for date in dates:
                current_date_str = utils.date_type2str(date)
                stocks = get_listed_stocks(self.db_interface, date)

                # price data
                pbar.set_description(f'下载{current_date_str}的日行情')
                price_df = self.w.wss(stocks, list(renaming_dict.keys()), trade_date=date,
                                      options='priceAdj=U;cycle=D;unit=1')
                price_df.rename(renaming_dict, axis=1, inplace=True)

                self.db_interface.update_df(price_df, '股票日行情')

                pbar.update(1)

    def update_stock_daily_data(self):
        start_date = self._check_db_timestamp('股票日行情', dt.date(1990, 12, 10))
        self.get_stock_daily_data(start_date=start_date)

    def _get_stock_minutes_data(self, start_time: dt.datetime, end_time: dt.datetime) -> None:
        table_name = '股票分钟行情'
        replace_dict = self._factor_param[table_name]

        data = self.w.wsi(self.all_stocks, "open,high,low,close,volume,amt", start_time, end_time, "")
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
            self._get_stock_minutes_data(query_date + start_delta, query_date + end_delta)

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

    def update_adj_factor(self):
        current_data = self.db_interface.read_table('复权因子', '复权因子').groupby('ID').tail(1)
        end_date = self.calendar.yesterday()
        stock_adj_factor = self.w.wss(self.all_stocks, 'adjfactor', trade_date=end_date).iloc[:, 0]
        stock_adj_factor.name = '复权因子'

        def data_func(date: utils.DateType, ticker: str) -> pd.Series:
            data = self.w.wsd(ticker, 'adjfactor', date, date)
            data.name = '复权因子'
            return data

        current_stock_data = current_data.loc[slice(None), self.all_stocks, :]
        self.sparse_data_queryer(data_func, current_stock_data, stock_adj_factor, '更新股票复权因子')

        funds_info = self.db_interface.read_table('ETF上市日期').reset_index()
        wind_code = funds_info['ID'].tolist()
        fund_adj_factor = self.w.wss(wind_code, 'adjfactor', trade_date=end_date).iloc[:, 0]
        fund_adj_factor.name = '复权因子'

        current_fund_data = current_data.loc[slice(None), wind_code, :]
        self.sparse_data_queryer(data_func, current_fund_data, fund_adj_factor, '更新基金复权因子')

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

    def _update_industry(self, provider: str) -> None:
        """更新行业信息

        :param provider:
        """
        table_name = f'{provider}行业'
        query_date = self.calendar.yesterday()
        latest = self.db_interface.get_latest_timestamp(table_name)
        if latest is None:
            latest = utils.date_type2datetime(constants.INDUSTRY_START_DATE[provider])
            initial_data = self._get_industry_data(self.all_stocks, provider, latest).dropna()
            self.db_interface.insert_df(initial_data, table_name)
        else:
            initial_data = self.db_interface.read_table(table_name)

        new_data = self._get_industry_data(self.all_stocks, provider, query_date).dropna()

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

    def update_industry(self) -> None:
        needed_update_provider = []
        for provider in constants.INDUSTRY_DATA_PROVIDER:
            timestamp = self.db_interface.get_latest_timestamp(f'{provider}行业')
            if timestamp < dt.datetime.now() - dt.timedelta(days=30):
                needed_update_provider.append(provider)

        for provider in needed_update_provider:
            self._update_industry(provider)

    #######################################
    # etf funcs
    #######################################
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
                data = self.w.wss(wind_codes, "open,low,high,close,volume,amt,nav,unit_total", trade_date=date,
                                  priceAdj='U', cycle='D')
                data.rename(self._factor_param[table_name], axis=1, inplace=True)
                data.dropna(inplace=True)
                self.db_interface.insert_df(data, table_name)
                pbar.update()

    #######################################
    # future funcs
    #######################################
    def get_future_symbols(self) -> List[str]:
        all_contracts = self.w.wset('sectorconstituent', date=dt.date.today(), sectorid='1000028001000000')
        IDs = all_contracts.index.get_level_values(1).tolist()
        symbols = [re.sub(r'\d*', '', it) for it in IDs]
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

        logging.info(f'更新自{start_date.strftime("%Y-%m-%d")}以来的期货品种.')
        symbols_table_name = '期货品种'
        symbols_table = self.db_interface.read_table(symbols_table_name)
        symbols = symbols_table.iloc[:, 1].tolist()
        storage = []
        for symbol in symbols:
            storage.append(self.w.wset('futurecc', startdate=start_date, enddate=dt.date.today(), wind_code=symbol))
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
                data = self.w.wss(contract_id, "open, high, low, close, settle, volume, amt, oi", trade_date=date,
                                  options='priceAdj=U;cycle=D')
                data.rename(self._factor_param[contract_daily_table_name], axis=1, inplace=True)
                self.db_interface.insert_df(data, contract_daily_table_name)
                pbar.update()

    #######################################
    # option funcs
    #######################################
    def update_stock_option_list(self) -> None:
        contract_table_name = '期权合约'

        start_date = self.db_interface.get_column(contract_table_name, '上市日期')
        if start_date is None:
            start_date = dt.date(1990, 1, 1)
        else:
            start_date = max(start_date)
        end_date = dt.date.today() - dt.timedelta(days=1)
        storage = []
        option_dict = {'510050.SH': 'sse', '510300.SH': 'sse', '510500.SH': 'sse', '159919.SZ': 'szse',
                       '000300.SH': 'cffex'}
        exchange_dict = {'sse': 'SH', 'szse': 'SZ', 'cffex': 'CFE'}
        fields = "wind_code,sec_name,option_mark_code,call_or_put,exercise_price,contract_unit,limit_month,listed_date,exercise_date"
        logging.info('更新期权合约.')
        for underlying, exchange in option_dict.items():
            data = self.w.wset("optioncontractbasicinfo", exchange=exchange, windcode=underlying, status='all',
                               startdate=start_date.strftime('%Y-%m-%d'), enddate=end_date.strftime('%Y-%m-%d'),
                               options=f'field={fields}')
            index = ['.'.join([it, exchange_dict[exchange]]) for it in data.index]
            data.index = index
            storage.append(data)

        all_data = pd.concat(storage)
        all_data.rename(self._factor_param[contract_table_name], axis=1, inplace=True)
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
                                  trade_date=date, priceAdj='U', cycle='D')
                data.rename(self._factor_param[contract_daily_table_name], axis=1, inplace=True)
                self.db_interface.insert_df(data, contract_daily_table_name)
                pbar.update()

    #######################################
    # index funcs
    #######################################
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
                data = self.w.wss(indexes, "open,low,high,close,volume,amt", trade_date=date, priceAdj='U', cycle='D')
                data.rename(self._factor_param[table_name], axis=1, inplace=True)
                self.db_interface.insert_df(data, table_name)
                pbar.update()

    #######################################
    # helper funcs
    #######################################
    def sparse_data_queryer(self, data_func, start_series: pd.Series = None, end_series: pd.Series = None,
                            desc: str = ''):
        default_start_date = dt.date(1990, 12, 19)
        start_stocks = [] if start_series.empty else start_series.index.get_level_values('ID')

        all_stocks = sorted(list(set(start_stocks) | set(end_series.index.get_level_values('ID'))))

        tmp = start_series.reset_index().set_index('ID').reindex(all_stocks)
        tmp['DateTime'] = tmp['DateTime'].where(tmp.DateTime.notnull(), default_start_date)
        start_series = tmp.reset_index().set_index(['DateTime', 'ID']).iloc[:, 0]

        end_index = pd.MultiIndex.from_product([[end_series.index.get_level_values('DateTime')[0]], all_stocks],
                                               names=['DateTime', 'ID'])
        end_series = end_series.reindex(end_index)

        with tqdm(start_series) as pbar:
            for i in range(start_series.shape[0]):
                new_val = end_series.iloc[i:i + 1]
                old_val = start_series.iloc[i:i + 1]
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
                mid_data = data_func(mid_date, ticker)
                self._binary_data_queryer(data_func, start_data, mid_data)
                self._binary_data_queryer(data_func, mid_data, end_data)
