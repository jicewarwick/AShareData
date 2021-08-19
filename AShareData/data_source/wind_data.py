import datetime as dt
import re
from functools import cached_property
from typing import Dict, List, Sequence, Union

import numpy as np
import pandas as pd
import WindPy
from tqdm import tqdm

from .data_source import DataSource
from .. import algo, config, constants, date_utils, utils
from ..database_interface import DBInterface
from ..tickers import ConvertibleBondTickers, ETFOptionTickers, FutureTickers, IndexOptionTickers, \
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
            raise ValueError(f'Failed to get data, ErrorCode: {error_code}, Error Message: {api_data[1].iloc[0, 0]}')

    @staticmethod
    def _standardize_date(date: date_utils.DateType = None):
        if date is None:
            date = dt.date.today()
        if isinstance(date, (dt.date, dt.datetime)):
            date = date.strftime('%Y-%m-%d')
        return date

    @staticmethod
    def _to_df(out: WindPy.w.WindData) -> Union[pd.Series, pd.DataFrame]:
        times = date_utils.date_type2datetime(out.Times)
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

    @date_utils.dtlize_input_dates
    def wss(self, codes: Union[str, List[str]], fields: Union[str, List[str]], options: str = '',
            date: date_utils.DateType = None, **kwargs) -> pd.DataFrame:
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


class WindData(DataSource):
    """Wind 数据源"""

    def __init__(self, db_interface: DBInterface = None, param_json_loc: str = None):
        super().__init__(db_interface)

        self._factor_param = utils.load_param('wind_param.json', param_json_loc)
        self.w = WindWrapper()

    def login(self):
        self.w.connect()

    def logout(self):
        self.w.disconnect()

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
    def convertible_bond_list(self):
        return ConvertibleBondTickers(self.db_interface)

    #######################################
    # stock funcs
    #######################################
    def get_stock_daily_data(self, date: date_utils.DateType) -> None:
        """获取 ``date`` 的股票日行情, 包括: 开高低收量额

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
        """更新股票最新价, 将替换数据库内原先所有数据"""
        tickers = self.stock_list.ticker(dt.date.today())
        storage = []
        for ticker in algo.chunk_list(tickers, 3000):
            storage.append(self.w.wsq(ticker, 'rt_latest'))
        data = pd.concat(storage)
        data.index.names = ['ID']
        data.columns = ['最新价']
        data['DateTime'] = dt.datetime.now()
        data.set_index('DateTime', append=True, inplace=True)
        self.db_interface.purge_table('股票最新价')
        self.db_interface.insert_df(data, '股票最新价')

    def update_stock_daily_data(self):
        """更新股票日行情"""
        table_name = '股票日行情'
        start_date = self.db_interface.get_latest_timestamp(table_name, dt.date(1990, 12, 10))
        dates = self.calendar.select_dates(start_date, dt.date.today(), inclusive=(False, True))
        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的{table_name}')
                self.get_stock_daily_data(date)
                pbar.update()

    def update_stock_minutes_data(self) -> None:
        """更新股票分钟行情"""
        table_name = '股票分钟行情'
        latest = self.db_interface.get_latest_timestamp(table_name, dt.datetime.today() - dt.timedelta(days=365 * 3))
        date_range = self.calendar.select_dates(latest.date(), dt.date.today(), inclusive=(False, True))

        with tqdm(date_range) as pbar:
            for date in date_range:
                pbar.set_description(f'更新{date}的{table_name}')
                tickers = self.stock_list.ticker(date)
                self.get_minute_data_base(table_name=table_name, date=date, tickers=tickers)
                pbar.update()

    def update_stock_adj_factor(self):
        """更新股票复权因子"""

        def data_func(ticker: str, date: date_utils.DateType) -> pd.Series:
            data = self.w.wsd(ticker, 'adjfactor', date, date)
            data.name = '复权因子'
            return data

        self.sparse_data_template('复权因子', data_func)

    def update_stock_units(self):
        """更新股本信息"""

        # 流通股本
        def float_share_func(ticker: str, date: date_utils.DateType) -> pd.Series:
            data = self.w.wsd(ticker, 'float_a_shares', date, date, 'unit=1')
            data.name = 'A股流通股本'
            return data

        self.sparse_data_template('A股流通股本', float_share_func)

        # 自由流通股本
        def free_float_share_func(ticker: str, date: date_utils.DateType) -> pd.Series:
            data = self.w.wsd(ticker, 'free_float_shares', date, date, 'unit=1')
            data.name = '自由流通股本'
            return data

        self.sparse_data_template('自由流通股本', free_float_share_func)

        # 总股本
        def total_share_func(ticker: str, date: date_utils.DateType) -> pd.Series:
            data = self.w.wsd(ticker, 'total_shares', date, date, 'unit=1')
            data.name = '总股本'
            return data

        self.sparse_data_template('总股本', total_share_func)

        # A股总股本
        def total_a_share_func(ticker: str, date: date_utils.DateType) -> pd.Series:
            data = self.w.wsd(ticker, 'share_totala', date, date, 'unit=1')
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
            wind_data = wind_data.str.replace('III|Ⅲ|IV|Ⅳ$', '', regex=True)
            return wind_data

        table_name = f'{provider}行业'
        query_date = self.calendar.yesterday()
        latest = self.db_interface.get_latest_timestamp(table_name)
        if latest is None:
            latest = date_utils.date_type2datetime(constants.INDUSTRY_START_DATE[provider])
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
        """更新行业信息"""
        for provider in constants.INDUSTRY_DATA_PROVIDER:
            self._update_industry(provider)

    def update_pause_stock_info(self):
        """更新股票停牌信息"""
        table_name = '股票停牌'
        start_date = self.db_interface.get_latest_timestamp(table_name, dt.date(1990, 12, 10)) + dt.timedelta(days=1)
        end_date = self.calendar.yesterday()
        chunks = self.calendar.split_to_chunks(start_date, end_date, 20)

        renaming_dict = self._factor_param[table_name]
        with tqdm(chunks) as pbar:
            pbar.set_description('下载股票停牌数据')
            for range_start, range_end in chunks:
                start_date_str = range_start.strftime('%Y%m%d')
                end_date_str = range_end.strftime('%Y%m%d')
                pbar.set_postfix_str(f'{start_date_str} - {end_date_str}')
                data = self.w.wset('tradesuspend',
                                   f'startdate={start_date_str};enddate={end_date_str};field=date,wind_code,suspend_type,suspend_reason')
                data.rename(renaming_dict, axis=1, inplace=True)
                ind1 = (data['停牌类型'] == '盘中停牌') & (data['停牌原因'].str.startswith('股票价格'))
                ind2 = (data['停牌原因'].str.startswith('盘中'))
                ind3 = (data['停牌类型'] == '上午停牌') & (data['停牌类型'] == '下午停牌')
                data = data.loc[(~ind1) & (~ind2) & (~ind3), :]
                self.db_interface.insert_df(data, table_name)
                pbar.update()

    #######################################
    # convertible bond funcs
    #######################################
    def update_convertible_bond_daily_data(self):
        """更新可转债日行情"""
        table_name = '可转债日行情'
        renaming_dict = self._factor_param['可转债日行情']
        start_date = self.db_interface.get_latest_timestamp(table_name, dt.datetime(1993, 2, 9))
        dates = self.calendar.select_dates(start_date, dt.date.today(), inclusive=(False, True))

        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的{table_name}')
                tickers = self.convertible_bond_list.ticker(date)
                if tickers:
                    data = self.w.wss(tickers, list(renaming_dict.keys()), date=date, options='priceAdj=DP;cycle=D')
                    data.rename(renaming_dict, axis=1, inplace=True)
                    self.db_interface.insert_df(data, table_name)
                pbar.update()

    def update_cb_convertible_price(self):
        table_name = '可转债转股价'
        cb_tickers = ConvertibleBondTickers(self.db_interface)
        ticker = cb_tickers.all_ticker()
        start_date = cb_tickers.list_date()

        def convert_price_func(ticker: str, date: date_utils.DateType) -> pd.Series:
            data = self.w.wsd(ticker, 'clause_conversion2_swapshareprice', date, date, '')
            data.name = table_name
            return data

        self.sparse_data_template(table_name, convert_price_func, ticker=ticker, default_start_date=start_date)

    def update_cb_minutes_data(self) -> None:
        """更新可转债分钟行情"""
        table_name = '可转债分钟行情'
        latest = self.db_interface.get_latest_timestamp(table_name, dt.datetime.today() - dt.timedelta(days=365 * 3))
        date_range = self.calendar.select_dates(latest.date(), dt.date.today(), inclusive=(False, True))

        with tqdm(date_range) as pbar:
            for date in date_range[:10]:
                pbar.set_description(f'更新{date}的{table_name}')
                tickers = self.stock_list.ticker(date)
                self.get_minute_data_base(table_name=table_name, date=date, tickers=tickers)
                pbar.update()

    #######################################
    # future funcs
    #######################################
    def update_future_daily_data(self):
        """更新期货日行情"""
        contract_daily_table_name = '期货日行情'

        start_date = self.db_interface.get_latest_timestamp(contract_daily_table_name)
        dates = self.calendar.select_dates(start_date, dt.date.today(), inclusive=(False, True))

        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的{contract_daily_table_name}')
                data = self.w.wss(self.future_list.ticker(date), 'open, high, low, close, settle, volume, amt, oi',
                                  date=date, options='priceAdj=U;cycle=D')
                data.rename(self._factor_param[contract_daily_table_name], axis=1, inplace=True)
                self.db_interface.insert_df(data, contract_daily_table_name)
                pbar.update()

    #######################################
    # fund funcs
    #######################################
    def init_fund_info(self):
        self.get_fund_base_info()
        self.get_fund_time_info()

    def update_fund_info(self):
        date_str = self.calendar.today().strftime('%Y-%m-%d')

        # otc funds
        new_funds = self.w.wset('sectorconstituent', f'date={date_str};sectorid=1000007789000000;field=wind_code')
        new_funds = new_funds.index.tolist()
        self.get_fund_base_info(new_funds)
        self.get_fund_time_info(new_funds)

        old_funds = self.w.wset('sectorconstituent', f'date={date_str};sectorid=1000007791000000;field=wind_code')
        old_funds = old_funds.index.tolist()
        self.get_fund_time_info(old_funds)

    def get_fund_base_info(self, tickers: Sequence[str] = None):
        table_name = '基金列表'
        if tickers is None:
            date_str = dt.date.today().strftime('%Y-%m-%d')
            otc = self.w.wset('sectorconstituent', f'date={date_str};sectorid=1000008492000000;field=wind_code')
            tickers = otc.index.tolist()

            listed = self.w.wset('sectorconstituent', f'date={date_str};sectorid=1000027452000000;field=wind_code')
            tickers.extend(listed.index.tolist())

            tickers = [it for it in tickers if ((not it.startswith('F')) & (not it.startswith('P')))]
        else:
            db_tickers = self.db_interface.get_all_id(table_name)
            tickers = sorted(set(tickers) - set(db_tickers))
            if not tickers:
                return

        fields = {'证券名称': 'sec_name', '全名': 'fund_fullname', '管理人': 'fund_corp_fundmanagementcompany',
                  '封闭式': 'fund_type', '投资类型': 'fund_investtype', '初始基金': 'fund_initial',
                  '分级基金': 'fund_structuredfundornot', '定开': 'fund_regulopenfundornot',
                  '定开时长(月)': 'fund_operateperiod_cls', '管理费率': 'fund_managementfeeratio',
                  '浮动管理费': 'fund_floatingmgntfeeornot', '浮动管理费说明': 'fund_floatingmgntfeedescrip',
                  '托管费率': 'fund_custodianfeeratio', '销售服务费率': 'fund_salefeeratio',
                  }
        pre_fee_fields = {'前端申购费': 'fund_purchasefee', '最高申购费': 'fund_purchasefeeratio'}
        after_fee_fields = {'后端申购费': 'fund_purchasefee', '赎回费': 'fund_redemptionfee'}

        if len(tickers) > 3000:
            storage = []
            pre_fee_storage = []
            after_fee_storage = []
            for it in algo.chunk_list(tickers, 3000):
                storage.append(self.w.wss(it, ','.join(fields.values())))
                pre_fee_storage.append(self.w.wss(it, ','.join(pre_fee_fields.values()), 'chargesType=0'))
                after_fee_storage.append(self.w.wss(it, ','.join(after_fee_fields.values()), 'chargesType=1'))
            data = pd.concat(storage)
            pre_fee_data = pd.concat(pre_fee_storage)
            after_fee_data = pd.concat(after_fee_storage)
        else:
            data = self.w.wss(tickers, ','.join(fields.values()))
            pre_fee_data = self.w.wss(tickers, ','.join(pre_fee_fields.values()), 'chargesType=0')
            after_fee_data = self.w.wss(tickers, ','.join(after_fee_fields.values()), 'chargesType=1')
        data.columns = list(fields.keys())
        data.index.name = 'ID'
        data['封闭式'] = data['封闭式'].str.contains('封闭式')
        data['初始基金'] = (data['初始基金'] == '是')
        data['分级基金'] = (data['分级基金'] == '是')
        data['浮动管理费'] = (data['浮动管理费'] == '是')
        data['定开'] = (data['定开'] == '是')
        data['债券型'] = data['投资类型'].str.contains('债')
        data['ETF'] = (data['全名'].str.contains('交易型开放式')) & (~data.index.str.endswith('OF')) & (
            ~data['全名'].str.contains('联接基金').fillna(False))
        data['封闭运作转LOF时长(月)'] = data['全名'].apply(algo.extract_close_operate_period)
        pre_fee_data.columns = list(pre_fee_fields.keys())
        pre_fee_data.index.name = 'ID'
        after_fee_data.columns = list(after_fee_fields.keys())
        after_fee_data.index.name = 'ID'
        data = pd.concat([data, pre_fee_data, after_fee_data], axis=1)
        data['免赎回费持有期(日)'] = data['赎回费'].apply(self.extract_zero_redemption_fee_holding_period)
        self.db_interface.insert_df(data, table_name)

    def get_fund_time_info(self, tickers: Sequence[str] = None):
        init = True if tickers is None else False
        if tickers is None:
            tickers = self.db_interface.get_all_id('基金列表')

        exchange_ticker = [it for it in tickers if not it.endswith('.OF')]
        otc_ticker = [it for it in tickers if it.endswith('.OF')]
        storage = []
        if exchange_ticker:
            data = self.w.wss(exchange_ticker, 'ipo_date,delist_date')
            data.columns = ['上市日期', '退市日期']
            storage.append(data)
        if otc_ticker:
            if len(otc_ticker) > 3000:
                storage1 = []
                for it in algo.chunk_list(otc_ticker, 3000):
                    storage1.append(self.w.wss(it, 'fund_setupdate,fund_maturitydate'))
                data = pd.concat(storage1)
            else:
                data = self.w.wss(otc_ticker, 'fund_setupdate,fund_maturitydate')
            data.columns = ['上市日期', '退市日期']
            storage.append(data)

        data = pd.concat(storage)
        data.index.name = 'ID'
        data.loc[data['上市日期'] == dt.datetime(1899, 12, 30), '上市日期'] = pd.NaT
        data.loc[data['退市日期'] == dt.datetime(1899, 12, 30), '退市日期'] = pd.NaT
        if not init:
            self.db_interface.delete_id_records('基金时间表', data.index.tolist())
        self.db_interface.insert_df(data, '基金时间表')

        list_time_info = data.stack().reset_index()
        list_time_info.columns = ['ID', '上市状态', 'DateTime']
        list_time_info['上市状态'] = list_time_info['上市状态'].map({'上市日期': True, '退市日期': False})
        list_time_info['证券类型'] = ['场外基金' if it.endswith('.OF') else '场内基金' for it in list_time_info.ID.tolist()]
        list_time_info.set_index(['DateTime', 'ID'], inplace=True)
        self.db_interface.update_df(list_time_info, '证券代码')

    #######################################
    # option funcs
    #######################################
    def get_stock_option_daily_data(self, date: dt.datetime) -> None:
        """获取 ``date`` 的股票ETF期权和股指期权日行情"""
        contract_daily_table_name = '期权日行情'

        tickers = self.etf_option_list.ticker(date) + self.option_list.ticker(date)
        data = self.w.wss(tickers,
                          'high,open,low,close,volume,amt,oi,delta,gamma,vega,theta,rho',
                          date=date, priceAdj='U', cycle='D')
        data.rename(self._factor_param[contract_daily_table_name], axis=1, inplace=True)
        self.db_interface.insert_df(data, contract_daily_table_name)

    def update_stock_option_daily_data(self) -> None:
        """更新股票ETF期权和股指期权日行情"""
        contract_daily_table_name = '期权日行情'

        start_date = self.db_interface.get_latest_timestamp(contract_daily_table_name, dt.datetime(2015, 2, 8))
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
        """更新主要股指日行情"""
        table_name = '指数日行情'

        start_date = self.db_interface.get_latest_timestamp(table_name)
        dates = self.calendar.select_dates(start_date, dt.date.today(), inclusive=(False, True))

        indexes = list(constants.STOCK_INDEXES.values())
        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的{table_name}')
                indicators = 'open,low,high,close,volume,amt,mkt_cap_ard,total_shares,float_a_shares,free_float_shares,pe_ttm'
                data = self.w.wss(indexes, indicators, date=date, priceAdj='U', cycle='D')
                data = data.rename(self._factor_param[table_name], axis=1)
                self.db_interface.insert_df(data, table_name)
                pbar.update()

    #######################################
    # helper funcs
    #######################################
    def sparse_data_queryer(self, data_func, start_series: pd.Series = None, end_series: pd.Series = None,
                            desc: str = '', default_start_date: Union[Dict, date_utils.DateType] = None):
        start_ticker = [] if start_series.empty else start_series.index.get_level_values('ID')
        all_ticker = sorted(list(set(start_ticker) | set(end_series.index.get_level_values('ID'))))
        start_ts = None if start_series.empty else start_series.index.get_level_values('DateTime').max()

        tmp = start_series.reset_index().set_index('ID').reindex(all_ticker)
        start_series = tmp.reset_index().set_index(['DateTime', 'ID']).iloc[:, 0]

        end_index = pd.MultiIndex.from_product([[end_series.index.get_level_values('DateTime')[0]], all_ticker],
                                               names=['DateTime', 'ID'])
        end_series = end_series.reindex(end_index)

        ind = np.logical_not(start_series.isnull().values & end_series.isnull().values)
        start_series = start_series.loc[ind, :]
        end_series = end_series.loc[ind, :]

        if start_series.dtype == 'float64' or start_series.dtype == 'int64':
            ind = np.abs(start_series.values - end_series.values) > 0.001
            ind = ind | start_series.isnull().values | end_series.isnull().values
            ind = ind & (start_series.values != 0)
        else:
            ind = (start_series.values != end_series.values)
        start_series = start_series.loc[ind]
        end_series = end_series.loc[ind, :]

        if end_series.empty:
            return

        if start_ts and self.calendar.days_count(start_ts, end_series.index.get_level_values('DateTime')[0]) == 1:
            self.db_interface.insert_df(end_series, end_series.name)
            return

        with tqdm(start_series) as pbar:
            for i in range(start_series.shape[0]):
                new_val = end_series.iloc[i:i + 1]
                old_val = start_series.iloc[i:i + 1]
                if np.isnan(old_val.index.get_level_values('DateTime').values[0]):
                    ticker = old_val.index.get_level_values('ID').values[0]
                    if isinstance(default_start_date, dict):
                        index_date = default_start_date[ticker]
                    else:
                        index_date = date_utils.date_type2datetime(default_start_date)
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
                             default_start_date: Union[Dict, date_utils.DateType] = None):
        if default_start_date is None:
            default_start_date = self.stock_list.list_date()
        if ticker is None:
            ticker = self.stock_list.all_ticker()
        start_series = self.db_interface.read_table(table_name).groupby('ID').tail(1)
        start_series = start_series.loc[start_series.index.get_level_values('ID').isin(ticker), :]
        end_date = self.calendar.today()
        end_series = data_func(ticker=ticker, date=end_date)
        end_series.name = table_name

        self.sparse_data_queryer(data_func, start_series, end_series, f'更新{table_name}',
                                 default_start_date=default_start_date)

    def get_minute_data_base(self, table_name: str, date: dt.datetime, tickers: Sequence[str], options: str = ''):
        replace_dict = self._factor_param[table_name]

        start_time = dt.datetime.combine(date.date(), dt.time(hour=8))
        end_time = dt.datetime.combine(date.date(), dt.time(hour=16))
        storage = []
        for section in algo.chunk_list(tickers, 100):
            partial_data = self.w.wsi(section, 'open,high,low,close,volume,amt', start_time, end_time, options)
            storage.append(partial_data.dropna())
        data = pd.concat(storage)
        data.set_index('windcode', append=True, inplace=True)
        data.index.names = ['DateTime', 'ID']
        data.rename(replace_dict, axis=1, inplace=True)
        self.db_interface.insert_df(data, table_name)

    @staticmethod
    def extract_zero_redemption_fee_holding_period(entry: str) -> int:
        if not entry:
            return 0

        def period2days(period: str) -> int:
            num = int(period[:-1])
            if period.endswith('年'):
                num *= 365
            elif period.endswith('月'):
                num *= 30
            elif not period.endswith('日'):
                raise ValueError(f'{period} is not a valid period')
            return num

        line = entry.split('\r\n')
        days_storage = []
        days_reg = re.compile('\d+[日月年]')
        for it in line:
            t = days_reg.search(it)
            if t is None:
                continue
            days_storage.append(period2days(t.group()))
        if days_storage:
            return max(days_storage)
        else:
            return 0

    @classmethod
    def from_config(cls, config_loc: str):
        """根据 ``config_loc`` 的适配信息生成 ``WindData`` 实例"""
        db_interface = config.generate_db_interface_from_config(config_loc)
        return cls(db_interface)
