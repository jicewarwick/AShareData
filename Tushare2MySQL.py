import datetime as dt
import json
from time import sleep
from typing import Union, Sequence, Dict, List, Optional, Callable
import logging

import numpy as np
import pandas as pd
import sqlalchemy as sa
import tushare as ts
from sqlalchemy import Column, Float, DateTime, Text, Table, Integer, VARCHAR
from sqlalchemy.dialects.mysql import insert, DOUBLE
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from tqdm import tqdm

DateType = Union[str, dt.datetime, dt.date]

Base = declarative_base()


class DataFrameMySQLWriter(object):
    type_mapper = {
        'datetime': DateTime,
        'float': Float,
        'double': DOUBLE,
        'str': Text,
        'int': Integer,
        'varchar': VARCHAR(20)
    }

    def __init__(self, ip: str, port: int, username: str, password: str, db_name: str,
                 driver: str = 'mysql+pymysql') -> None:
        """ DataFrame to MySQL Database Writer

        Write pd.DataFrame to MySQL server. Feature:
        - Auto Create Table Set 'DateTime' and 'ID' as primary key and index if they are available as DataFrame index.
        - Add new columns when necessary
        - Handle datetime insertion
        - Handle nan insertion
        - Insert new or update old records using on_duplicate_key_update()

        :param ip: server ip
        :param port: server port
        :param username: username
        :param password: password
        :param db_name: database name
        :param driver: MySQL driver
        """
        assert 'mysql' in driver, 'This class is MySQL database ONLY!!!'
        url = URL(drivername=driver, username=username, password=password, host=ip, port=port, database=db_name)
        self.engine = sa.create_engine(url)

    def create_table(self, table_name, table_info: Dict[str, str]) -> None:
        meta = sa.MetaData(bind=self.engine, reflect=True)
        col_names = list(table_info.keys())
        col_types = [self.type_mapper[it] for it in table_info.values()]
        primary_keys = list({'DateTime', 'ID'} & set(col_names))
        existing_tables = [it.lower() for it in meta.tables]
        if table_name.lower() in existing_tables:
            logging.debug(table_name + ' already exists!!!')
            return

        new_table = Table(table_name, meta,
                          *(Column(col_name, col_type) for col_name, col_type in zip(col_names, col_types)),
                          sa.PrimaryKeyConstraint(*primary_keys))
        new_table.create()
        logging.info('Table ' + table_name + ' created')

    def update_df(self, df: pd.DataFrame, table_name: str) -> None:
        """ Write DataFrame to database

        :param df: DataFrame
        :param table_name: table name
        :return:
        """
        metadata = sa.MetaData(self.engine, reflect=True)
        table = metadata.tables[table_name]
        flat_df = df.reset_index()

        date_cols = flat_df.select_dtypes(np.datetime64).columns.values.tolist()
        for col in date_cols:
            flat_df[col] = flat_df[col].apply(self.date2str)

        # replace nan to None so that insert will not error out
        # it seems that this operation changes dtypes. so do it last
        for col in flat_df.columns:
            flat_df[col] = np.where(flat_df[col].isnull(), None, flat_df[col])
        for _, row in flat_df.iterrows():
            insert_statement = insert(table).values(**row.to_dict())
            statement = insert_statement.on_duplicate_key_update(**row.to_dict())
            self.engine.execute(statement)

    def get_latest_update_time(self, table_name: str) -> dt.datetime:
        metadata = sa.MetaData(self.engine, reflect=True)
        assert table_name in metadata.tables.keys(), f'数据库中无名为 {table_name} 的表'
        table = metadata.tables[table_name]
        assert 'DateTime' in table.columns.keys(), f'{table_name} 表中无时间列'
        session_maker = sessionmaker(bind=self.engine)
        session = session_maker()
        return session.query(func.max(table.c.DateTime)).one()[0]

    @staticmethod
    def date2str(date) -> Optional[str]:
        if isinstance(date, pd.Timestamp):
            return date.strftime('%Y-%m-%d %H:%M:%S')


class Tushare2MySQL(object):
    STOCK_EXCHANGES = ['SSE', 'SZSE']
    FUTURE_EXCHANGES = ['CFFEX', 'DCE', 'CZCE', 'SHFE', 'INE']
    ALL_EXCHANGES = STOCK_EXCHANGES + FUTURE_EXCHANGES
    STOCK_INDEXES = {'上证指数': '000001.SH', '深证成指': '399001.SZ', '中小板指': '399005.SZ', '创业板指': '399006.SZ',
                     '上证50': '000016.SH', '沪深300': '000300.SH', '中证500': '000905.SH'}

    def __init__(self, tushare_token: str, param_json: str = None) -> None:
        """
        Tushare to MySQL. 将tushare下载的数据写入MySQL中

        :param tushare_token: tushare token
        :param param_json: tushare 返回df的列名信息
        """
        self._tushare_token = tushare_token
        self._pro = ts.pro_api(self._tushare_token)

        with open(param_json, 'r', encoding='utf-8') as f:
            parameters = json.load(f)

        self._factor_param = parameters['参数描述']
        self._db_parameters = parameters['数据库参数']

        self._all_stocks = None
        self._calendar = None

        self.mysql_writer = None

    def initialize_db_table(self):
        for table_name, type_info in self._db_parameters.items():
            self.mysql_writer.create_table(table_name, type_info)

    def add_mysql_db(self, ip: str, port: int, username: str, password: str, db_name='tushare_db') -> None:
        """添加MySQLWriter"""
        self.mysql_writer = DataFrameMySQLWriter(ip, port, username, password, db_name)

    # get data
    # ------------------------------------------
    def update_routine(self) -> None:
        # get latest update date
        latest = self.mysql_writer.get_latest_update_time('股票日行情')
        date = dt.date.today()
        self.get_daily_hq(latest, date)
        self.update_index_daily()
        self.get_company_info()
        self.get_ipo_info()

    def get_company_info(self) -> pd.DataFrame:
        """
        获取上市公司基本信息

        ref: https://tushare.pro/document/2?doc_id=112
        :return: 上市公司基础信息df
        """
        data_category = '上市公司基本信息'
        table_name = '输出参数'
        column_desc = self._factor_param[data_category][table_name]
        fields = ','.join(list(column_desc.keys()))

        storage = []
        for exchange in self.STOCK_EXCHANGES:
            storage.append(self._pro.stock_company(exchange=exchange, fields=fields))
        df = pd.concat(storage)
        df = self._standardize_df(df, column_desc)
        self.mysql_writer.update_df(df, data_category)
        return df

    def get_daily_hq(self, trade_date: DateType = None,
                     start_date: DateType = None, end_date: DateType = dt.datetime.now()) -> None:
        """获取每日行情

        行情信息包括: 开高低收, 量额, 复权因子, 换手率, 市盈, 市净, 市销, 股本, 市值
        :param trade_date: 交易日期
        :param start_date: 开始日期
        :param end_date: 结束日期
        交易日期查询一天, 开始结束日期查询区间. 二选一
        :return: None
        """
        if (not trade_date) & (not start_date):
            raise ValueError('trade_date 和 start_date 必填一个!')
        dates = [trade_date] if trade_date else self.select_dates(start_date, end_date)

        price_category = '日线行情'
        adj_factor_category = '复权因子'
        indicator_category = '每日指标'
        table_name = '输出参数'

        price_desc = self._factor_param[price_category][table_name]
        price_fields = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
        adj_factor_desc = self._factor_param[adj_factor_category][table_name]
        indicator_desc = self._factor_param[indicator_category][table_name]
        indicator_fields = ['ts_code', 'trade_date', 'total_share', 'float_share', 'free_share']

        with tqdm(dates) as pbar:
            for date in dates:
                current_date_str = self._datetime2str(date) if not isinstance(date, str) else date
                pbar.set_description(f'下载股票日行情: {current_date_str}')

                # price data
                df = self._pro.daily(trade_date=current_date_str, fields=','.join(price_fields))
                df['amount'] = df['amount'] * 1000
                price_df = self._standardize_df(df, price_desc)

                # adj_factor data
                df = self._pro.adj_factor(trade_date=current_date_str)
                adj_df = self._standardize_df(df, adj_factor_desc)

                # indicator data
                df = self._pro.query('daily_basic', trade_date=current_date_str, fields=indicator_fields)
                df[['total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv']] = \
                    df[['total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv']] * 10000
                indicator_df = self._standardize_df(df, indicator_desc)

                # combine all dfs
                full_df = pd.concat([price_df, adj_df, indicator_df], axis=1)
                full_df.dropna(subset=['开盘价'], inplace=True)
                self.mysql_writer.update_df(full_df, '股票日行情')

                pbar.update(1)

    def get_past_names(self, ticker: str = None, start_date: DateType = None) -> pd.DataFrame:
        """获取曾用名
        ref: https://tushare.pro/document/2?doc_id=100
        """
        data_category = '股票曾用名'
        table_name = '输出参数'
        column_desc = self._factor_param[data_category][table_name]
        fields = 'ts_code, name, start_date'

        df = self._pro.namechange(ts_code=ticker, start_date=start_date, fields=fields)
        df = self._standardize_df(df, column_desc, index=['start_date', 'ts_code'])
        return df

    def get_all_past_names(self):
        """获取所有股票的曾用名"""
        if not self._all_stocks:
            self.get_all_stocks()
        with tqdm(self._all_stocks) as pbar:
            for stock in self._all_stocks:
                pbar.set_description(f'下载股票名称: {stock}')
                df = self.get_past_names(stock)
                self.mysql_writer.update_df(df, '股票曾用名')
                pbar.update(1)

    def get_ipo_info(self, end_date: str = None) -> pd.DataFrame:
        """
        IPO新股列表

        ref: https://tushare.pro/document/2?doc_id=123
        :return: IPO新股列表df
        """
        data_category = 'IPO新股列表'
        table_name = '输出参数'
        column_desc = self._factor_param[data_category][table_name]

        df = self._pro.new_share(end_date=end_date)
        df[['amount', 'market_amount', 'limit_amount']] = df[['amount', 'market_amount', 'limit_amount']] * 10000
        df['funds'] = df['funds'] * 100000000
        df = self._standardize_df(df, column_desc, index=['ts_code'])
        self.mysql_writer.update_df(df, data_category)
        return df

    def update_index_daily(self, start_date: DateType = None) -> None:
        """
        更新指数行情信息. 包括开高低收, 量额, 换手率, 市盈, 市净, 市销, 股本, 市值
        默认指数为沪指, 深指, 中小盘, 创业板, 50, 300, 500
        注:300不包含市盈等指标

        :param start_date: 开始时间
        :return: 指数行情信息
        """
        category = '指数日线行情'
        db_table_name = '指数日行情'
        basic_category = '大盘指数每日指标'
        table_name = '输出参数'
        desc = self._factor_param[category][table_name]
        basic_desc = self._factor_param[basic_category][table_name]

        indexes = list(self.STOCK_INDEXES.values())
        if not start_date:
            try:
                latest = self.mysql_writer.get_latest_update_time(category)
            except AssertionError:
                latest = dt.date(2008, 1, 1)
            start_date = self._datetime2str(latest)

        storage = []
        for index in indexes:
            storage.append(self._pro.index_daily(ts_code=index, start_date=start_date))
        df = pd.concat(storage)
        df = self._standardize_df(df, desc)
        self.mysql_writer.update_df(df, db_table_name)

        storage = []
        for index in indexes:
            storage.append(self._pro.index_dailybasic(ts_code=index, start_date=start_date))
        df = pd.concat(storage)
        df = self._standardize_df(df, basic_desc)
        self.mysql_writer.update_df(df, db_table_name)

    def get_financial(self, stock_list: Sequence[str] = None) -> None:
        """
        获取所有公司公布的 资产负债表, 现金流量表 和 利润表, 并写入数据库

        注:
        - 由于接口限流严重, 这个函数通过循环股票完成, 需要很长很长时间才能完成(1天半?)
        """
        request_interval = 60.0 / 80.0 / 2.5
        balance_sheet = '资产负债表'
        income = '利润表'
        cashflow = '现金流量表'
        output_desc = '输出参数'
        report_type_info = '主要报表类型说明'
        company_type_info = '公司类型'

        balance_sheet_desc = self._factor_param[balance_sheet][output_desc]
        report_type_desc = self._factor_param[balance_sheet][report_type_info]
        company_type_desc = self._factor_param[balance_sheet][company_type_info]
        income_desc = self._factor_param[income][output_desc]
        cashflow_desc = self._factor_param[cashflow][output_desc]

        def download_data(api_func: Callable, report_type_list: Sequence[str],
                          column_name_dict: Dict[str, str], table_name: str) -> None:
            storage = []
            for i in report_type_list:
                storage.append(api_func(ts_code=ticker, start_date='19900101', report_type=i))
                sleep(request_interval)
            df = pd.concat(storage)
            df = self._standardize_df(df, column_name_dict, index=['f_ann_date', 'ts_code'])
            df = df.reset_index()
            df['DateTime'] = df['DateTime'] - df['报表类型'].map(lambda x: pd.DateOffset(seconds=int(x)))
            df.replace({'报表类型': report_type_desc, '公司类型': company_type_desc}, inplace=True)
            df.set_index(['DateTime', 'ID'], drop=True, inplace=True)
            self.mysql_writer.update_df(df, table_name)

        # 分 合并/母公司, 单季/年
        if not stock_list:
            self.get_all_stocks()
            stock_list = self._all_stocks
        with tqdm(stock_list) as pbar:
            loop_vars = [(self._pro.income, income_desc, income), (self._pro.cashflow, cashflow_desc, cashflow)]
            for ticker in stock_list:
                pbar.set_description(f'下载财报: {ticker}')
                download_data(self._pro.balancesheet, ['1', '4', '5', '11'], balance_sheet_desc, '合并' + balance_sheet)
                download_data(self._pro.balancesheet, ['6', '9', '10', '12'], balance_sheet_desc, '母公司' + balance_sheet)
                for f, desc, table in loop_vars:
                    download_data(f, ['7', '8'], desc, '母公司单季度' + table)
                    download_data(f, ['6', '9', '10'], desc, '母公司' + table)
                    download_data(f, ['2', '3'], desc, '合并单季度' + table)
                    download_data(f, ['1', '4', '5'], desc, '合并' + table)
                pbar.update(1)

    # utilities
    # ------------------------------------------
    @staticmethod
    def _str2datetime(date: str) -> Optional[dt.datetime]:
        if date not in ['', 'nan']:
            return dt.datetime.strptime(date, '%Y%m%d')

    @staticmethod
    def _datetime2str(date: Union[dt.datetime, dt.date]) -> str:
        return date.strftime('%Y%m%d')

    def _standardize_df(self, df: pd.DataFrame, parameter_info: Dict,
                        index: Sequence[str] = None) -> pd.DataFrame:
        dates_columns = [it for it in df.columns if 'date' in it]
        for it in dates_columns:
            df[it] = df[it].apply(self._str2datetime)
        if not index:
            index = sorted(list({'trade_date', 'ts_code'} & set(df.columns)))
        if len(index) > 2:
            raise RuntimeError('index 的个数不能超过 2 !')
        if len(index) == 2:
            index_names = ['DateTime', 'ID']
        else:
            if 'date' in index[0]:
                index_names = ['DateTime']
            else:
                index_names = ['ID']

        df = df.set_index(index, drop=True)
        df.rename(parameter_info, axis=1, inplace=True)
        df.index.names = index_names
        return df

    def get_all_stocks(self) -> List[str]:
        """ 获取所有股票列表, 包括上市, 退市和暂停上市的股票

        ref: https://tushare.pro/document/2?doc_id=25

        """
        storage = []
        for list_status in ['L', 'D', 'P']:
            storage.append(self._pro.stock_basic(exchange='', list_status=list_status, fields='ts_code'))
        output = pd.concat(storage)
        if self.mysql_writer:
            output.to_sql('股票列表', self.mysql_writer.engine, if_exists='replace')
        self._all_stocks = output.ts_code.values.tolist()
        return self._all_stocks

    def get_calendar(self) -> List[dt.datetime]:
        """返回上交所交易日历"""
        df = self._pro.query('trade_cal', is_open=1)
        calendar = df.cal_date.values.tolist()
        if self.mysql_writer:
            df.to_sql('交易日历', self.mysql_writer.engine, if_exists='replace')
        self._calendar = [self._str2datetime(it) for it in calendar]
        return self._calendar

    def select_dates(self, start: DateType, end: DateType):
        """返回区间内的所有交易日(包含start和end)"""
        if not self._calendar:
            self.get_calendar()

        if isinstance(start, str):
            start = self._str2datetime(start)

        if isinstance(end, str):
            end = self._str2datetime(end)

        return [it for it in self._calendar if (start <= it <= end)]
