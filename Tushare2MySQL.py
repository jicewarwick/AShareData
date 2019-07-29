import datetime as dt
import json
from typing import Tuple, Any, Union, Sequence, Dict, List, Optional

import numpy as np
import pandas as pd
import pymysql
import sqlalchemy as sa
import tushare as ts
from sqlalchemy import Column, String, Float, DateTime, Text
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from tqdm import tqdm

DateType = Union[str, dt.datetime, dt.date]

Base = declarative_base()


class DataFrameMySQLWriter(object):
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

    def update_df(self, df: pd.DataFrame, table_name: str, index: bool = True) -> None:
        """ Write DataFrame to database

        :param df: DataFrame
        :param table_name: table name
        :param index: if True, set df.index to be the database table index
        :return:
        """
        # if table exists
        inspector = sa.inspect(self.engine)
        existing_tables = inspector.get_table_names()
        if table_name not in existing_tables:
            attr_dict = {'__tablename__': table_name}
            if index:
                for i, it in enumerate(df.index.names):
                    if isinstance(df.index.get_level_values(i)[0], pd.Timestamp):
                        attr_dict[it] = Column(DateTime)
                    else:
                        attr_dict[it] = Column(String(20))
                attr_dict['__table_args__'] = (sa.PrimaryKeyConstraint(*df.index.names), sa.Index(*df.index.names))
            for it in df.columns:
                attr_dict[it] = Column(guess_type(df[it])[0])

            input_table_class = type('InputTableClass', (Base,), attr_dict)
            input_table_class().__table__.create(bind=self.engine)

        # if all columns exists
        else:
            metadata = sa.MetaData(self.engine, reflect=True)
            table = metadata.tables[table_name]
            missing_columns = list(set(df.columns.tolist()) - set(table.columns.keys()))
            if missing_columns:
                sql_str = 'ALTER TABLE ' + table_name + ' ADD COLUMN ('
                storage = []
                for it in missing_columns:
                    storage.append(' '.join([it, guess_type(df[it])[1]]))
                query = sql_str + ', '.join(storage) + ')'
                self.engine.execute(query)

        # upsert data
        metadata = sa.MetaData(self.engine, reflect=True)
        table = metadata.tables[table_name]
        flat_df = df.reset_index()

        date_cols = flat_df.select_dtypes(np.datetime64).columns.values.tolist()
        for col in date_cols:
            flat_df[col] = flat_df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

        # replace nan to None so that insert will not error out
        # it seems that this operation changes dtypes. so do it last
        for col in flat_df.columns:
            flat_df[col] = np.where(flat_df[col].isnull(), None, flat_df[col])
        for _, row in flat_df.iterrows():
            insert_statement = insert(table).values(**row.to_dict())
            statement = insert_statement.on_duplicate_key_update(**row.to_dict())
            self.engine.execute(statement)


def guess_type(input_series: pd.Series) -> Tuple[Any, str]:
    if input_series.dtype == np.float64:
        return Float, 'Float'
    if input_series.dtype == type(object):
        return Text, 'Text'
    if input_series.dtype in [dt.datetime, dt.date, '<M8[ns]']:
        return DateTime, 'DATETIME'


class Tushare2MySQL(object):
    STOCK_EXCHANGES = ['SSE', 'SZSE']
    FUTURE_EXCHANGES = ['CFFEX', 'DCE', 'CZCE', 'SHFE', 'INE']
    ALL_EXCHANGES = STOCK_EXCHANGES + FUTURE_EXCHANGES

    def __init__(self, tushare_token: str, param_json: str = None) -> None:
        """
        Tushare to MySQL. 将tushare下载的数据写入MySQL中

        :param tushare_token: tushare token
        :param param_json: tushare 返回df的列名信息
        """
        self._tushare_token = tushare_token
        self._pro = ts.pro_api(self._tushare_token)

        with open(param_json, 'r', encoding='utf-8') as f:
            self._parameters = json.load(f)

        self._all_stocks = None
        self._calendar = None

        self.mysql_writer = None

    def add_mysql_db(self, ip: str, port: int, username: str, password: str, db_name='tushare_db') -> None:
        """添加MySQLWriter"""
        self.mysql_writer = DataFrameMySQLWriter(ip, port, username, password, db_name)

    # get data
    # ------------------------------------------
    def update_routine(self) -> None:
        # date = dt.date.today()
        # self.get_daily_hq(date)
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
        column_desc = self._parameters[data_category][table_name]
        fields = ','.join(column_desc['名称'].values.tolist())

        storage = []
        for exchange in self.STOCK_EXCHANGES:
            storage.append(self._pro.stock_company(exchange=exchange, fields=fields))
        df = pd.concat(storage)
        df = self._standardize_df(df, column_desc)
        self.mysql_writer.update_df(df, data_category)
        return df

    def get_daily_hq(self, trade_date: DateType = None,
                     start_date: DateType = None, end_date: DateType = None) -> None:
        """获取每日行情

        行情信息包括: 开高低收, 量额, 复权因子, 换手率, 市盈, 市净, 市销, 股本, 市值
        :param trade_date: 交易日期
        :param start_date: 开始日期
        :param end_date: 结束日期
        交易日期查询一天, 开始结束日期查询区间. 二选一
        :return: None
        """
        dates = [trade_date] if trade_date else self.select_dates(start_date, end_date)

        price_category = '日线行情'
        adj_factor_category = '复权因子'
        indicator_category = '每日指标'
        table_name = '输出参数'

        price_desc = self._parameters[price_category][table_name]
        adj_factor_desc = self._parameters[adj_factor_category][table_name]
        indicator_desc = self._parameters[indicator_category][table_name]

        with tqdm(dates) as pbar:
            for date in dates:
                current_date_str = self._datetime2str(date) if not isinstance(date, str) else date
                pbar.set_description('下载股票日行情: ' + current_date_str)

                # price data
                df = self._pro.daily(trade_date=current_date_str)
                # 涨跌幅未复权, 没有意义
                df.drop('pct_chg', axis=1, inplace=True)
                df = self._standardize_df(df, price_desc)
                self.mysql_writer.update_df(df, '股票日行情')

                # adj_factor data
                df = self._pro.query('adj_factor', trade_date=date)
                df = self._standardize_df(df, adj_factor_desc)
                self.mysql_writer.update_df(df, '股票日行情')

                # indicator data
                df = self._pro.query('daily_basic', trade_date=date)
                df = self._standardize_df(df, indicator_desc)
                self.mysql_writer.update_df(df, '股票日行情')

                pbar.update(1)

    def get_past_names(self, ticker: str = None, start_date: DateType = None) -> pd.DataFrame:
        """获取曾用名
        ref: https://tushare.pro/document/2?doc_id=100
        """
        data_category = '股票曾用名'
        table_name = '输出参数'
        column_desc = self._parameters[data_category][table_name]
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
                pbar.set_description('下载股票名称: ' + stock)
                pbar.update(1)
                df = self.get_past_names(stock)
                # df = df[['证券名称']]
                self.mysql_writer.update_df(df, '股票名称')

    def get_ipo_info(self) -> pd.DataFrame:
        """
        IPO新股列表

        ref: https://tushare.pro/document/2?doc_id=123
        :return: IPO新股列表df
        """
        data_category = 'IPO新股列表'
        table_name = '输出参数'
        column_desc = self._parameters[data_category][table_name]

        df = self._pro.new_share()
        df = self._standardize_df(df, column_desc, index=['ts_code'])
        self.mysql_writer.update_df(df, data_category)
        return df

    # todo:
    def get_balance_sheet(self, ticker: str, period: str):
        df = self._pro.balancesheet(ts_code=ticker, period=period)
        return df

    # todo:
    def batch_download_balance_sheet(self, ticker: list, report_type: str, period: str):
        pass

    # utilities
    # ------------------------------------------
    @staticmethod
    def _str2datetime(date: str) -> Optional[dt.datetime]:
        if date:
            return dt.datetime.strptime(date, '%Y%m%d')

    @staticmethod
    def _datetime2str(date: dt.datetime) -> str:
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

        return [it for it in self._calendar if start <= it <= end]
