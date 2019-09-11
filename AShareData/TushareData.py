import datetime as dt
import json
import logging
from importlib.resources import open_text
from time import sleep
from typing import Sequence, List, Callable, Mapping

import pandas as pd
import sqlalchemy as sa
import tushare as ts
from cached_property import cached_property
from tqdm import tqdm

from AShareData.DataFrameMySQLWriter import DataFrameMySQLWriter
from AShareData.TradingCalendar import TradingCalendar
from AShareData.constants import STOCK_EXCHANGES, STOCK_INDEXES, BOARD_INDEXES
from AShareData.utils import date_type2str, date_type2datetime, DateType


class TushareData(object):
    def __init__(self, tushare_token: str, engine: sa.engine.Engine,
                 param_json_loc: str = None, db_schema_loc: str = None, init: bool = False) -> None:
        """
        Tushare to MySQL. 将tushare下载的数据写入MySQL中

        :param tushare_token: tushare token
        :param engine: sqlalchemy engine
        :param param_json_loc: tushare 返回df的列名信息
        :param db_schema_loc: MySQL database quasi-schema
        """
        self._pro = ts.pro_api(tushare_token)

        if param_json_loc is None:
            from AShareData import data
            self._factor_param = json.load(open_text(data, 'param.json'))
        else:
            with open(param_json_loc, 'r', encoding='utf-8') as f:
                self._factor_param = json.load(f)

        if db_schema_loc is None:
            from AShareData import data
            self._db_parameters = json.load(open_text(data, 'db_schema.json'))
        else:
            with open(db_schema_loc, 'r', encoding='utf-8') as f:
                self._db_parameters = json.load(f)

        self.engine = engine
        self.mysql_writer = DataFrameMySQLWriter(engine)

        self._all_stocks = None
        self._get_all_stocks()

        if init:
            self.update_calendar()
            logging.debug('检查数据库完整性.')
            for table_name, type_info in self._db_parameters.items():
                self.mysql_writer.create_table(table_name, type_info)

    @property
    def all_stocks(self) -> List[str]:
        """ 获取所有股票列表"""
        return self._all_stocks

    @cached_property
    def calendar(self) -> TradingCalendar:
        """获取交易日历"""
        return TradingCalendar(self.engine)

    # get data
    # ------------------------------------------
    def update_routine(self) -> None:
        # self.get_company_info()
        self.get_shibor(start_date=self._check_db_timestamp('Shibor利率数据', dt.date(2006, 10, 8)))
        self.get_ipo_info(start_date=self._check_db_timestamp('IPO新股列表', dt.datetime(1990, 1, 1)))

        self.get_daily_hq(start_date=self._check_db_timestamp('股票日行情', dt.date(2008, 1, 1)), end_date=dt.date.today())
        self.get_past_names(start_date=self._check_db_timestamp('股票曾用名', dt.datetime(1990, 1, 1)))

        self.get_index_daily(self._check_db_timestamp('指数日行情', dt.date(2008, 1, 1)))
        latest = self._check_db_timestamp('指数成分股权重', '20050101')
        if latest < dt.datetime.now() - dt.timedelta(days=20):
            self.get_index_weight(start_date=latest)

        self.get_hs_const()
        self.get_hs_holding(start_date=self._check_db_timestamp('沪深港股通持股明细', dt.date(2016, 6, 29)))

        stocks = self.mysql_writer.get_all_id('合并资产负债表')
        stocks = list(set(self.all_stocks) - set(stocks)) if stocks else self.all_stocks
        if stocks:
            self.get_financial(stocks)

    def get_company_info(self) -> pd.DataFrame:
        """
        获取上市公司基本信息

        ref: https://tushare.pro/document/2?doc_id=112
        :return: 上市公司基础信息df
        """
        data_category = '上市公司基本信息'
        column_desc = self._factor_param[data_category]['输出参数']
        fields = list(column_desc.keys())

        logging.debug(f'开始下载{data_category}.')
        storage = []
        for exchange in STOCK_EXCHANGES:
            storage.append(self._pro.stock_company(exchange=exchange, fields=fields))
        df = pd.concat(storage)
        df = self._standardize_df(df, column_desc)
        self.mysql_writer.update_df(df, data_category)
        logging.info(f'{data_category}下载完成.')
        return df

    def get_daily_hq(self, trade_date: DateType = None,
                     start_date: DateType = None, end_date: DateType = dt.datetime.now()) -> None:
        """获取每日行情

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

        output_fields = '输出参数'

        price_desc = self._factor_param['日线行情'][output_fields]
        price_fields = list(price_desc.keys())
        adj_factor_desc = self._factor_param['复权因子'][output_fields]
        indicator_desc = self._factor_param['每日指标'][output_fields]
        indicator_fields = list(indicator_desc.keys())

        with tqdm(dates) as pbar:
            for date in dates:
                current_date_str = date_type2str(date)
                pbar.set_description(f'下载{current_date_str}的日行情')

                # price data
                df = self._pro.daily(trade_date=current_date_str, fields=price_fields)
                df['vol'] = df['vol'] * 100
                df['amount'] = df['amount'] * 1000
                price_df = self._standardize_df(df, price_desc)

                # adj_factor data
                df = self._pro.adj_factor(trade_date=current_date_str)
                adj_df = self._standardize_df(df, adj_factor_desc)

                # indicator data
                df = self._pro.daily_basic(trade_date=current_date_str, fields=indicator_fields)
                df[['total_share', 'float_share', 'free_share']] = \
                    df[['total_share', 'float_share', 'free_share']] * 10000
                indicator_df = self._standardize_df(df, indicator_desc)

                # combine all dfs
                full_df = pd.concat([price_df, adj_df, indicator_df], axis=1)
                full_df.dropna(subset=['开盘价'], inplace=True)
                self.mysql_writer.update_df(full_df, '股票日行情')

                pbar.update(1)

    # todo: select index
    def get_financial_index(self, ticker: str, period: str) -> pd.DataFrame:
        data_category = '财务指标数据'
        column_desc = self._factor_param[data_category]['输出参数']

        df = self._pro.fina_indicator(ts_code=ticker, period=period, fields=list(column_desc.keys()))
        df = self._standardize_df(df, column_desc)
        return df

    def get_past_names(self, ticker: str = None, start_date: DateType = None) -> pd.DataFrame:
        """获取曾用名
        ref: https://tushare.pro/document/2?doc_id=100
        """
        data_category = '股票曾用名'
        column_desc = self._factor_param[data_category]['输出参数']
        fields = list(column_desc.keys())
        start_date = date_type2str(start_date)

        logging.debug(f'开始下载{ticker if ticker else ""}{data_category}.')
        df = self._pro.namechange(ts_code=ticker, start_date=start_date, fields=fields)
        df = self._standardize_df(df, column_desc)
        self.mysql_writer.update_df(df, data_category)
        logging.debug(f'{ticker if ticker else ""}{data_category}下载完成.')
        return df

    def get_all_past_names(self):
        """获取所有股票的曾用名"""
        interval = 60.0 / 100.0
        with tqdm(self.all_stocks) as pbar:
            for stock in self.all_stocks:
                pbar.set_description(f'下载{stock}的股票名称')
                self.get_past_names(stock)
                pbar.update(1)
                sleep(interval)

    def get_dividend(self) -> None:
        """
        获取上市公司分红送股信息

        ref: https://tushare.pro/document/2?doc_id=103
        :return: 上市公司分红送股信息df
        """
        interval = 60.0 / 100.0
        data_category = '分红送股'
        column_desc = self._factor_param[data_category]['输出参数']
        fields = list(column_desc.keys())

        logging.debug(f'开始下载{data_category}.')
        with tqdm(self.all_stocks) as pbar:
            for stock in self.all_stocks:
                pbar.set_description(f'下载{stock}的分红送股数据')
                df = self._pro.dividend(ts_code=stock, fields=fields)
                df = df.loc[df['div_proc'] == '实施', :]
                # 无公布时间的权宜之计
                df['ann_date'].where(df['ann_date'].notnull(), df['imp_ann_date'], inplace=True)
                df.drop(['div_proc', 'imp_ann_date'], axis=1, inplace=True)
                df = self._standardize_df(df, column_desc)
                self.mysql_writer.update_df(df, data_category)
                sleep(interval)
                pbar.update(1)

        logging.info(f'{data_category}信息下载完成.')

    def get_ipo_info(self, start_date: DateType = None) -> pd.DataFrame:
        """
        IPO新股列表

        ref: https://tushare.pro/document/2?doc_id=123
        :return: IPO新股列表df
        """
        start_date = date_type2str(start_date)

        data_category = 'IPO新股列表'
        column_desc = self._factor_param[data_category]['输出参数']

        logging.debug(f'开始下载{data_category}.')
        df = self._pro.new_share(start_date=start_date)
        df[['amount', 'market_amount', 'limit_amount']] = df[['amount', 'market_amount', 'limit_amount']] * 10000
        df['funds'] = df['funds'] * 100000000
        df = self._standardize_df(df, column_desc)
        self.mysql_writer.update_df(df, data_category)
        logging.info(f'{data_category}下载完成.')
        return df

    def get_hs_const(self) -> None:
        """
        返回沪深股通成分股进出记录. 月末更新.
        """
        data_category = '沪深股通成份股'
        logging.debug(f'开始下载{data_category}.')
        storage = []
        for hs_type in ['SH', 'SZ']:
            for is_new in ['0', '1']:
                storage.append(self._pro.hs_const(hs_type=hs_type, is_new=is_new))
        df = pd.concat(storage)
        in_part = df.loc[:, ['in_date', 'ts_code']]
        in_part[data_category] = True
        out_part = df.loc[:, ['out_date', 'ts_code']].dropna()
        out_part[data_category] = False
        out_part.rename({'out_date': 'in_date'}, axis=1, inplace=True)
        stacked_df = pd.concat([in_part, out_part])
        stacked_df = self._standardize_df(stacked_df, {'in_date': 'DateTime', 'ts_code': 'ID'})
        self.mysql_writer.update_df(stacked_df, data_category)
        logging.info(f'{data_category}数据下载完成')

    def get_hs_holding(self, trade_date: DateType = None,
                       start_date: DateType = None, end_date: DateType = None) -> None:
        interval = 60
        dates = [trade_date] if trade_date else self.calendar.select_dates(start_date, end_date)

        data_category = '沪深港股通持股明细'
        desc = self._factor_param[data_category]['输出参数']
        fields = list(desc.keys())

        logging.debug(f'开始下载{data_category}.')
        with tqdm(dates) as pbar:
            for date in dates:
                current_date_str = date_type2str(date)
                pbar.set_description(f'下载{current_date_str}的沪深港股通持股明细')
                df = self._pro.hk_hold(trade_date=current_date_str, fields=fields)
                df = self._standardize_df(df, desc)
                self.mysql_writer.update_df(df, data_category)
                pbar.update(1)
                sleep(interval)
        logging.info(f'{data_category}下载完成.')

    def get_index_daily(self, start_date: DateType = None, end_date: DateType = None) -> None:
        """
        更新指数行情信息. 包括开高低收, 量额, 市盈, 市净, 市值
        默认指数为沪指, 深指, 中小盘, 创业板, 50, 300, 500
        注:300不包含市盈等指标

        :param start_date: 开始时间
        :param end_date: 结束时间
        :return: 指数行情信息
        """
        db_table_name = '指数日行情'
        table_name = '输出参数'
        desc = self._factor_param['指数日线行情'][table_name]
        price_fields = list(desc.keys())
        basic_desc = self._factor_param['大盘指数每日指标'][table_name]
        basic_fields = list(basic_desc.keys())

        logging.debug(f'开始下载{db_table_name}.')
        storage = []
        indexes = list(STOCK_INDEXES.values())
        start_date, end_date = date_type2datetime(start_date), date_type2datetime(end_date)
        for index in indexes:
            storage.append(self._pro.index_daily(ts_code=index, start_date=start_date, end_date=end_date,
                                                 fields=price_fields))
        df = pd.concat(storage)
        df['vol'] = df['vol'] * 100
        df['amount'] = df['amount'] * 1000
        df = self._standardize_df(df, desc)
        self.mysql_writer.update_df(df, db_table_name)

        storage.clear()
        for index in indexes:
            storage.append(self._pro.index_dailybasic(ts_code=index, start_date=start_date, end_date=end_date,
                                                      fields=basic_fields))
        df = pd.concat(storage)
        df = self._standardize_df(df, basic_desc)
        self.mysql_writer.update_df(df, db_table_name)
        logging.info(f'{db_table_name}下载完成')

    def get_financial(self, stock_list: Sequence[str] = None, start_date: DateType = '19900101') -> None:
        """
        获取公司的 资产负债表, 现金流量表 和 利润表, 并写入数据库

        注:
        - 由于接口限流严重, 这个函数通过循环股票完成, 需要很长很长时间才能完成(1天半?)
        """
        request_interval = 60.0 / 80.0 / 2.5
        balance_sheet = '资产负债表'
        income = '利润表'
        cash_flow = '现金流量表'
        start_date = date_type2str(start_date)

        balance_sheet_desc = self._factor_param[balance_sheet]['输出参数']
        income_desc = self._factor_param[income]['输出参数']
        cash_flow_desc = self._factor_param[cash_flow]['输出参数']
        report_type_desc = self._factor_param[balance_sheet]['主要报表类型说明']
        company_type_desc = self._factor_param[balance_sheet]['公司类型']

        def download_data(api_func: Callable, report_type_list: Sequence[str],
                          column_name_dict: Mapping[str, str], table_name: str) -> None:
            storage = []
            for i in report_type_list:
                storage.append(api_func(ts_code=ticker, start_date=start_date, report_type=i))
                sleep(request_interval)
            df = pd.concat(storage)
            df.replace({'report_type': report_type_desc, 'comp_type': company_type_desc}, inplace=True)
            df = self._standardize_df(df, column_name_dict)
            self.mysql_writer.update_df(df, table_name)

        # 分 合并/母公司, 单季/年
        if stock_list is None:
            stock_list = self.all_stocks
        logging.debug(f'开始下载财报.')
        with tqdm(stock_list) as pbar:
            loop_vars = [(self._pro.income, income_desc, income), (self._pro.cashflow, cash_flow_desc, cash_flow)]
            for ticker in stock_list:
                pbar.set_description(f'下载{ticker}的财报')
                download_data(self._pro.balancesheet, ['1', '4', '5', '11'], balance_sheet_desc, f'合并{balance_sheet}')
                download_data(self._pro.balancesheet, ['6', '9', '10', '12'], balance_sheet_desc, f'母公司{balance_sheet}')
                for f, desc, table in loop_vars:
                    download_data(f, ['7', '8'], desc, f'母公司单季度{table}')
                    download_data(f, ['6', '9', '10'], desc, f'母公司{table}')
                    download_data(f, ['2', '3'], desc, f'合并单季度{table}')
                    download_data(f, ['1', '4', '5'], desc, f'合并{table}')
                pbar.update(1)
        logging.info(f'财报下载完成')

    def get_shibor(self, start_date: DateType = None, end_date: DateType = None) -> pd.DataFrame:
        data_category = 'Shibor利率数据'
        desc = self._factor_param[data_category]['输出参数']

        logging.debug(f'开始下载{data_category}.')
        start_date, end_date = date_type2datetime(start_date), date_type2datetime(end_date)
        df = self._pro.shibor(start_date=start_date, end_date=end_date)
        df = self._standardize_df(df, desc)
        self.mysql_writer.update_df(df, data_category)
        logging.info(f'{data_category}下载完成.')
        return df

    def get_index_weight(self, indexes: Sequence[str] = None,
                         start_date: DateType = None, end_date: DateType = dt.date.today()) -> None:
        interval = 60.0 / 70.0
        if indexes is None:
            indexes = BOARD_INDEXES
        start_date, end_date = date_type2datetime(start_date), date_type2datetime(end_date)
        dates = self.calendar.get_last_day_of_month(start_date, end_date)
        dates = sorted(list(set([start_date] + dates + [end_date])))

        data_category = '指数成分和权重'
        column_desc = self._factor_param[data_category]['输出参数']
        logging.debug(f'开始下载{data_category}.')
        with tqdm(dates) as pbar:
            for i in range(len(dates) - 1):
                storage = []
                for index in indexes:
                    pbar.set_description(f'下载{dates[i]} 到 {dates[i + 1]} 的 {index} 的 成分股权重')
                    storage.append(self._pro.index_weight(index_code=index,
                                                          start_date=date_type2str(dates[i]),
                                                          end_date=date_type2str(dates[i + 1])))
                    sleep(interval)
                df = self._standardize_df(pd.concat(storage), column_desc)
                self.mysql_writer.update_df(df, '指数成分股权重')
                pbar.update(1)
        logging.info(f'{data_category}下载完成.')

    # utilities
    # ------------------------------------------
    @staticmethod
    def _standardize_df(df: pd.DataFrame, parameter_info: Mapping[str, str]) -> pd.DataFrame:
        dates_columns = [it for it in df.columns if it.endswith('date')]
        for it in dates_columns:
            df[it] = df[it].apply(date_type2datetime)

        df.rename(parameter_info, axis=1, inplace=True)
        index = sorted(list({'DateTime', 'ID', '报告期', 'IndexCode'} & set(df.columns)))
        df = df.set_index(index, drop=True)
        return df

    def _get_all_stocks(self) -> List[str]:
        """ 获取所有股票列表, 包括上市, 退市和暂停上市的股票

        ref: https://tushare.pro/document/2?doc_id=25

        """
        data_category = '股票列表'
        desc = self._factor_param[data_category]['输出参数']

        logging.debug(f'开始下载{data_category}.')
        storage = []
        list_status = ['L', 'D', 'P']
        fields = ['ts_code', 'list_date', 'delist_date']
        for status in list_status:
            storage.append(self._pro.stock_basic(exchange='', list_status=status, fields=fields))
        output = pd.concat(storage)
        listed = output.loc[:, ['ts_code', 'list_date']]
        listed['list_status'] = True
        unlisted = output.loc[:, ['ts_code', 'delist_date']].dropna().rename({'delist_date': 'list_date'}, axis=1)
        unlisted['list_status'] = False
        output = pd.concat([listed, unlisted])
        self._all_stocks = sorted(list(set(output['ts_code'].values.tolist())))
        output = self._standardize_df(output, desc)
        self.mysql_writer.update_df(output, '股票上市退市')

        logging.info(f'{data_category}下载完成.')
        return self._all_stocks

    def update_calendar(self) -> None:
        """返回上交所交易日历"""
        df = self._pro.trade_cal(is_open=1)
        cal_date = df.cal_date
        cal_date.name = '交易日期'
        cal_date = cal_date.map(date_type2datetime)
        cal_date.to_sql('交易日历', self.mysql_writer.engine, if_exists='replace')

    def _check_db_timestamp(self, table_name: str, default_timestamp: DateType) -> dt.datetime:
        latest_time, _ = self.mysql_writer.get_progress(table_name)
        if latest_time is None:
            latest_time = date_type2datetime(default_timestamp)
        return latest_time
