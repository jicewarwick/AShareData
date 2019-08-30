import datetime as dt
import json
import logging
from time import sleep
from typing import Sequence, List, Callable, Mapping

import pandas as pd
import sqlalchemy as sa
import tushare as ts
from tqdm import tqdm

from DataFrameMySQLWriter import DataFrameMySQLWriter
from utils import date_type2str, date_type2datetime, DateType, select_dates, STOCK_EXCHANGES, STOCK_INDEXES


class TushareData(object):

    def __init__(self, tushare_token: str, param_json: str, db_schema: str, engine: sa.engine.Engine) -> None:
        """
        Tushare to MySQL. 将tushare下载的数据写入MySQL中

        :param tushare_token: tushare token
        :param param_json: tushare 返回df的列名信息
        """
        self._tushare_token = tushare_token
        self._pro = ts.pro_api(self._tushare_token)

        with open(param_json, 'r', encoding='utf-8') as f:
            self._factor_param = json.load(f)

        with open(db_schema, 'r', encoding='utf-8') as f:
            self._db_parameters = json.load(f)

        self._listed_stocks = None
        self._all_stocks = None
        self._calendar = None

        self.mysql_writer = DataFrameMySQLWriter(engine)
        self._initialize_db_table()

    @property
    def all_stocks(self) -> List[str]:
        """ 获取所有股票列表"""
        if self._all_stocks is None:
            self._get_all_stocks()
        return self._all_stocks

    @property
    def listed_stocks(self) -> List[str]:
        """获取现上市的股票列表"""
        if self._listed_stocks is None:
            self._get_all_stocks()
        return self._listed_stocks

    @property
    def calendar(self) -> List[dt.datetime]:
        """获取交易日历"""
        if self._calendar is None:
            self._get_calendar()
        return self._calendar

    def _initialize_db_table(self):
        logging.debug('检查数据库完整性.')
        for table_name, type_info in self._db_parameters.items():
            self.mysql_writer.create_table(table_name, type_info)

    # get data
    # ------------------------------------------
    def update_routine(self) -> None:
        self.get_company_info()
        self.get_ipo_info()

        latest_time, _ = self.mysql_writer.get_progress('指数日行情')
        if latest_time is None:
            latest_time = dt.date(2008, 1, 1)
        self.get_index_daily(latest_time)

        date = dt.date.today()
        latest_time, _ = self.mysql_writer.get_progress('股票日行情')
        if latest_time is None:
            latest_time = dt.date(2008, 1, 1)
        self.get_daily_hq(start_date=latest_time, end_date=date)

        stocks = self.mysql_writer.get_all_id('合并资产负债表')
        stocks = list(set(self._all_stocks) - set(stocks)) if stocks else self.all_stocks
        self.get_financial(stocks)

        latest_time, _ = self.mysql_writer.get_progress('沪深港股通持股明细')
        if latest_time is None:
            latest_time = dt.date(2016, 6, 29)
        self.get_hs_holding(start_date=latest_time)

        self.get_shibor()

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
        for exchange in STOCK_EXCHANGES:
            storage.append(self._pro.stock_company(exchange=exchange, fields=fields))
        df = pd.concat(storage)
        df = self._standardize_df(df, column_desc)
        self.mysql_writer.update_df(df, data_category)
        logging.info('上市公司基本信息下载完成.')
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
        dates = [trade_date] if trade_date else select_dates(self.calendar, start_date, end_date)

        price_category = '日线行情'
        adj_factor_category = '复权因子'
        indicator_category = '每日指标'
        table_name = '输出参数'

        price_desc = self._factor_param[price_category][table_name]
        price_fields = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
        adj_factor_desc = self._factor_param[adj_factor_category][table_name]
        indicator_desc = self._factor_param[indicator_category][table_name]
        indicator_fields = ['ts_code', 'trade_date', 'total_share', 'float_share', 'free_share']

        with tqdm(dates, ascii=True) as pbar:
            for date in dates:
                current_date_str = date_type2str(date)
                pbar.set_description(f'下载{current_date_str}的日行情: ')

                # price data
                df = self._pro.daily(trade_date=current_date_str, fields=','.join(price_fields))
                df['vol'] = df['vol'] * 100
                df['amount'] = df['amount'] * 1000
                price_df = self._standardize_df(df, price_desc)

                # adj_factor data
                df = self._pro.adj_factor(trade_date=current_date_str)
                adj_df = self._standardize_df(df, adj_factor_desc)

                # indicator data
                df = self._pro.query('daily_basic', trade_date=current_date_str, fields=indicator_fields)
                df[['total_share', 'float_share', 'free_share']] = df[['total_share', 'float_share',
                                                                       'free_share']] * 10000
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
        interval = 60.0 / 100.0
        with tqdm(self.all_stocks, ascii=True) as pbar:
            for stock in self.all_stocks:
                pbar.set_description(f'下载{stock}的股票名称: ')
                df = self.get_past_names(stock)
                self.mysql_writer.update_df(df, '股票曾用名')
                sleep(interval)
                pbar.update(1)

    def get_dividend(self) -> None:
        """
        获取上市公司分红送股信息

        ref: https://tushare.pro/document/2?doc_id=103
        :return: 上市公司分红送股信息df
        """
        interval = 60.0 / 100.0
        data_category = '分红送股'
        table_name = '输出参数'
        column_desc = self._factor_param[data_category][table_name]
        fields = ['ts_code', 'end_date', 'ann_date', 'div_proc', 'stk_div', 'cash_div_tax', 'record_date', 'ex_date',
                  'pay_date', 'imp_ann_date']

        with tqdm(self.all_stocks) as pbar:
            for stock in self.all_stocks:
                pbar.set_description(f'下载{stock}的分红送股数据: ')
                df = self._pro.dividend(ts_code=stock, fields=fields)
                df = df.loc[df['div_proc'] == '实施', :]
                # 无公布时间的权宜之计
                df['ann_date'].where(df['ann_date'].notnull(), df['imp_ann_date'], inplace=True)
                df.drop(['div_proc', 'imp_ann_date'], axis=1, inplace=True)
                df = self._standardize_df(df, column_desc, index=['ann_date', 'ts_code'])
                self.mysql_writer.update_df(df, data_category)
                sleep(interval)
                pbar.update(1)

        logging.info('市公司分红送股信息下载完成.')

    def get_ipo_info(self, end_date: DateType = None) -> pd.DataFrame:
        """
        IPO新股列表

        ref: https://tushare.pro/document/2?doc_id=123
        :return: IPO新股列表df
        """
        data_category = 'IPO新股列表'
        table_name = '输出参数'
        column_desc = self._factor_param[data_category][table_name]

        if end_date:
            end_date = date_type2str(end_date)
        df = self._pro.new_share(end_date=end_date)
        df[['amount', 'market_amount', 'limit_amount']] = df[['amount', 'market_amount', 'limit_amount']] * 10000
        df['funds'] = df['funds'] * 100000000
        df = self._standardize_df(df, column_desc, index=['ts_code'])
        self.mysql_writer.update_df(df, data_category)
        logging.info('IPO新股列表下载完成.')
        return df

    def get_hs_const(self) -> None:
        """
        返回沪深股通成分股进出记录. 月末更新.
        """
        table_name = '沪深股通成份股'
        storage = []
        for hs_type in ['SH', 'SZ']:
            for is_new in ['0', '1']:
                storage.append(self._pro.hs_const(hs_type=hs_type, is_new=is_new))
        df = pd.concat(storage)
        in_part = df.loc[:, ['in_date', 'ts_code']]
        in_part[table_name] = True
        out_part = df.loc[:, ['out_date', 'ts_code']].dropna()
        out_part[table_name] = False
        out_part.rename({'out_date': 'in_date'}, axis=1, inplace=True)
        stacked_df = pd.concat([in_part, out_part])
        stacked_df = self._standardize_df(stacked_df, {}, index=['in_date', 'ts_code'])
        self.mysql_writer.update_df(stacked_df, table_name)
        logging.info(f'{table_name}数据下载完成')

    def get_hs_holding(self, trade_date: DateType = None,
                       start_date: DateType = None, end_date: DateType = None) -> None:
        interval = 60
        if start_date:
            start_date = date_type2str(start_date)
        if end_date:
            end_date = date_type2str(end_date)
        dates = [trade_date] if trade_date else select_dates(self.calendar, start_date, end_date)

        category = '沪深港股通持股明细'
        table_name = '输出参数'
        desc = self._factor_param[category][table_name]
        fields = ['trade_date', 'ts_code', 'vol']

        logging.debug('开始下载沪深港股通持股明细.')

        with tqdm(dates, ascii=True) as pbar:
            for date in dates:
                current_date_str = date_type2str(date)
                pbar.set_description(f'下载{current_date_str}的沪深港股通持股明细: ')
                df = self._pro.hk_hold(trade_date=current_date_str, fields=fields)
                df = self._standardize_df(df, desc)
                self.mysql_writer.update_df(df, category)
                pbar.update(1)
                sleep(interval)

    def get_index_daily(self, start_date: DateType = None, end_date: DateType = None) -> None:
        """
        更新指数行情信息. 包括开高低收, 量额, 市盈, 市净, 市值
        默认指数为沪指, 深指, 中小盘, 创业板, 50, 300, 500
        注:300不包含市盈等指标

        :param start_date: 开始时间
        :param end_date: 结束时间
        :return: 指数行情信息
        """
        if start_date:
            start_date = date_type2str(start_date)
        if end_date:
            end_date = date_type2str(end_date)

        category = '指数日线行情'
        db_table_name = '指数日行情'
        basic_category = '大盘指数每日指标'
        table_name = '输出参数'
        desc = self._factor_param[category][table_name]
        price_fields = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
        basic_desc = self._factor_param[basic_category][table_name]
        basic_fields = ['ts_code', 'trade_date', 'total_mv', 'float_mv',
                        'total_share', 'float_share', 'free_share',
                        'pe', 'pe_ttm', 'pb']

        logging.debug('开始下载指数日数据.')
        storage = []
        indexes = list(STOCK_INDEXES.values())
        for index in indexes:
            storage.append(self._pro.index_daily(ts_code=index, start_date=start_date, end_date=end_date,
                                                 fields=','.join(price_fields)))
        df = pd.concat(storage)
        df['vol'] = df['vol'] * 100
        df['amount'] = df['amount'] * 1000
        df = self._standardize_df(df, desc)
        self.mysql_writer.update_df(df, db_table_name)

        storage.clear()
        for index in indexes:
            storage.append(self._pro.index_dailybasic(ts_code=index, start_date=start_date, end_date=end_date,
                                                      fields=','.join(basic_fields)))
        df = pd.concat(storage)
        df = self._standardize_df(df, basic_desc)
        self.mysql_writer.update_df(df, db_table_name)
        logging.info('指数日数据下载完成')

    def get_financial(self, stock_list: Sequence[str] = None) -> None:
        """
        获取公司的 资产负债表, 现金流量表 和 利润表, 并写入数据库

        注:
        - 由于接口限流严重, 这个函数通过循环股票完成, 需要很长很长时间才能完成(1天半?)
        """
        request_interval = 60.0 / 80.0 / 2.5
        balance_sheet = '资产负债表'
        income = '利润表'
        cash_flow = '现金流量表'
        output_desc = '输出参数'
        report_type_info = '主要报表类型说明'
        company_type_info = '公司类型'

        balance_sheet_desc = self._factor_param[balance_sheet][output_desc]
        report_type_desc = self._factor_param[balance_sheet][report_type_info]
        company_type_desc = self._factor_param[balance_sheet][company_type_info]
        income_desc = self._factor_param[income][output_desc]
        cash_flow_desc = self._factor_param[cash_flow][output_desc]

        def download_data(api_func: Callable, report_type_list: Sequence[str],
                          column_name_dict: Mapping[str, str], table_name: str) -> None:
            storage = []
            for i in report_type_list:
                storage.append(api_func(ts_code=ticker, start_date='19900101', report_type=i))
                sleep(request_interval)
            df = pd.concat(storage)
            df = self._standardize_df(df, column_name_dict, index=['f_ann_date', 'ts_code'])

            df = df.reset_index()
            df.replace({'报表类型': report_type_desc, '公司类型': company_type_desc}, inplace=True)
            df.set_index(['DateTime', 'ID'], drop=True, inplace=True)

            self.mysql_writer.update_df(df, table_name)

        # 分 合并/母公司, 单季/年
        if stock_list is None:
            stock_list = self.all_stocks
        with tqdm(stock_list, ascii=True) as pbar:
            loop_vars = [(self._pro.income, income_desc, income), (self._pro.cashflow, cash_flow_desc, cash_flow)]
            for ticker in stock_list:
                pbar.set_description(f'下载{ticker}的财报: ')
                download_data(self._pro.balancesheet, ['1', '4', '5', '11'], balance_sheet_desc, '合并' + balance_sheet)
                download_data(self._pro.balancesheet, ['6', '9', '10', '12'], balance_sheet_desc, '母公司' + balance_sheet)
                for f, desc, table in loop_vars:
                    download_data(f, ['7', '8'], desc, '母公司单季度' + table)
                    download_data(f, ['6', '9', '10'], desc, '母公司' + table)
                    download_data(f, ['2', '3'], desc, '合并单季度' + table)
                    download_data(f, ['1', '4', '5'], desc, '合并' + table)
                pbar.update(1)

    def get_shibor(self, start_date: DateType = None, end_date: DateType = None) -> None:
        if start_date:
            start_date = date_type2str(start_date)
        if end_date:
            end_date = date_type2str(end_date)

        data_category = 'Shibor利率数据'
        table_name = '输出参数'
        desc = self._factor_param[data_category][table_name]
        df = self._pro.shibor(start_date=start_date, end_date=end_date)
        df = self._standardize_df(df, desc, index=['date'])
        self.mysql_writer.update_df(df, data_category)

    # utilities
    # ------------------------------------------
    @staticmethod
    def _standardize_df(df: pd.DataFrame, parameter_info: Mapping[str, str],
                        index: Sequence[str] = None) -> pd.DataFrame:
        dates_columns = [it for it in df.columns if 'date' in it]
        for it in dates_columns:
            df[it] = df[it].apply(date_type2datetime)
        if index is None:
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

    def _get_all_stocks(self) -> List[str]:
        """ 获取所有股票列表, 包括上市, 退市和暂停上市的股票

        ref: https://tushare.pro/document/2?doc_id=25

        """
        storage = []
        lookup_dict = {'listed': 'L', 'de-listed': 'D', 'paused': 'P'}
        storage_dict = {}
        for list_status, symbol in lookup_dict.items():
            storage_dict[list_status] = self._pro.stock_basic(exchange='', list_status=symbol, fields='ts_code')
            storage.append(storage_dict[list_status])
        output = pd.concat(storage)
        output = output.ts_code
        output.name = '证券代码'
        output.to_sql('股票列表', self.mysql_writer.engine, if_exists='replace')
        self._listed_stocks = storage_dict['listed'].ts_code.values.tolist()
        self._all_stocks = output.values.tolist()
        return self._all_stocks

    def _get_calendar(self) -> List[dt.datetime]:
        """返回上交所交易日历"""
        df = self._pro.query('trade_cal', is_open=1)
        cal_date = df.cal_date
        cal_date.name = '交易日期'
        cal_date = cal_date.map(date_type2datetime)
        self._calendar = cal_date.dt.to_pydatetime().tolist()
        cal_date.to_sql('交易日历', self.mysql_writer.engine, if_exists='replace')
        return self._calendar
