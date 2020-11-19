import datetime as dt
import itertools
import logging
import re
from time import sleep
from typing import Callable, Mapping, Sequence, Union

import pandas as pd
import tushare as ts
from cached_property import cached_property
from tqdm import tqdm

from . import constants, DateUtils, utils
from .DataSource import DataSource
from .DBInterface import DBInterface
from .Tickers import StockTickers


class TushareData(DataSource):
    def __init__(self, tushare_token: str, db_interface: DBInterface, param_json_loc: str = None) -> None:
        """
        Tushare to Database. 将tushare下载的数据写入数据库中

        :param tushare_token: tushare token
        :param db_interface: DBInterface
        :param param_json_loc: tushare 返回df的列名信息
        """
        super().__init__(db_interface)
        self._pro = ts.pro_api(tushare_token)
        self._factor_param = utils.load_param('tushare_param.json', param_json_loc)

        if self._check_db_timestamp('证券代码', dt.datetime(1990, 1, 1)) < dt.datetime.today():
            self.update_base_info()

    def update_base_info(self):
        self.update_calendar()
        self.update_stock_list_date()
        self.update_convertible_bond_list_date()
        self.update_fund_list_date()
        self.update_option_list_date()

    @cached_property
    def stock_tickers(self) -> StockTickers:
        return StockTickers(self.db_interface)

    def update_routine(self) -> None:
        """自动更新函数"""
        self.get_company_info()
        self.get_shibor(start_date=self._check_db_timestamp('Shibor利率数据', dt.date(2006, 10, 8)))
        self.get_ipo_info(start_date=self._check_db_timestamp('IPO新股列表', dt.datetime(1990, 1, 1)))

        # self.get_daily_hq(start_date=self._check_db_timestamp('股票日行情', dt.date(2008, 1, 1)), end_date=dt.date.today())
        # self.get_daily_hq(start_date=self._check_db_timestamp('总股本', dt.date(2008, 1, 1)), end_date=dt.date.today())
        self.get_past_names(start_date=self._check_db_timestamp('证券名称', dt.datetime(1990, 1, 1)))

        # self.get_index_daily(self._check_db_timestamp('指数日行情', dt.date(2008, 1, 1)))
        # latest = self._check_db_timestamp('指数成分股权重', '20050101')
        # if latest < dt.datetime.now() - dt.timedelta(days=20):
        #     self.get_index_weight(start_date=latest)

        self.get_hs_const()
        self.get_hs_holding(start_date=self._check_db_timestamp('沪深港股通持股明细', dt.date(2016, 6, 29)))

        # stocks = self.db_interface.get_all_id('合并资产负债表')
        # stocks = list(set(self.all_stocks) - set(stocks)) if stocks else self.all_stocks
        # if stocks:
        #     self.get_financial(stocks)

    def get_company_info(self) -> pd.DataFrame:
        """
        获取上市公司基本信息

        :ref: https://tushare.pro/document/2?doc_id=112

        :return: 上市公司基础信息df
        """
        data_category = '上市公司基本信息'
        column_desc = self._factor_param[data_category]['输出参数']
        fields = list(column_desc.keys())

        logging.debug(f'开始下载{data_category}.')
        storage = []
        for exchange in constants.STOCK_EXCHANGES:
            storage.append(self._pro.stock_company(exchange=exchange, fields=fields))
        df = pd.concat(storage)
        df = self._standardize_df(df, column_desc)
        self.db_interface.update_df(df, data_category)
        logging.info(f'{data_category}下载完成.')
        return df

    def get_daily_hq(self, trade_date: DateUtils.DateType = None,
                     start_date: DateUtils.DateType = None, end_date: DateUtils.DateType = None) -> None:
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
        if not end_date:
            end_date = dt.datetime.today()
        dates = [trade_date] if trade_date else self.calendar.select_dates(start_date, end_date)
        pre_date = self.calendar.offset(dates[0], -1)

        output_fields = '输出参数'

        price_desc = self._factor_param['日线行情'][output_fields]
        price_fields = list(price_desc.keys())
        adj_factor_desc = self._factor_param['复权因子'][output_fields]
        indicator_desc = self._factor_param['每日指标'][output_fields]
        indicator_fields = list(indicator_desc.keys())

        # pre data:
        def get_pre_data(tn: str) -> pd.Series:
            return self.db_interface.read_table(tn, tn, end_date=pre_date).groupby('ID').tail(1)

        pre_adj_factor = get_pre_data('复权因子')
        pre_dict = {'total_share': get_pre_data('总股本'),
                    'float_share': get_pre_data('流通股本'),
                    'free_share': get_pre_data('自由流通股本')}

        with tqdm(dates) as pbar:
            for date in dates:
                current_date_str = DateUtils.date_type2str(date)
                pbar.set_description(f'下载{current_date_str}的日行情')

                # price data
                df = self._pro.daily(trade_date=current_date_str, fields=price_fields)
                df['vol'] = df['vol'] * 100
                df['amount'] = df['amount'] * 1000
                price_df = self._standardize_df(df, price_desc)
                self.db_interface.update_df(price_df, '股票日行情')

                # adj_factor data
                df = self._pro.adj_factor(trade_date=current_date_str)
                adj_df = self._standardize_df(df, adj_factor_desc)
                self.db_interface.update_compact_df(adj_df, '复权因子', pre_adj_factor)
                pre_adj_factor = adj_df

                # indicator data
                df = self._pro.daily_basic(trade_date=current_date_str, fields=indicator_fields)
                df = self._standardize_df(df, indicator_desc).multiply(10000)
                for key, value in pre_dict.items():
                    col_name = indicator_desc[key]
                    self.db_interface.update_compact_df(df[col_name], col_name, value)
                    pre_dict[key] = df[col_name]

                pbar.update()

    # todo: TBD
    def get_financial_index(self, ticker: str, period: str) -> pd.DataFrame:
        """ 获取财务指标, 返回pd.DataFrame, 未入库(WARNING: UNSTABLE API)

        :param ticker: 证券代码(000001.SZ)
        :param period: 报告期截止日(20001231)
        :return: 财务指标
        """
        data_category = '财务指标数据'
        column_desc = self._factor_param[data_category]['输出参数']

        df = self._pro.fina_indicator(ts_code=ticker, period=period, fields=list(column_desc.keys()))
        df = self._standardize_df(df, column_desc)
        return df

    @DateUtils.strlize_input_dates
    def get_past_names(self, ticker: str = None, start_date: DateUtils.DateType = None) -> pd.DataFrame:
        """获取曾用名

        ref: https://tushare.pro/document/2?doc_id=100

        :param ticker: 证券代码(000001.SZ)
        :param start_date: 开始日期
        """
        data_category = '证券名称'
        column_desc = self._factor_param[data_category]['输出参数']
        fields = list(column_desc.keys())

        logging.debug(f'开始下载{ticker if ticker else ""}{data_category}.')
        df = self._pro.namechange(ts_code=ticker, start_date=start_date, fields=fields)
        df = self._standardize_df(df, column_desc)
        self.db_interface.update_df(df, data_category)
        logging.debug(f'{ticker if ticker else ""}{data_category}下载完成.')
        return df

    def get_all_past_names(self):
        """获取所有股票的曾用名"""
        interval = 60.0 / 100.0
        raw_df = self.get_past_names()
        raw_df_start_dates = raw_df.index.get_level_values('DateTime').min()
        uncovered_stocks = self.stock_tickers.ticker(raw_df_start_dates)

        with tqdm(uncovered_stocks) as pbar:
            for stock in uncovered_stocks:
                pbar.set_description(f'下载{stock}的股票名称')
                self.get_past_names(stock)
                pbar.update()
                sleep(interval)
        logging.info('股票曾用名下载完成.')

    def get_dividend(self) -> None:
        """ 获取上市公司分红送股信息 """
        interval = 60.0 / 100.0
        data_category = '分红送股'
        column_desc = self._factor_param[data_category]['输出参数']
        fields = list(column_desc.keys())

        logging.debug(f'开始下载{data_category}.')
        tickers = self.stock_tickers.ticker()
        with tqdm(tickers) as pbar:
            for stock in tickers:
                pbar.set_description(f'下载{stock}的分红送股数据')
                df = self._pro.dividend(ts_code=stock, fields=fields)
                df = df.loc[df['div_proc'] == '实施', :]
                # 无公布时间的权宜之计
                df['ann_date'].where(df['ann_date'].notnull(), df['imp_ann_date'], inplace=True)
                df.drop(['div_proc', 'imp_ann_date'], axis=1, inplace=True)
                df = self._standardize_df(df, column_desc)
                self.db_interface.update_df(df, data_category)
                sleep(interval)
                pbar.update()

        logging.info(f'{data_category}信息下载完成.')

    @DateUtils.strlize_input_dates
    def get_ipo_info(self, start_date: DateUtils.DateType = None) -> pd.DataFrame:
        """ IPO新股列表 """
        data_category = 'IPO新股列表'
        column_desc = self._factor_param[data_category]['输出参数']

        logging.debug(f'开始下载{data_category}.')
        df = self._pro.new_share(start_date=start_date)
        df[['amount', 'market_amount', 'limit_amount']] = df[['amount', 'market_amount', 'limit_amount']] * 10000
        df['funds'] = df['funds'] * 100000000
        df = self._standardize_df(df, column_desc)
        self.db_interface.update_df(df, data_category)
        logging.info(f'{data_category}下载完成.')
        return df

    def get_hs_const(self) -> None:
        """ 沪深股通成分股进出记录. 月末更新. """
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
        self.db_interface.update_df(stacked_df, data_category)
        logging.info(f'{data_category}数据下载完成')

    def get_hs_holding(self, trade_date: DateUtils.DateType = None,
                       start_date: DateUtils.DateType = None, end_date: DateUtils.DateType = None) -> None:
        """ 沪深港股通持股明细 """
        interval = 60
        dates = [trade_date] if trade_date else self.calendar.select_dates(start_date, end_date)

        data_category = '沪深港股通持股明细'
        desc = self._factor_param[data_category]['输出参数']
        fields = list(desc.keys())

        logging.debug(f'开始下载{data_category}.')
        with tqdm(dates) as pbar:
            for date in dates:
                current_date_str = DateUtils.date_type2str(date)
                pbar.set_description(f'下载{current_date_str}的沪深港股通持股明细')
                df = self._pro.hk_hold(trade_date=current_date_str, fields=fields)
                df = self._standardize_df(df, desc)
                self.db_interface.update_df(df, data_category)
                pbar.update()
                sleep(interval)
        logging.info(f'{data_category}下载完成.')

    def get_index_daily(self, start_date: DateUtils.DateType = None, end_date: DateUtils.DateType = None) -> None:
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
        indexes = list(constants.STOCK_INDEXES.values())
        start_date, end_date = DateUtils.date_type2str(start_date), DateUtils.date_type2str(end_date)
        for index in indexes:
            storage.append(self._pro.index_daily(ts_code=index, start_date=start_date, end_date=end_date,
                                                 fields=price_fields))
        df = pd.concat(storage)
        df['vol'] = df['vol'] * 100
        df['amount'] = df['amount'] * 1000
        df = self._standardize_df(df, desc)
        self.db_interface.update_df(df, db_table_name)

        storage.clear()
        for index in indexes:
            storage.append(self._pro.index_dailybasic(ts_code=index, start_date=start_date, end_date=end_date,
                                                      fields=basic_fields))
        df = pd.concat(storage)
        df = self._standardize_df(df, basic_desc)
        self.db_interface.update_df(df, db_table_name)
        logging.info(f'{db_table_name}下载完成')

    @DateUtils.strlize_input_dates
    def get_financial(self, stock_list: Sequence[str] = None, start_date: DateUtils.DateType = '19900101') -> None:
        """
        获取公司的 资产负债表, 现金流量表 和 利润表, 并写入数据库

        注:
        - 由于接口限流严重, 这个函数通过循环股票完成, 需要很长很长时间才能完成(1天半?)
        """
        # db_end_date = self._check_db_timestamp('合并资产负债表', dt.datetime(1990, 1, 1))
        # db_end_date = DateUtils.date_type2str(db_end_date)
        # disclose_date = self._pro.disclosure_date(ts_code='600518.SH', end_date='20191231',
        #                                           fields=['ts_code', 'ann_date', 'end_date', 'pre_date', 'actual_date',
        #                                                   'modify_date'])
        # df = self._pro.balancesheet(ts_code='600518.SH')
        # income_df = self._pro.income(ts_code='600518.SH')
        # cashflow_df = self._pro.cashflow(ts_code='600518.SH', period='20191231', fields=list(cash_flow_desc.keys()))

        request_interval = 60.0 / 80.0 / 2.5
        balance_sheet = '资产负债表'
        income = '利润表'
        cash_flow = '现金流量表'

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
            self.db_interface.update_df(df, table_name)

        # 分 合并/母公司
        if stock_list is None:
            stock_list = self.stock_tickers.ticker()
        logging.debug(f'开始下载财报.')
        with tqdm(stock_list) as pbar:
            loop_vars = [(self._pro.income, income_desc, income), (self._pro.cashflow, cash_flow_desc, cash_flow)]
            for ticker in stock_list:
                pbar.set_description(f'下载{ticker}的财报')
                download_data(self._pro.balancesheet, ['1', '4', '5', '11'], balance_sheet_desc, f'合并{balance_sheet}')
                download_data(self._pro.balancesheet, ['6', '9', '10', '12'], balance_sheet_desc, f'母公司{balance_sheet}')
                for f, desc, table in loop_vars:
                    download_data(f, ['6', '9', '10'], desc, f'母公司{table}')
                    download_data(f, ['1', '4', '5'], desc, f'合并{table}')
                pbar.update()
        logging.info(f'财报下载完成')

    @DateUtils.strlize_input_dates
    def get_shibor(self, start_date: DateUtils.DateType = None, end_date: DateUtils.DateType = None) -> pd.DataFrame:
        """ Shibor利率数据 """
        data_category = 'Shibor利率数据'
        desc = self._factor_param[data_category]['输出参数']

        logging.debug(f'开始下载{data_category}.')
        df = self._pro.shibor(start_date=start_date, end_date=end_date)
        df = self._standardize_df(df, desc)
        self.db_interface.update_df(df, data_category)
        logging.info(f'{data_category}下载完成.')
        return df

    @DateUtils.dtlize_input_dates
    def get_index_weight(self, indexes: Sequence[str] = None,
                         start_date: DateUtils.DateType = None, end_date: DateUtils.DateType = None) -> None:
        """ 指数成分和权重

        默认指数为 ['000016.SH', '399300.SH', '000905.SH'], 即50, 300, 500

        :param indexes: 指数代码
        :param start_date: 开始时间
        :param end_date: 结束时间
        :return: None
        """
        interval = 60.0 / 70.0
        data_category = '指数成分和权重'
        column_desc = self._factor_param[data_category]['输出参数']
        indexes = constants.BOARD_INDEXES if indexes is None else indexes

        if not end_date:
            end_date = dt.datetime.today()
        dates = self.calendar.last_day_of_month(start_date, end_date)
        dates = sorted(list(set([start_date] + dates + [end_date])))

        logging.debug(f'开始下载{data_category}.')
        with tqdm(dates) as pbar:
            for i in range(len(dates) - 1):
                storage = []
                curr_date_str = DateUtils.date_type2str(dates[i])
                next_date_str = DateUtils.date_type2str(dates[i + 1])
                for index in indexes:
                    pbar.set_description(f'下载{curr_date_str} 到 {next_date_str} 的 {index} 的 成分股权重')
                    storage.append(self._pro.index_weight(index_code=index,
                                                          start_date=curr_date_str, end_date=next_date_str))
                    sleep(interval)
                df = self._standardize_df(pd.concat(storage), column_desc)
                self.db_interface.update_df(df, '指数成分股权重')
                pbar.update()
        logging.info(f'{data_category}下载完成.')

    # utilities
    # ------------------------------------------
    @staticmethod
    def _standardize_df(df: pd.DataFrame, parameter_info: Mapping[str, str], start_time: dt.datetime = None) \
            -> Union[pd.Series, pd.DataFrame]:
        dates_columns = [it for it in df.columns if it.endswith('date')]
        for it in dates_columns:
            df[it] = df[it].apply(DateUtils.date_type2datetime)

        df.rename(parameter_info, axis=1, inplace=True)
        index = sorted(list({'DateTime', 'ID', '报告期', 'IndexCode'} & set(df.columns)))
        if start_time and 'DateTime' in index:
            df = df.loc[(df.DateTime >= start_time) & (df.DateTime <= dt.datetime.today()), :]
        df = df.set_index(index, drop=True)
        if df.shape[1] == 1:
            df = df.iloc[:, 0]
        return df

    def update_calendar(self) -> None:
        """ 更新上交所交易日历 """
        table_name = '交易日历'
        df = self._pro.trade_cal(is_open=1)
        cal_date = df.cal_date
        cal_date.name = '交易日期'
        cal_date = cal_date.map(DateUtils.date_type2datetime)

        self.db_interface.purge_table(table_name)
        self.db_interface.insert_df(cal_date, table_name)

    def update_stock_list_date(self) -> None:
        """ 获取所有股票列表, 包括上市, 退市和暂停上市的股票

        ref: https://tushare.pro/document/2?doc_id=25

        """
        data_category = '股票列表'

        logging.debug(f'开始下载{data_category}.')
        storage = []
        list_status = ['L', 'D', 'P']
        fields = ['ts_code', 'list_date', 'delist_date']
        for status in list_status:
            storage.append(self._pro.stock_basic(exchange='', list_status=status, fields=fields))
        output = pd.concat(storage)
        output['证券类型'] = 'A股股票'
        list_info = self.format_list_date(output.loc[:, ['ts_code', 'list_date', 'delist_date', '证券类型']])
        self.db_interface.update_df(list_info, '证券代码')
        logging.info(f'{data_category}下载完成.')

    def update_convertible_bond_list_date(self) -> None:
        """ 获取可转债信息
            ref: https://tushare.pro/document/2?doc_id=185
        """
        data_category = '可转债基本信息'
        desc = self._factor_param[data_category]['输出参数']

        logging.debug(f'开始下载{data_category}.')
        output = self._pro.cb_basic(fields=list(desc.keys()))

        # list date
        list_info = output.loc[:, ['ts_code', 'list_date', 'delist_date']]
        list_info['证券类型'] = '可转债'
        list_info = self.format_list_date(list_info, extend_delist_date=True)
        self.db_interface.update_df(list_info, '证券代码')
        logging.info(f'{data_category}下载完成.')

        # names
        name_info = output.loc[:, ['list_date', 'ts_code', 'bond_short_name']].rename({'list_date': 'DateTime'},
                                                                                      axis=1).dropna()
        name_info = self._standardize_df(name_info, desc)
        self.db_interface.update_df(name_info, '证券名称')

        # info
        output = self._standardize_df(output, desc)
        self.db_interface.update_df(output, '可转债列表')
        logging.info(f'{data_category}下载完成.')

    def update_future_list_date(self) -> None:
        """ 获取期货合约
            ref: https://tushare.pro/document/2?doc_id=135
        """
        data_category = '期货合约信息表'
        desc = self._factor_param[data_category]['输出参数']

        def find_start_num(a):
            g = re.match(r'[\d.]*', a)
            return float(g.group(0))

        logging.debug(f'开始下载{data_category}.')
        storage = []
        for exchange in constants.FUTURE_EXCHANGES:
            storage.append(self._pro.fut_basic(exchange=exchange, fields=list(desc.keys()) + ['per_unit']))
        output = pd.concat(storage, ignore_index=True)
        output.multiplier = output.multiplier.where(output.multiplier.notna(), output.per_unit)
        output = output.dropna(subset=['multiplier']).drop('per_unit', axis=1)
        output.quote_unit_desc = output.quote_unit_desc.apply(find_start_num)

        # list date
        list_info = output.loc[:, ['ts_code', 'list_date', 'delist_date']]
        list_info['证券类型'] = '期货'
        list_info = self.format_list_date(list_info, extend_delist_date=True)
        self.db_interface.update_df(list_info, '证券代码')
        logging.info(f'{data_category}下载完成.')

        # names
        name_info = output.loc[:, ['list_date', 'ts_code', 'name']].rename({'list_date': 'DateTime'}, axis=1)
        name_info = self._standardize_df(name_info, desc)
        self.db_interface.update_df(name_info, '证券名称')

        # info
        output = self._standardize_df(output, desc)
        self.db_interface.update_df(output, '期货合约')
        logging.info(f'{data_category}下载完成.')

    def update_option_list_date(self) -> None:
        """ 获取期权合约
            ref: https://tushare.pro/document/2?doc_id=158
        """
        data_category = '期权合约信息'
        desc = self._factor_param[data_category]['输出参数']

        logging.debug(f'开始下载{data_category}.')
        storage = []
        for exchange in constants.FUTURE_EXCHANGES + constants.STOCK_EXCHANGES:
            storage.append(self._pro.opt_basic(exchange=exchange, fields=list(desc.keys())))
        output = pd.concat(storage)
        output.opt_code = output.opt_code.str.replace('OP', '')

        # list date
        list_info = output.loc[:, ['ts_code', 'list_date', 'delist_date', 'opt_type']]
        list_info = self.format_list_date(list_info, extend_delist_date=True)
        self.db_interface.update_df(list_info, '证券代码')
        logging.info(f'{data_category}下载完成.')

        # names
        name_info = output.loc[:, ['list_date', 'ts_code', 'name']].rename({'list_date': 'DateTime'}, axis=1)
        name_info = self._standardize_df(name_info, desc)
        self.db_interface.update_df(name_info, '证券名称')

        # info
        info = self._standardize_df(output, desc)
        self.db_interface.update_df(info, '期权合约')
        logging.info(f'{data_category}下载完成.')

    def update_fund_list_date(self) -> None:
        """ 获取基金列表
            ref: https://tushare.pro/document/2?doc_id=19
        """
        data_category = '公募基金列表'
        desc = self._factor_param[data_category]['输出参数']

        logging.debug(f'开始下载{data_category}.')
        storage = []
        for market, status in itertools.product(['E', 'O'], ['D', 'I', 'L']):
            storage.append(self._pro.fund_basic(market=market, status=status, fields=list(desc.keys())))
        output = pd.concat(storage)
        etf_type = ['ETF' if it.endswith('ETF') else '' for it in output['name']]
        openness = ['' if it == '契约型开放式' else '封闭' for it in output['type']]
        exchange_type = ['' if it == 'E' else '场外' for it in output['market']]
        end_type = '基金'
        output.fund_type = output.fund_type + etf_type + openness + exchange_type + end_type

        # list date
        exchange_part = output.loc[output.market == 'E', :]
        listed1 = exchange_part.loc[:, ['ts_code', 'list_date', 'delist_date', 'fund_type']]
        list_info1 = self.format_list_date(listed1, extend_delist_date=True)

        otc_part = output.loc[output.market == 'O', :]
        listed2 = otc_part.loc[:, ['ts_code', 'found_date', 'due_date', 'fund_type']]
        list_info2 = self.format_list_date(listed2, extend_delist_date=True)

        list_info = pd.concat([list_info1, list_info2])
        self.db_interface.update_df(list_info, '证券代码')
        logging.info(f'{data_category}下载完成.')

        # names
        exchange_name = exchange_part.loc[:, ['ts_code', 'list_date', 'name']]
        otc_name = otc_part.loc[:, ['ts_code', 'found_date', 'name']].rename({'found_date': 'list_date'}, axis=1)
        name_info = pd.concat([exchange_name, otc_name]).dropna()
        name_info.columns = ['ID', 'DateTime', '证券名称']
        name_info.DateTime = DateUtils.date_type2datetime(name_info.DateTime.tolist())
        name_info = name_info.set_index(['DateTime', 'ID'])
        self.db_interface.update_df(name_info, '证券名称')

        # info
        output = output.drop(['type', 'market'], axis=1)
        content = self._standardize_df(output, desc)
        self.db_interface.update_df(content, '基金列表')
        logging.info(f'{data_category}下载完成.')

    @staticmethod
    def format_list_date(df: pd.DataFrame, extend_delist_date: bool = False) -> pd.Series:
        df.columns = ['ID', 'list_date', 'delist_date', '证券类型']
        listed = df.loc[:, ['ID', '证券类型', 'list_date']]
        listed['上市状态'] = True
        unlisted = df.loc[:, ['ID', '证券类型', 'delist_date']].dropna().rename({'delist_date': 'list_date'}, axis=1)
        unlisted['上市状态'] = False
        if extend_delist_date:
            listed['list_date'] = DateUtils.date_type2datetime(listed['list_date'].tolist())
            unlisted['list_date'] = \
                [d + dt.timedelta(days=1) for d in DateUtils.date_type2datetime(unlisted['list_date'].tolist())]
        output = pd.concat([listed, unlisted], ignore_index=True).dropna()
        output = output.loc[output.list_date <= dt.datetime.now(), :]
        output = TushareData._standardize_df(output, {'ts_code': 'ID', 'list_date': 'DateTime'})
        return output
