import datetime as dt
import itertools
import json
import logging
import re
from functools import cached_property
from itertools import product
from typing import Callable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import tushare as ts
from ratelimiter import RateLimiter
from retrying import retry
from tqdm import tqdm

from .data_source import DataSource
from .. import config, constants, date_utils, utils
from ..database_interface import DBInterface
from ..ticker_utils import FutureTickerFormatter
from ..tickers import FundTickers, FundWithStocksTickers, StockFundTickers, StockTickers

START_DATE = {
    'common': dt.datetime(1990, 1, 1),
    'shibor': dt.datetime(2006, 10, 8),
    'ggt': dt.datetime(2016, 6, 29),  # 港股通
    'hk_cal': dt.datetime(1980, 1, 1),
    'hk_daily': dt.datetime(1990, 1, 2),
    'fund_daily': dt.datetime(1998, 4, 6),
    'index_daily': dt.datetime(2008, 1, 1),
    'index_weight': dt.datetime(2005, 1, 1)
}


class TushareData(DataSource):
    def __init__(self, tushare_token: str = None, db_interface: DBInterface = None, param_json_loc: str = None) -> None:
        """
        Tushare to Database. 将tushare下载的数据写入数据库中

        :param tushare_token: tushare token
        :param db_interface: DBInterface
        :param param_json_loc: tushare 返回df的列名信息
        """
        if tushare_token is None:
            tushare_token = config.get_global_config()['tushare']['token']
            db_interface = config.get_db_interface()
        super().__init__(db_interface)
        self.token = tushare_token
        self._pro = None
        self._factor_param = utils.load_param('tushare_param.json', param_json_loc)

    def login(self):
        self._pro = ts.pro_api(self.token)

    def logout(self):
        self._pro = ts.pro_api('')

    def init_db(self):
        """Initialize database data. They cannot be achieved by naive ``update_*`` function"""
        self.init_hk_calendar()
        self.init_stock_names()
        self.init_accounting_data()
        fund_tickers = FundTickers(self.db_interface).all_ticker()
        self.update_fund_portfolio(fund_tickers)

    def update_base_info(self):
        """Update calendar and ticker lists"""
        self.update_calendar()
        self.update_hk_calendar()
        self.update_stock_list_date()
        self.update_convertible_bond_list_date()
        # self.update_fund_list_date()
        self.update_future_list_date()
        self.update_option_list_date()

    #######################################
    # init func
    #######################################
    def init_hk_calendar(self) -> None:
        """ 更新港交所交易日历 """
        table_name = '港股交易日历'
        if self.db_interface.get_latest_timestamp(table_name):
            df = self._pro.hk_tradecal(is_open=1)
        else:
            storage = []
            end_dates = ['19850101', '19900101', '19950101', '20000101', '20050101', '20100101', '20150101', '20200101']
            for end_date in end_dates:
                storage.append(self._pro.hk_tradecal(is_open=1, end_date=end_date))
            storage.append(self._pro.hk_tradecal(is_open=1))
            df = pd.concat(storage, ignore_index=True).drop_duplicates()

        cal_date = df.cal_date
        cal_date = cal_date.sort_values()
        cal_date.name = '交易日期'
        cal_date = cal_date.map(date_utils.date_type2datetime)

        self.db_interface.update_df(cal_date, table_name)

    def init_stock_names(self):
        """获取所有股票的曾用名"""
        raw_df = self.update_stock_names()
        raw_df_start_dates = raw_df.index.get_level_values('DateTime').min()
        uncovered_stocks = self.stock_tickers.ticker(raw_df_start_dates)

        with tqdm(uncovered_stocks) as pbar:
            for stock in uncovered_stocks:
                pbar.set_description(f'下载{stock}的股票名称')
                self.update_stock_names(stock)
                pbar.update()
        logging.getLogger(__name__).info('股票曾用名下载完成.')

    @cached_property
    def stock_tickers(self) -> StockTickers:
        return StockTickers(self.db_interface)

    #######################################
    # listing funcs
    #######################################
    def update_calendar(self) -> None:
        """ 更新上交所交易日历 """
        table_name = '交易日历'
        df = self._pro.trade_cal(is_open=1)
        cal_date = df.cal_date
        cal_date.name = '交易日期'
        cal_date = cal_date.map(date_utils.date_type2datetime)

        self.db_interface.purge_table(table_name)
        self.db_interface.insert_df(cal_date, table_name)

    def update_hk_calendar(self) -> None:
        """ 更新港交所交易日历 """
        table_name = '港股交易日历'
        df = self._pro.hk_tradecal(is_open=1)
        cal_date = df.cal_date
        cal_date = cal_date.sort_values()
        cal_date.name = '交易日期'
        cal_date = cal_date.map(date_utils.date_type2datetime)
        cal_date.index.name = 'index'
        db_data = self.db_interface.read_table(table_name)
        db_data = db_data.loc[db_data['交易日期'] < cal_date.min(), :]
        data = pd.concat([db_data.set_index('index').iloc[:, 0], cal_date], ignore_index=True)

        self.db_interface.purge_table(table_name)
        self.db_interface.insert_df(data, table_name)

    def update_stock_list_date(self) -> None:
        """ 更新所有股票列表, 包括上市, 退市和暂停上市的股票

        ref: https://tushare.pro/document/2?doc_id=25

        """
        data_category = '股票列表'

        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        storage = []
        list_status = ['L', 'D', 'P']
        fields = ['ts_code', 'list_date', 'delist_date']
        for status in list_status:
            storage.append(self._pro.stock_basic(exchange='', list_status=status, fields=fields))
        output = pd.concat(storage)
        output['证券类型'] = 'A股股票'
        list_info = self._format_list_date(output.loc[:, ['ts_code', 'list_date', 'delist_date', '证券类型']])
        self.db_interface.update_df(list_info, '证券代码')
        logging.getLogger(__name__).info(f'{data_category}下载完成.')

    # TODO
    def get_hk_stock_list_date(self):
        """ 更新所有港股股票列表, 包括上市, 退市和暂停上市的股票

        ref: https://tushare.pro/document/2?doc_id=25

        """
        data_category = '股票列表'

        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        storage = []
        list_status = ['L', 'D']
        for status in list_status:
            storage.append(self._pro.hk_basic(list_status=status))
        output = pd.concat(storage)
        output['证券类型'] = '港股股票'
        list_info = self._format_list_date(output.loc[:, ['ts_code', 'list_date', 'delist_date', '证券类型']])
        self.db_interface.update_df(list_info, '证券代码')
        logging.getLogger(__name__).info(f'{data_category}下载完成.')

    def update_convertible_bond_list_date(self) -> None:
        """ 更新可转债信息
            ref: https://tushare.pro/document/2?doc_id=185
        """
        data_category = '可转债基本信息'
        desc = self._factor_param[data_category]['输出参数']

        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        output = self._pro.cb_basic(fields=list(desc.keys()))

        # list date
        list_info = output.loc[:, ['ts_code', 'list_date', 'delist_date']]
        list_info['证券类型'] = '可转债'
        list_info = self._format_list_date(list_info, extend_delist_date=True)
        self.db_interface.update_df(list_info, '证券代码')

        # names
        name_info = output.loc[:, ['list_date', 'ts_code', 'bond_short_name']].rename({'list_date': 'DateTime'},
                                                                                      axis=1).dropna()
        name_info = self._standardize_df(name_info, desc)
        self.db_interface.update_df(name_info, '证券名称')

        # info
        output = self._standardize_df(output, desc)
        self.db_interface.update_df(output, '可转债列表')
        logging.getLogger(__name__).info(f'{data_category}下载完成.')

    def update_future_list_date(self) -> None:
        """ 更新期货合约
            ref: https://tushare.pro/document/2?doc_id=135
        """
        data_category = '期货合约信息表'
        desc = self._factor_param[data_category]['输出参数']

        def find_start_num(a):
            g = re.match(r'[\d.]*', a)
            return float(g.group(0))

        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        storage = []
        for exchange in constants.FUTURE_EXCHANGES:
            storage.append(self._pro.fut_basic(exchange=exchange, fields=list(desc.keys()) + ['per_unit']))
        output = pd.concat(storage, ignore_index=True)
        output.ts_code = self.format_ticker(output['ts_code'].tolist())
        output.multiplier = output.multiplier.where(output.multiplier.notna(), output.per_unit)
        output = output.dropna(subset=['multiplier']).drop('per_unit', axis=1)
        output.quote_unit_desc = output.quote_unit_desc.apply(find_start_num)

        # exclude XINE's TAS contracts
        output = output.loc[~output.symbol.str.endswith('TAS'), :]
        # drop AP2107.CZC
        output = output.loc[output.symbol != 'AP107', :]

        db_ids = self.db_interface.get_all_id('证券代码')
        output = output.loc[~output['ts_code'].isin(db_ids), :]

        # list date
        list_info = output.loc[:, ['ts_code', 'list_date', 'delist_date']]
        list_info['证券类型'] = '期货'
        list_info = self._format_list_date(list_info, extend_delist_date=True)
        self.db_interface.insert_df(list_info, '证券代码')

        # names
        name_info = output.loc[:, ['list_date', 'ts_code', 'name']].rename({'list_date': 'DateTime'}, axis=1)
        name_info = self._standardize_df(name_info, desc)
        self.db_interface.update_df(name_info, '证券名称')

        # info
        output = self._standardize_df(output, desc)
        self.db_interface.insert_df(output, '期货合约')
        logging.getLogger(__name__).info(f'{data_category}下载完成.')

    def update_option_list_date(self) -> None:
        """ 更新期权合约
            ref: https://tushare.pro/document/2?doc_id=158
        """
        data_category = '期权合约信息'
        desc = self._factor_param[data_category]['输出参数']

        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        storage = []
        for exchange in constants.FUTURE_EXCHANGES + constants.STOCK_EXCHANGES:
            storage.append(self._pro.opt_basic(exchange=exchange, fields=list(desc.keys())))
        output = pd.concat(storage)
        output.opt_code = output.opt_code.str.replace('OP$', '', regex=True)
        output.opt_code = self.format_ticker(output['opt_code'].tolist())
        output.ts_code = self.format_ticker(output['ts_code'].tolist())

        db_ids = self.db_interface.get_all_id('证券代码')
        output = output.loc[~output['ts_code'].isin(db_ids), :]

        # list date
        list_info = output.loc[:, ['ts_code', 'list_date', 'delist_date', 'opt_type']]
        list_info = self._format_list_date(list_info, extend_delist_date=True)
        self.db_interface.insert_df(list_info, '证券代码')

        # names
        name_info = output.loc[:, ['list_date', 'ts_code', 'name']].rename({'list_date': 'DateTime'}, axis=1)
        name_info = self._standardize_df(name_info, desc)
        self.db_interface.insert_df(name_info, '证券名称')

        # info
        info = self._standardize_df(output, desc)
        self.db_interface.insert_df(info, '期权合约')
        logging.getLogger(__name__).info(f'{data_category}下载完成.')

    def update_fund_list_date(self) -> None:
        """ 更新基金列表
            ref: https://tushare.pro/document/2?doc_id=19
        """
        data_category = '公募基金列表'
        desc = self._factor_param[data_category]['输出参数']

        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        storage = []
        for market, status in itertools.product(['E', 'O'], ['D', 'I', 'L']):
            storage.append(self._pro.fund_basic(market=market, status=status, fields=list(desc.keys())))
        output = pd.concat(storage)
        output = output.loc[self.filter_valid_cn_equity_ticker(output.ts_code), :]
        etf_type = ['ETF' if it.endswith('ETF') else '' for it in output['name']]
        openness = ['' if it == '契约型开放式' else '封闭' for it in output['type']]
        exchange_type = ['' if it == 'E' else '场外' for it in output['market']]
        end_type = '基金'
        output.fund_type = output.fund_type + etf_type + openness + exchange_type + end_type

        # list date
        exchange_part = output.loc[output.market == 'E', :]
        listed1 = exchange_part.loc[:, ['ts_code', 'list_date', 'delist_date', 'fund_type']]
        list_info1 = self._format_list_date(listed1, extend_delist_date=True)

        otc_part = output.loc[output.market == 'O', :]
        listed2 = otc_part.loc[:, ['ts_code', 'found_date', 'due_date', 'fund_type']]
        list_info2 = self._format_list_date(listed2, extend_delist_date=True)

        list_info = pd.concat([list_info1, list_info2])
        self.db_interface.update_df(list_info, '证券代码')

        # names
        exchange_name = exchange_part.loc[:, ['ts_code', 'list_date', 'name']]
        otc_name = otc_part.loc[:, ['ts_code', 'found_date', 'name']].rename({'found_date': 'list_date'}, axis=1)
        name_info = pd.concat([exchange_name, otc_name]).dropna()
        name_info.columns = ['ID', 'DateTime', '证券名称']
        name_info.DateTime = date_utils.date_type2datetime(name_info['DateTime'].tolist())
        name_info = name_info.set_index(['DateTime', 'ID'])
        self.db_interface.update_df(name_info, '证券名称')

        # info
        output = output.drop(['type', 'market'], axis=1)
        content = self._standardize_df(output, desc)
        self.db_interface.purge_table('基金列表')
        self.db_interface.insert_df(content, '基金列表')
        logging.getLogger(__name__).info(f'{data_category}下载完成.')

    #######################################
    # interest funcs
    #######################################
    @date_utils.strlize_input_dates
    def get_shibor(self, start_date: date_utils.DateType = None, end_date: date_utils.DateType = None) -> pd.DataFrame:
        """ Shibor利率数据 """
        data_category = 'Shibor利率数据'
        desc = self._factor_param[data_category]['输出参数']

        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        df = self._pro.shibor(start_date=start_date, end_date=end_date)
        df = self._standardize_df(df, desc)
        self.db_interface.update_df(df, data_category)
        logging.getLogger(__name__).info(f'{data_category}下载完成.')
        return df

    #######################################
    # stock funcs
    #######################################
    def get_company_info(self) -> pd.DataFrame:
        """
        获取上市公司基本信息

        :ref: https://tushare.pro/document/2?doc_id=112

        :return: 上市公司基础信息df
        """
        data_category = '上市公司基本信息'
        column_desc = self._factor_param[data_category]['输出参数']
        fields = list(column_desc.keys())

        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        storage = []
        for exchange in constants.STOCK_EXCHANGES:
            storage.append(self._pro.stock_company(exchange=exchange, fields=fields))
        df = pd.concat(storage)
        df = self._standardize_df(df, column_desc)
        self.db_interface.update_df(df, data_category)
        logging.getLogger(__name__).info(f'{data_category}下载完成.')
        return df

    @date_utils.strlize_input_dates
    def get_ipo_info(self, start_date: date_utils.DateType = None) -> pd.DataFrame:
        """ IPO新股列表 """
        data_category = 'IPO新股列表'
        column_desc = self._factor_param[data_category]['输出参数']

        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        df = self._pro.new_share(start_date=start_date)
        df[['amount', 'market_amount', 'limit_amount']] = df[['amount', 'market_amount', 'limit_amount']] * 10000
        df['funds'] = df['funds'] * 100000000

        # list_date
        list_date_data = df.loc[df.issue_date != '', ['issue_date', 'ts_code']]
        list_date_data['证券类型'] = 'A股股票'
        list_date_data['上市状态'] = True
        list_date_data = self._standardize_df(list_date_data, {'issue_date': 'DateTime', 'ts_code': 'ID'})
        list_date_data = list_date_data.loc[list_date_data.index.get_level_values('DateTime') < dt.datetime.now(), :]
        self.db_interface.update_df(list_date_data, '证券代码')

        # info
        df = self._standardize_df(df, column_desc)
        self.db_interface.update_df(df, data_category)
        logging.getLogger(__name__).info(f'{data_category}下载完成.')
        return df

    def update_stock_names(self, ticker: str = None) -> pd.DataFrame:
        """更新曾用名

        ref: https://tushare.pro/document/2?doc_id=100

        :param ticker: 证券代码(000001.SZ)
        """
        data_category = '证券名称'
        column_desc = self._factor_param[data_category]['输出参数']
        fields = list(column_desc.keys())

        logging.getLogger(__name__).debug(f'开始下载{ticker if ticker else ""}{data_category}.')
        df = self._pro.namechange(ts_code=ticker, fields=fields)
        df = self._standardize_df(df, column_desc)
        self.db_interface.update_df(df, data_category)
        logging.getLogger(__name__).debug(f'{ticker if ticker else ""}{data_category}下载完成.')
        return df

    def get_daily_hq(self, trade_date: date_utils.DateType = None,
                     start_date: date_utils.DateType = None, end_date: date_utils.DateType = None) -> None:
        """更新每日行情

        行情信息包括: 开高低收, 量额, 复权因子, 股本

        :param trade_date: 交易日期
        :param start_date: 开始日期
        :param end_date: 结束日期

        交易日期查询一天, 开始结束日期查询区间. 二选一

        :return: None
        """
        if (not trade_date) & (not start_date):
            raise ValueError('trade_date 和 start_date 必填一个!')
        if end_date is None:
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
                current_date_str = date_utils.date_type2str(date)
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

    def update_pause_stock_info(self):
        """更新股票停牌信息"""
        table_name = '股票停牌'
        renaming_dict = self._factor_param[table_name]['输出参数']
        start_date = self.db_interface.get_latest_timestamp(table_name, dt.date(1990, 12, 10)) + dt.timedelta(days=1)
        end_date = self.calendar.yesterday()

        df = self._pro.suspend_d(start_date=date_utils.date_type2str(start_date),
                                 end_date=date_utils.date_type2str(end_date),
                                 suspend_type='S')
        output = df.loc[pd.isna(df.suspend_timing), ['ts_code', 'trade_date']]
        output['停牌类型'] = '停牌一天'
        output['停牌原因'] = ''
        output = self._standardize_df(output, renaming_dict)
        self.db_interface.insert_df(output, table_name)

    def get_all_dividend(self) -> None:
        """ 获取上市公司分红送股信息 """
        data_category = '分红送股'
        column_desc = self._factor_param[data_category]['输出参数']

        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        tickers = self.stock_tickers.all_ticker()
        with tqdm(tickers) as pbar:
            for stock in tickers:
                pbar.set_description(f'下载{stock}的分红送股数据')
                df = self._pro.dividend(ts_code=stock, fields=(list(column_desc.keys())))
                df = df.loc[df['div_proc'] == '实施', :]
                # 无公布时间的权宜之计
                df['ann_date'].where(df['ann_date'].notnull(), df['imp_ann_date'], inplace=True)
                df.drop(['div_proc', 'imp_ann_date'], axis=1, inplace=True)
                df.dropna(subset=['ann_date'], inplace=True)
                df = self._standardize_df(df, column_desc)
                df = df.drop_duplicates()

                try:
                    self.db_interface.insert_df(df, data_category)
                except Exception:
                    print(f'请手动处理{stock}的分红数据')

                pbar.update()

        logging.getLogger(__name__).info(f'{data_category}信息下载完成.')

    def update_dividend(self) -> None:
        """ 更新上市公司分红送股信息 """
        data_category = '分红送股'
        column_desc = self._factor_param[data_category]['输出参数']

        db_date = self.db_interface.get_column_max(data_category, '股权登记日')
        dates_range = self.calendar.select_dates(db_date, dt.date.today(), inclusive=(False, True))
        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        with tqdm(dates_range) as pbar:
            for date in dates_range:
                pbar.set_description(f'下载{date}的分红送股数据')
                date_str = date_utils.date_type2str(date)
                df = self._pro.dividend(record_date=date_str, fields=(list(column_desc.keys())))
                df = df.loc[df['div_proc'] == '实施', :]
                # 无公布时间的权宜之计
                df['ann_date'].where(df['ann_date'].notnull(), df['imp_ann_date'], inplace=True)
                df.drop(['div_proc', 'imp_ann_date'], axis=1, inplace=True)
                df = self._standardize_df(df, column_desc)
                self.db_interface.update_df(df, data_category)
                pbar.update()

        logging.getLogger(__name__).info(f'{data_category}信息下载完成.')

    def get_financial(self, ticker: str) -> None:
        """ 获取公司的 资产负债表, 现金流量表 和 利润表, 并写入数据库 """
        balance_sheet = '资产负债表'
        income = '利润表'
        cash_flow = '现金流量表'

        balance_sheet_desc = self._factor_param[balance_sheet]['输出参数']
        income_desc = self._factor_param[income]['输出参数']
        cash_flow_desc = self._factor_param[cash_flow]['输出参数']
        company_type_desc = self._factor_param[balance_sheet]['公司类型']

        combined_types = ['1', '4', '5', '11']
        mother_types = ['6', '9', '10', '12']

        def download_data(api_func: Callable, report_type_list: Sequence[str],
                          column_name_dict: Mapping[str, str], table_name: str) -> None:
            storage = []
            for i in report_type_list:
                storage.append(api_func(ts_code=ticker, report_type=i, fields=list(column_name_dict.keys())))
            df = pd.concat(storage, ignore_index=True)
            if df.empty:  # 000508 无数据
                return
            df = df.dropna(subset=['ann_date', 'f_ann_date', 'end_date'])  # 000166 申万宏源 早期数据无时间戳
            df = df.sort_values('update_flag').groupby(['ann_date', 'end_date', 'report_type']).tail(1)
            df = df.drop('update_flag', axis=1).fillna(np.nan).replace(0, np.nan).dropna(how='all', axis=1)
            df = df.set_index(['ann_date', 'f_ann_date', 'report_type']).sort_index().drop_duplicates(
                keep='first').reset_index()
            df = df.sort_values('report_type').drop(['ann_date', 'report_type'], axis=1)
            df = df.replace({'comp_type': company_type_desc})
            df = self._standardize_df(df, column_name_dict)
            df = df.loc[~df.index.duplicated(), :].sort_index()
            df = df.loc[df.index.get_level_values('报告期').month % 3 == 0, :]
            df = self.append_report_date_cache(df)

            self.db_interface.delete_id_records(table_name, ticker)
            try:
                self.db_interface.insert_df(df, table_name)
            except BaseException:
                logging.getLogger(__name__).error(f'{ticker} - {table_name} failed to get coherent data')

        loop_vars = [(self._pro.income, income_desc, income),
                     (self._pro.cashflow, cash_flow_desc, cash_flow),
                     (self._pro.balancesheet, balance_sheet_desc, balance_sheet)]
        for f, desc, table in loop_vars:
            download_data(f, mother_types, desc, f'母公司{table}')
            download_data(f, combined_types, desc, f'合并{table}')

    def init_accounting_data(self):
        tickers = self.stock_tickers.all_ticker()
        db_ticker = self.db_interface.get_column_max('合并资产负债表', 'ID')
        if db_ticker:
            tickers = tickers[tickers.index(db_ticker):]

        rate_limiter = RateLimiter(self._factor_param['资产负债表']['每分钟限速'] / 8, 60)
        logging.getLogger(__name__).debug('开始下载财报.')
        with tqdm(tickers) as pbar:
            for ticker in tickers:
                with rate_limiter:
                    pbar.set_description(f'下载 {ticker} 的财务数据')
                    self.get_financial(ticker)
                    pbar.update()

        logging.getLogger(__name__).info('财报下载完成')

    @retry(stop_max_attempt_number=3)
    def update_financial_data(self, date: dt.datetime = None):
        table_name = '财报披露计划'
        desc = self._factor_param[table_name]['输出参数']
        ref_table = '合并资产负债表'
        db_data = self.db_interface.read_table(ref_table, '期末总股本')
        latest = db_data.groupby('ID').tail(1).reset_index().loc[:, ['DateTime', 'ID']].rename({'ID': 'ts_code'},
                                                                                               axis=1)
        update_tickers = set(self.stock_tickers.all_ticker()) - set(latest.ts_code.tolist())

        report_dates = date_utils.ReportingDate.get_latest_report_date(date)
        for report_date in report_dates:
            df = self._pro.disclosure_date(end_date=date_utils.date_type2str(report_date), fields=list(desc.keys()))
            df.actual_date = df.actual_date.apply(date_utils.date_type2datetime)
            tmp = df.modify_date.str.split(',').apply(lambda x: x[-1] if x else None)
            df.modify_date = tmp.apply(date_utils.date_type2datetime)
            df2 = df.merge(latest, on='ts_code', how='left')
            df3 = df2.loc[(df2.actual_date > df2.DateTime) | (df2.modify_date > df2.DateTime), :]
            if not df3.empty:
                update_tickers = update_tickers.union(set(df3.ts_code.tolist()))

        with tqdm(update_tickers) as pbar:
            for ticker in update_tickers:
                pbar.set_description(f'更新 {ticker} 的财报')
                self.get_financial_index(ticker)
                self.get_financial(ticker)
                pbar.update()

    def get_financial_index(self, ticker: str) -> Optional[pd.DataFrame]:
        """ 获取财务指标

        :param ticker: 证券代码(000001.SZ)
        :return: 财务指标
        """
        data_category = '财务指标'
        column_desc = self._factor_param[data_category]['输出参数']

        df = self._pro.fina_indicator(ts_code=ticker, fields=list(column_desc.keys()))
        if df.empty:
            return
        df = df.dropna(subset=['ann_date', 'end_date'])
        df = df.sort_values('update_flag').groupby(['ann_date', 'end_date']).tail(1)
        df = df.drop('update_flag', axis=1).fillna(np.nan).replace(0, np.nan).dropna(how='all', axis=1)
        df = self._standardize_df(df, column_desc).sort_index()
        df = df.loc[df.index.get_level_values('报告期').month % 3 == 0, :]
        df = self.append_report_date_cache(df)
        self.db_interface.delete_id_records(data_category, ticker)
        self.db_interface.insert_df(df, data_category)
        return df

    def get_hs_constitute(self) -> None:
        """ 沪深股通成分股进出记录. 月末更新. """
        data_category = '沪深股通成份股'
        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        storage = []
        for hs_type, is_new in product(['SH', 'SZ'], ['0', '1']):
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
        logging.getLogger(__name__).info(f'{data_category}数据下载完成')

    @date_utils.strlize_input_dates
    def get_hs_holding(self, date: date_utils.DateType):
        data_category = '沪深港股通持股明细'
        desc = self._factor_param[data_category]['输出参数']
        fields = list(desc.keys())

        df = self._pro.hk_hold(trade_date=date, fields=fields)
        df = self._standardize_df(df, desc)
        self.db_interface.update_df(df, data_category)

    def update_hs_holding(self) -> None:
        """ 沪深港股通持股明细 """
        data_category = '沪深港股通持股明细'
        start_date = self.db_interface.get_latest_timestamp(data_category, START_DATE['ggt'])
        dates = self.calendar.select_dates(start_date, dt.date.today())

        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        with tqdm(dates) as pbar:
            for date in dates:
                pbar.set_description(f'下载{date}的沪深港股通持股明细')
                self.get_hs_holding(date)
                pbar.update()
        logging.getLogger(__name__).info(f'{data_category}下载完成.')

    #######################################
    # HK stock funcs
    #######################################
    def update_hk_stock_daily(self):
        table_name = '港股日行情'

        hk_cal = date_utils.HKTradingCalendar(self.db_interface)
        start_date = self.db_interface.get_latest_timestamp(table_name, START_DATE['hk_daily'])
        end_date = hk_cal.yesterday()
        dates = hk_cal.select_dates(start_date=start_date, end_date=end_date, inclusive=(False, True))

        rate = self._factor_param[table_name]['每分钟限速']
        rate_limiter = RateLimiter(rate, 60)

        with tqdm(dates) as pbar:
            for date in dates:
                with rate_limiter:
                    pbar.set_description(f'下载{date}的{table_name}')
                    self.get_hk_stock_daily(date)
                    pbar.update()

    @date_utils.strlize_input_dates
    def get_hk_stock_daily(self, date: date_utils.DateType) -> pd.DataFrame:
        table_name = '港股日行情'
        desc = self._factor_param[table_name]['输出参数']
        df = self._pro.hk_daily(trade_date=date, fields=list(desc.keys()))
        price_df = self._standardize_df(df, desc)
        self.db_interface.insert_df(price_df, table_name)
        return price_df

    #######################################
    # index funcs
    #######################################
    @date_utils.strlize_input_dates
    def get_index_daily(self, date: date_utils.DateType) -> None:
        """
        获取指数行情信息. 包括开高低收, 量额, 市盈, 市净, 市值
        默认指数为沪指, 深指, 中小盘, 创业板, 50, 300, 500
        注: 300不包含市盈等指标

        :param date: 日期
        :return: 指数行情信息
        """
        table_name = '指数日行情'
        desc = self._factor_param[table_name]['输出参数']
        price_fields = list(desc.keys())
        basic_desc = self._factor_param['大盘指数每日指标']['输出参数']
        basic_fields = list(basic_desc.keys())

        storage = []
        indexes = list(constants.STOCK_INDEXES.values())
        for index in indexes:
            storage.append(self._pro.index_daily(ts_code=index, start_date=date, end_date=date, fields=price_fields))
        price_info = pd.concat(storage)
        price_info['vol'] = price_info['vol'] * 100
        price_info['amount'] = price_info['amount'] * 1000
        price_info = self._standardize_df(price_info, desc)

        valuation_info = self._pro.index_dailybasic(trade_date=date, fields=basic_fields)
        valuation_info = self._standardize_df(valuation_info, basic_desc)
        data = pd.concat([price_info, valuation_info], axis=1)
        data = data.loc[data.index.get_level_values('ID').isin(indexes), :]
        self.db_interface.insert_df(data, table_name)

    def update_index_daily(self):
        table_name = '指数日行情'
        start_date = self.db_interface.get_latest_timestamp(table_name, START_DATE['index_daily'])
        dates = self.calendar.select_dates(start_date, dt.date.today(), inclusive=(False, True))

        rate = self._factor_param[table_name]['每分钟限速']
        rate_limiter = RateLimiter(rate, period=60)

        logging.getLogger(__name__).debug(f'开始下载{table_name}.')
        with tqdm(dates) as pbar:
            for date in dates:
                with rate_limiter:
                    pbar.set_description(f'下载{date}的{table_name}')
                    self.get_index_daily(date)
                    pbar.update()
        logging.getLogger(__name__).info(f'{table_name}下载完成')

    @date_utils.dtlize_input_dates
    def get_index_weight(self, indexes: Sequence[str] = None,
                         start_date: date_utils.DateType = None, end_date: date_utils.DateType = None) -> None:
        """ 指数成分和权重

        默认指数为 ['000016.SH', '399300.SH', '000905.SH'], 即50, 300, 500

        :param indexes: 指数代码
        :param start_date: 开始时间
        :param end_date: 结束时间
        :return: None
        """
        data_category = '指数成分和权重'
        column_desc = self._factor_param[data_category]['输出参数']
        indexes = constants.BOARD_INDEXES if indexes is None else indexes

        if end_date is None:
            end_date = dt.datetime.today()
        dates = self.calendar.last_day_of_month(start_date, end_date)
        dates = sorted(list(set([start_date] + dates + [end_date])))

        logging.getLogger(__name__).debug(f'开始下载{data_category}.')
        with tqdm(dates) as pbar:
            for i in range(len(dates) - 1):
                storage = []
                curr_date_str = date_utils.date_type2str(dates[i])
                next_date_str = date_utils.date_type2str(dates[i + 1])
                for index in indexes:
                    pbar.set_description(f'下载{curr_date_str} 到 {next_date_str} 的 {index} 的 成分股权重')
                    storage.append(self._pro.index_weight(index_code=index,
                                                          start_date=curr_date_str, end_date=next_date_str))
                df = self._standardize_df(pd.concat(storage), column_desc)
                self.db_interface.update_df(df, '指数成分股权重')
                pbar.update()
        logging.getLogger(__name__).info(f'{data_category}下载完成.')

    #######################################
    # future funcs
    #######################################
    # TODO
    @date_utils.strlize_input_dates
    def _get_future_settle_info(self, date):
        table_name = '期货结算参数'
        desc = self._factor_param[table_name]['输出参数']
        storage = []
        for exchange in constants.FUTURE_EXCHANGES:
            storage.append(self._pro.fut_settle(trade_date=date, exchange=exchange, fields=list(desc.keys())))
        data = pd.concat(storage, ignore_index=True)
        df = self._standardize_df(data, desc)
        return df

    #######################################
    # funds funcs
    #######################################
    def update_fund_daily(self):
        """更新基金日行情"""
        daily_table_name = '场内基金日行情'
        nav_table_name = '公募基金净值'
        asset_table_name = '基金规模数据'
        daily_params = self._factor_param[daily_table_name]['输出参数']
        nav_params = self._factor_param[nav_table_name]['输出参数']
        share_params = self._factor_param[asset_table_name]['输出参数']

        start_date = self.db_interface.get_latest_timestamp(daily_table_name, START_DATE['fund_daily'])
        start_date = self.calendar.offset(start_date, -4)
        end_date = dt.date.today()
        dates = self.calendar.select_dates(start_date, end_date, (False, True))
        rate = self._factor_param[nav_table_name]['每分钟限速']
        rate_limiter = RateLimiter(rate, period=60)
        with tqdm(dates) as pbar:
            for date in dates:
                with rate_limiter:
                    pbar.set_description(f'下载{date}的{daily_table_name}')
                    date_str = date_utils.date_type2str(date)

                    daily_data = self._pro.fund_daily(trade_date=date_str, fields=list(daily_params.keys()))
                    daily_data['vol'] = daily_data['vol'] * 100
                    daily_data['amount'] = daily_data['amount'] * 1000
                    daily_data = self._standardize_df(daily_data, daily_params)

                    ex_nav_data = self._pro.fund_nav(nav_date=date_str, market='E')
                    ex_nav_part = ex_nav_data.loc[:, ['ts_code', 'nav_date', 'unit_nav']]
                    ex_nav_part = self._standardize_df(ex_nav_part, nav_params)

                    share_data = self._pro.fund_share(trade_date=date_str)
                    share_data['fd_share'] = share_data['fd_share'] * 10000
                    ind = share_data['market'] == 'O'
                    share_data.drop(['fund_type', 'market'], axis=1, inplace=True)
                    share_data = self._standardize_df(share_data, share_params)
                    ex_share_data = share_data.loc[~ind, :]
                    of_share_data = share_data.loc[ind, :]

                    db_data = daily_data.join(ex_nav_part, how='left').join(ex_share_data, how='left')

                    nav_data = self._pro.fund_nav(nav_date=date_str, market='O', fields=list(nav_params.keys()))
                    nav_data = nav_data.loc[~(pd.isna(nav_data['accum_nav']) & nav_data['unit_nav'] == 1), :].drop(
                        'accum_nav', axis=1)
                    nav_part = self._standardize_df(nav_data.iloc[:, :3].copy(), nav_params)
                    asset_part = self._standardize_df(nav_data.iloc[:, [0, 1, 3, 4]].dropna(), nav_params)

                    self.db_interface.update_df(db_data, daily_table_name)
                    self.db_interface.update_df(nav_part, '场外基金净值')
                    self.db_interface.update_df(of_share_data, '场外基金份额')
                    self.db_interface.update_df(asset_part, '场外基金规模')

                    pbar.update()
        logging.getLogger(__name__).info(f'{daily_table_name} 更新完成.')

    def update_fund_asset(self, tickers: Sequence[str] = None):
        if tickers is None:
            tickers = FundTickers(self.db_interface).all_ticker()
        tickers = [it for it in tickers if len(it) == 9]
        asset_table_name = '场外基金份额'
        share_params = self._factor_param[asset_table_name]['输出参数']
        rate = self._factor_param[asset_table_name]['每分钟限速'] - 1
        rate_limiter = RateLimiter(rate, period=60)
        storage = []
        with tqdm(tickers) as pbar:
            for ticker in tickers:
                with rate_limiter:
                    pbar.set_description(f'更新 {ticker} 的份额')
                    storage.append(self._pro.fund_share(ts_code=ticker))
                    pbar.update()
        share_data = pd.concat(storage)
        share_data['fd_share'] = share_data['fd_share'] * 10000
        ind = share_data['market'] == 'O'
        share_data.drop(['fund_type', 'market'], axis=1, inplace=True)
        share_data = self._standardize_df(share_data, share_params)
        ex_share_data = share_data.loc[~ind, :]
        ex_to_of_share_data = self.generate_of_data_from_exchange_data(ex_share_data)
        of_share_data = pd.concat([share_data.loc[ind, :], ex_to_of_share_data])

        self.db_interface.update_df(ex_share_data, '场内基金日行情')
        self.db_interface.update_df(of_share_data, '场外基金份额')

    def update_fund_dividend(self):
        """更新基金分红信息"""
        table_name = '公募基金分红'
        params = self._factor_param[table_name]['输出参数']
        rate = self._factor_param[table_name]['每分钟限速']

        start_date = self.db_interface.get_latest_timestamp(table_name, dt.date(1999, 3, 31))
        end_date = dt.date.today()
        dates = self.calendar.select_dates(start_date, end_date, (False, True))
        with tqdm(dates) as pbar:
            rate_limiter = RateLimiter(rate - 1, period=60)
            for date in dates:
                with rate_limiter:
                    pbar.set_description(f'下载{date}的{table_name}')
                    df = self._pro.fund_div(ex_date=date_utils.date_type2str(date), fields=list(params.keys()))
                    df = df.dropna().drop_duplicates()

                    shsz_part = df.loc[~df['ts_code'].str.endswith('.OF'), ['ts_code', 'ex_date', 'div_cash']].copy()
                    shsz_part = self._standardize_df(shsz_part, params)

                    of_part = df.loc[:, ['ts_code', 'net_ex_date', 'div_cash']].copy()
                    of_part['ts_code'] = of_part['ts_code'].str.replace('SZ|SH', 'OF', regex=True)
                    of_part = self._standardize_df(of_part, params)

                    data = pd.concat([of_part, shsz_part])
                    self.db_interface.update_df(data, table_name)
                    pbar.update()
        logging.getLogger(__name__).info(f'{table_name} 更新完成.')

    def get_update_fund_portfolio_info(self):
        tickers = StockFundTickers(self.db_interface).ticker()
        fund_manager_info = self.db_interface.read_table('基金列表', ids=tickers)
        managers = fund_manager_info['管理人'].drop_duplicates().tolist()
        update_managers = []
        table_name = '公募基金持仓'
        rate = self._factor_param[table_name]['每分钟限速']

        rate_limiter = RateLimiter(rate - 1, period=60)
        report_period = None
        for manager in managers:
            for investment_type in ['灵活配置型基金', '偏股混合型基金']:
                mask = (fund_manager_info['管理人'] == manager) & (fund_manager_info['投资类型'] == investment_type)
                tickers = fund_manager_info.loc[mask, :].index.tolist()
                checker = False
                for ticker in tickers:
                    holding = self.db_interface.read_table('公募基金持仓', ids=ticker)
                    if not holding.empty:
                        db_reporting_period = holding.index.get_level_values('报告期')[0]
                        if db_reporting_period >= date_utils.ReportingDate.get_latest_report_date()[-1]:
                            break
                        with rate_limiter:
                            df = self._pro.fund_portfolio(ts_code=ticker)
                            tmp = date_utils.date_type2datetime(df['end_date'][0])
                            if tmp >= db_reporting_period:
                                report_period = tmp
                                update_managers.append(manager)
                                checker = True
                                break
                if checker:
                    break

        tickers = FundWithStocksTickers(self.db_interface).ticker()
        storage = []
        for manager in update_managers:
            man_ticker = set(tickers) & set(
                fund_manager_info.loc[fund_manager_info['管理人'] == manager, :].index.tolist())
            storage.extend(list(man_ticker))
        self.update_fund_portfolio(storage[25:], report_period)

    def update_fund_portfolio(self, tickers: Sequence[str] = None, end_date: dt.datetime = None):
        """更新公募基金持仓数据

        PS: 每个基金只保存最近的4000条记录
        """
        table_name = '公募基金持仓'
        params = self._factor_param[table_name]['输出参数']
        rate = self._factor_param[table_name]['每分钟限速']
        if tickers is None:
            tickers = FundTickers(self.db_interface).ticker()

        with tqdm(tickers) as pbar:
            rate_limiter = RateLimiter(rate - 1, period=60)
            for ticker in tickers:
                with rate_limiter:
                    pbar.set_description(f'下载{ticker}的{table_name}')
                    df = self._pro.fund_portfolio(ts_code=ticker, fields=list(params.keys()))
                    df = df.drop_duplicates(subset=list(params.keys())[:4])
                    df = self._standardize_df(df, params)
                    if end_date:
                        df = df.loc[df.index.get_level_values('报告期') == end_date, :]
                    try:
                        self.db_interface.insert_df(df, table_name)
                    except BaseException:
                        pass
                    pbar.update()
        logging.getLogger(__name__).info(f'{table_name} 更新完成.')

    #######################################
    # us stock funcs
    #######################################
    def get_us_stock(self, date: date_utils.DateType):
        """获取美股日行情"""
        table_name = '美股日行情'
        desc = self._factor_param[table_name]['输出参数']

        current_date_str = date_utils.date_type2str(date)
        df = self._pro.us_daily(trade_date=current_date_str, fields=list(desc.keys()))
        price_df = self._standardize_df(df, desc)
        self.db_interface.update_df(price_df, table_name)

    #######################################
    # utils funcs
    #######################################
    @staticmethod
    def _standardize_df(df: pd.DataFrame, parameter_info: Mapping[str, str]) -> Union[pd.Series, pd.DataFrame]:
        dates_columns = [it for it in df.columns if it.endswith('date')]
        for it in dates_columns:
            df[it] = df[it].apply(date_utils.date_type2datetime)

        df.rename(parameter_info, axis=1, inplace=True)
        index = [it for it in ['DateTime', 'ID', 'ConstituteTicker', '报告期'] if it in df.columns]
        df = df.set_index(index, drop=True)
        if df.shape[1] == 1:
            df = df.iloc[:, 0]
        return df

    @staticmethod
    def _format_list_date(df: pd.DataFrame, extend_delist_date: bool = False) -> pd.Series:
        df.columns = ['ID', 'list_date', 'delist_date', '证券类型']
        listed = df.loc[:, ['ID', '证券类型', 'list_date']]
        listed['上市状态'] = True
        unlisted = df.loc[:, ['ID', '证券类型', 'delist_date']].dropna().rename({'delist_date': 'list_date'}, axis=1)
        unlisted['上市状态'] = False
        if extend_delist_date:
            listed['list_date'] = date_utils.date_type2datetime(listed['list_date'].tolist())
            unlisted['list_date'] = \
                [d + dt.timedelta(days=1) for d in date_utils.date_type2datetime(unlisted['list_date'].tolist())]
        output = pd.concat([listed, unlisted], ignore_index=True).dropna()
        output = TushareData._standardize_df(output, {'ts_code': 'ID', 'list_date': 'DateTime'})
        output = output.loc[output.index.get_level_values('DateTime') <= dt.datetime.now(), :]
        return output

    @staticmethod
    def format_ticker(tickers: Union[Sequence[str], str]) -> Union[Sequence[str], str]:
        if isinstance(tickers, str):
            return TushareData._format_ticker(tickers)
        else:
            return [TushareData._format_ticker(it) for it in tickers]

    @staticmethod
    def _format_ticker(ticker: str) -> str:
        ticker = ticker.replace('.CFX', '.CFE').replace('.ZCE', '.CZC')
        if ticker.endswith('.CZC') and len(ticker) <= 10:
            ticker = FutureTickerFormatter().format_czc_ticker(ticker)
        return ticker

    @classmethod
    def from_config(cls, config_loc: str):
        with open(config_loc, 'r', encoding='utf-8') as f:
            mod_config = json.load(f)
        db_interface = config.generate_db_interface_from_config(config_loc)
        return cls(db_interface=db_interface, tushare_token=mod_config['tushare']['token'])

    @staticmethod
    def append_report_date_cache(data: pd.DataFrame) -> pd.DataFrame:
        dates = data.index.get_level_values('DateTime').unique()
        storage = []

        def deal_data(report_date, offset_str: str):
            pre_report_date = date_utils.ReportingDate.offset(report_date, offset_str)
            pre_record = info.loc[info.index.get_level_values('报告期') == pre_report_date].tail(1)
            pre_date = None if pre_record.empty else pre_record.index.get_level_values('DateTime')[0]
            return pre_date

        for date in dates:
            info = data.loc[data.index.get_level_values('DateTime') <= date, :]
            newest_entry = info.tail(1)
            report_date = pd.to_datetime(newest_entry.index.get_level_values('报告期')[0])

            # quarterly
            q1 = deal_data(report_date, 'q1')
            q2 = deal_data(report_date, 'q2')
            q4 = deal_data(report_date, 'q4')
            q5 = deal_data(report_date, 'q5')

            # yearly
            y1 = deal_data(report_date, 'y1')
            y2 = deal_data(report_date, 'y2')
            y3 = deal_data(report_date, 'y3')
            y5 = deal_data(report_date, 'y5')

            res = pd.DataFrame([q1, q2, q4, q5, y1, y2, y3, y5]).T.set_index(newest_entry.index)
            res.columns = ['q1', 'q2', 'q4', 'q5', 'y1', 'y2', 'y3', 'y5']
            storage.append(res)

        cache = pd.concat(storage)
        return pd.concat([data, cache], axis=1)

    @staticmethod
    def filter_valid_cn_equity_ticker(tickers: Sequence[str]) -> List[bool]:
        return [len(ticker) == 9 for ticker in tickers]

    @staticmethod
    def generate_of_data_from_exchange_data(exchange_data: Union[pd.Series, pd.DataFrame]):
        new_id = exchange_data.index.get_level_values('ID').str.replace('SH|SZ', 'OF', regex=True)
        new_index = pd.MultiIndex.from_arrays([exchange_data.index.get_level_values('DateTime'), new_id.values],
                                              names=['DateTime', 'ID'])
        exchange_data.index = new_index
        return exchange_data
