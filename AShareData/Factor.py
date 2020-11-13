import datetime as dt
import os
import pickle
import zipfile
from cached_property import cached_property
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import pandas as pd

from . import constants, DateUtils, utils
from .DBInterface import DBInterface


class Factor(object):
    """
    Factor base class
    """

    def __init__(self, db_interface: DBInterface, table_name: str, factor_names: Union[str, Sequence[str]]):
        super().__init__()
        self.table_name = table_name
        self.factor_names = factor_names

        self.db_interface = db_interface
        self.calendar = DateUtils.TradingCalendar(db_interface)

    def get_data(self, *args, **kwargs):
        """获取数据"""
        raise NotImplementedError()

    # helper functions
    def _check_args(self, table_name: str, factor_names: Union[str, Sequence[str]]):
        table_name = table_name.lower()
        assert self.db_interface.exist_table(table_name), f'数据库中不存在表 {table_name}'

        if factor_names:
            columns = self.db_interface.get_columns_names(table_name)
            if isinstance(factor_names, str):
                assert factor_names in columns, f'表 {table_name} 中不存在 {factor_names} 列'
            else:
                for name in factor_names:
                    assert name in columns, f'表 {table_name} 中不存在 {name} 列'


class NonFinancialFactor(Factor):
    """
    非财报数据
    """

    def __init__(self, db_interface: DBInterface, table_name: str, factor_names: Union[str, Sequence[str]] = None):
        super().__init__(db_interface, table_name, factor_names)
        self._check_args(table_name, factor_names)
        assert not any([it in table_name for it in constants.FINANCIAL_STATEMENTS_TYPE]), \
            f'{table_name} 为财报数据, 请使用 FinancialFactor 类!'

    def get_data(self, *args, **kwargs):
        raise NotImplementedError()


class CompactFactor(NonFinancialFactor):
    """
    Compact Factors

    数字变动很不平常的特性, 列如复权因子, 行业, 股本 等. 对于的数据库表格为: {'DateTime', 'ID', 'FactorName'}
    该类可以缓存以提升效率
    """

    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name, factor_name)
        self.data = db_interface.read_table(factor_name)

    def get_data(self, dates: Union[Sequence[dt.datetime], DateUtils.DateType] = None,
                 start_date: DateUtils.DateType = None, end_date: DateUtils.DateType = None,
                 ids: Union[Sequence[str], str] = None) -> pd.DataFrame:
        """
        :param start_date: start date
        :param end_date: end date
        :param dates: selected dates
        :param ids: query stocks
        :return: pandas.DataFrame with DateTime as index and stock as column
        """
        if isinstance(dates, dt.datetime):
            dates = [dates]
        data = self.data.copy()
        if ids:
            data = data.loc[(slice(None), ids)]
        if dates:
            end_date = max(dates)
            start_date = min(dates)
        if not end_date:
            end_date = dt.datetime.today()
        if not start_date:
            start_date = data.index.get_level_values('DateTime').min()

        previous_data = data.loc[data.index.get_level_values('DateTime') <= start_date].groupby('ID').tail(1)
        index = pd.MultiIndex.from_product([[start_date], previous_data.index.get_level_values('ID')])
        previous_data.index = index
        ranged_data = data.loc[(data.index.get_level_values('DateTime') > start_date) &
                               (data.index.get_level_values('DateTime') <= end_date)]
        data = pd.concat([previous_data, ranged_data])

        date_list = self.calendar.select_dates(start_date=start_date, end_date=end_date)
        df = data.unstack().reindex(date_list).ffill()
        if dates:
            df = df.loc[dates, :]
        return df


class IndustryFactor(CompactFactor):
    """
    股票行业分类
    """

    def __init__(self, db_interface: DBInterface, provider: str, level: int) -> None:
        """
        :param db_interface: DB Interface
        :param provider: Industry classification data provider
        :param level: Level of industry classification
        """
        assert 0 < level <= constants.INDUSTRY_LEVEL[provider], f'{provider}行业没有{level}级'
        table_name = f'{provider}行业'
        super().__init__(db_interface, table_name)

        if level != constants.INDUSTRY_LEVEL[provider]:
            translation = utils.load_param('industry.json')
            new_translation = {}
            for key, value in translation[table_name].items():
                new_translation[key] = value[f'level_{level}']

            self.data = self.data.map(new_translation)

    @DateUtils.dtlize_input_dates
    def list_constitutes(self, date: DateUtils.DateType, industry: str) -> List[str]:
        """
        获取行业内的股票构成
        :param date: 查询日期
        :param industry: 行业名称
        :return:
        """
        date_data = self.get_data(dates=date)
        data = date_data.loc[:, (date_data == industry).values[0]]
        return data.columns.tolist()

    @cached_property
    def all_industries(self) -> List[str]:
        return self.data.dropna().unique().tolist()


class OnTheRecordFactor(NonFinancialFactor):
    """
        有记录的数据, 例如涨跌停, 一字板, 停牌等
    """

    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)
        self.factor_names = factor_name

    @DateUtils.dtlize_input_dates
    def get_data(self, date: DateUtils.DateType) -> List:
        """
        :param date: selected dates
        :return: list of IDs on the record
        """
        tmp = self.db_interface.read_table(self.table_name, dates=[date])
        return tmp.index.get_level_values('ID').tolist()


class CompactRecordFactor(NonFinancialFactor):
    def __init__(self, compact_factor: CompactFactor, factor_name: str):
        super().__init__(compact_factor.db_interface, compact_factor.table_name, compact_factor.factor_names)
        self.base_factor = compact_factor
        self.factor_names = factor_name

    @DateUtils.dtlize_input_dates
    def get_data(self, date: DateUtils.DateType) -> List:
        """
        :param date: selected dates
        :return: list of IDs on the record
        """
        tmp = self.base_factor.get_data(dates=[date]).stack()
        return tmp.loc[tmp].index.get_level_values('ID').tolist()


class ContinuousFactor(NonFinancialFactor):
    """
    有连续记录的数据
    """

    def __init__(self, db_interface: DBInterface, table_name: str, factor_names: Union[str, Sequence[str]]):
        super().__init__(db_interface, table_name, factor_names)

    @DateUtils.dtlize_input_dates
    def get_data(self, dates: Union[Sequence[dt.datetime], DateUtils.DateType] = None,
                 start_date: DateUtils.DateType = None, end_date: DateUtils.DateType = None,
                 ids: Sequence[str] = None, unstack: bool = True) -> Union[pd.Series, pd.DataFrame]:
        """
        :param start_date: start date
        :param end_date: end date
        :param dates: selected dates
        :param ids: query stocks
        :param unstack: if unstack data from long to wide
        :return: pandas.DataFrame with DateTime as index and stock as column
        """

        df = self.db_interface.read_table(self.table_name, columns=self.factor_names,
                                          start_date=start_date, end_date=end_date,
                                          dates=dates, ids=ids)
        if not isinstance(self.factor_names, str):
            df.columns = self.factor_names
            return df

        if isinstance(df.index, pd.MultiIndex) & unstack:
            df = df.unstack()
        return df


class AccountingFactor(Factor):
    """
    财报数据
    """

    fields = {}
    date_cache = {}

    def __init__(self, db_interface: DBInterface, factor_name: str):
        if len(self.fields) == 0:
            fields_data = utils.load_param('db_schema.json')
            for statement_type in constants.FINANCIAL_STATEMENTS_TYPE:
                statement = fields_data[statement_type]
                for key, _ in statement.items():
                    self.fields[key] = statement_type
        assert factor_name in self.fields.keys(), f'{factor_name} 非财报关键字'

        if len(self.date_cache) == 0:
            cache_loc = os.path.join(Path.home(), '.AShareData', constants.ACCOUNTING_DATE_CACHE_NAME)
            if os.path.exists(cache_loc):
                with zipfile.ZipFile(cache_loc, 'r', compression=zipfile.ZIP_DEFLATED) as zf:
                    with zf.open('cache.pkl') as f:
                        self.date_cache = pickle.load(f)
            else:
                raise RuntimeError('You must build cache first to use accounting factors!')

        table_name = self.fields[factor_name]
        super().__init__(db_interface, f'合并{table_name}', factor_name)
        self.report_month = None
        self.buffer_length = 365 * 2

    @DateUtils.dtlize_input_dates
    def get_data(self, dates: Sequence[dt.datetime] = None,
                 start_date: DateUtils.DateType = None, end_date: DateUtils.DateType = None,
                 ids: Sequence[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        :param start_date: start date
        :param end_date: end date
        :param dates: selected dates
        :param ids: query stocks
        :return: pandas.DataFrame with DateTime as index and stock as column
        """
        buffer = dt.timedelta(days=self.buffer_length)
        if dates:
            start_date = min(dates)
            end_date = max(end_date)
        buffer_start = start_date - buffer

        data = self.db_interface.read_table(self.table_name, self.factor_names,
                                            start_date=buffer_start, end_date=end_date, report_month=self.report_month,
                                            ids=ids)
        tickers = data.index.get_level_values('ID').unique().tolist()
        storage = []
        for ticker in tickers:
            start_index = self.date_cache['cache_date'].bisect_left((ticker, start_date)) - 1
            end_index = self.date_cache['cache_date'].bisect_left((ticker, end_date))
            for index in range(start_index, end_index):
                # index = start_index
                cache_index, date_cache = self.date_cache['cache_date'].peekitem(index)
                val = self.func(data, date_cache)
                pd_index = pd.MultiIndex.from_product([[cache_index[1]], [cache_index[0]]], names=['DateTime', 'ID'])
                storage.append(pd.Series(val, index=pd_index))

        buffer_start = data.index.get_level_values('DateTime').min()
        full_dates = self.calendar.select_dates(buffer_start, end_date)
        intermediate_dates = self.calendar.select_dates(start_date, end_date)

        df = pd.concat(storage).unstack().reindex(full_dates).ffill().reindex(intermediate_dates)
        if dates:
            df = df.reindex(dates)
        return df

    @staticmethod
    def func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        raise NotImplementedError()


class QuarterlyFactor(AccountingFactor):
    """
    季报数据
    """

    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)
        self.func = self.balance_sheet_func if self.table_name == '合并资产负债表' else self.cash_flow_or_profit_func

    @staticmethod
    def balance_sheet_func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        raise NotImplementedError()

    @staticmethod
    def cash_flow_or_profit_func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        raise NotImplementedError()

    @staticmethod
    def func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        pass


class LatestAccountingFactor(AccountingFactor):
    """最新财报数据"""
    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)

    @staticmethod
    def func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        return data.loc[date_cache.q0]


class LatestQuarterAccountingFactor(QuarterlyFactor):
    """最新财报季度数据"""
    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)

    @staticmethod
    def cash_flow_or_profit_func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        val = (data.loc[date_cache.q0] - data.loc[date_cache.q1]) / \
              (data.loc[date_cache.q4] - data.loc[date_cache.q5]) - 1
        return val

    @staticmethod
    def balance_sheet_func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        val = data.loc[date_cache.q0] / data.loc[date_cache.q1] - 1
        return val


class YearlyReportAccountingFactor(AccountingFactor):
    """
    年报数据
    """

    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)
        self.report_month = 12

    @staticmethod
    def func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        return data.loc[date_cache.y1]


class QOQAccountingFactor(QuarterlyFactor):
    """环比数据"""
    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)

    @staticmethod
    def cash_flow_or_profit_func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        val = (data.loc[date_cache.q0] - data.loc[date_cache.q1]) / \
              (data.loc[date_cache.q1] - data.loc[date_cache.q2]) - 1
        return val

    @staticmethod
    def balance_sheet_func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        val = data.loc[date_cache.q0] / data.loc[date_cache.q1] - 1
        return val


class YOYPeriodAccountingFactor(AccountingFactor):
    """同比数据"""
    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)

    @staticmethod
    def func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        val = data.loc[date_cache.q0] / data.loc[date_cache.q4] - 1
        return val


class YOYQuarterAccountingFactor(QuarterlyFactor):
    """季度同比数据"""
    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)

    @staticmethod
    def cash_flow_or_profit_func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        val = (data.loc[date_cache.q0] - data.loc[date_cache.q1]) / \
              (data.loc[date_cache.q4] - data.loc[date_cache.q5]) - 1
        return val

    @staticmethod
    def balance_sheet_func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        val = data.loc[date_cache.q0] / data.loc[date_cache.q4] - 1
        return val


class TTMAccountingFactor(AccountingFactor):
    """TTM数据"""
    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)

    @staticmethod
    def func(data: pd.DataFrame, date_cache: utils.DateCache) -> np.float:
        val = data.loc[date_cache.q0] - data.loc[date_cache.q4] + data.loc[date_cache.y1]
        return val
