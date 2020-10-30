import datetime as dt
import os
import pickle
from functools import cached_property
from pathlib import Path
from typing import Callable, List, Sequence, Union

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

    @DateUtils.format_input_dates
    def list_constitutes(self, date: DateUtils.DateType, industry: str) -> List[str]:
        date_data = self.get_data(dates=date)
        data = date_data.loc[:, (date_data == industry).values[0]]
        return data.columns.tolist()

    @cached_property
    def all_industries(self) -> List[str]:
        return self.data.dropna().unique().tolist()


class OnTheRecordFactor(NonFinancialFactor):
    """
        有记录的因子, 例如涨跌停, 一字板, 停牌等
    """

    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)
        self.factor_names = factor_name

    @DateUtils.format_input_dates
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

    @DateUtils.format_input_dates
    def get_data(self, date: DateUtils.DateType) -> List:
        """
        :param date: selected dates
        :return: list of IDs on the record
        """
        tmp = self.base_factor.get_data(dates=[date]).stack()
        return tmp.loc[tmp].index.get_level_values('ID').tolist()


class ContinuousFactor(NonFinancialFactor):
    """
    Continuous Factors
    """

    def __init__(self, db_interface: DBInterface, table_name: str, factor_names: Union[str, Sequence[str]]):
        super().__init__(db_interface, table_name, factor_names)

    @DateUtils.format_input_dates
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
                with open(cache_loc, 'rb') as f:
                    self.date_cache = pickle.load(f)
            else:
                raise RuntimeError('Must build cache first to use accounting factors')

        table_name = self.fields[factor_name]
        super().__init__(db_interface, table_name, factor_name)

    def get_data(self, *args, **kwargs):
        raise NotImplementedError()

    def _flatten_data(self, data: pd.Series, agg_func: Callable, start_date: dt.datetime, end_date: dt.datetime,
                      dates: Sequence[dt.datetime]) -> pd.DataFrame:
        storage = []
        tickers = data.index.get_level_values('ID').unique().tolist()
        for ticker in tickers:
            id_data = data.loc[data.index.get_level_values('ID') == ticker, :]
            data_dates = id_data.index.get_level_values('DateTime').to_pydatetime().tolist()
            dates = sorted(list(set(data_dates)))
            for date in dates:
                date_id_data = id_data.loc[id_data.index.get_level_values('DateTime') <= date, :]
                each_date_data = date_id_data.groupby('报告期').tail(1)
                output_data = each_date_data.apply({data.name: agg_func})
                output_data.index = pd.MultiIndex.from_tuples([(date, ticker)], names=['DateTime', 'ID'])
                storage.append(output_data)

        df = pd.concat(storage)

        buffer_start = data.index.get_level_values('DateTime').min()
        full_dates = self.calendar.select_dates(buffer_start, end_date)
        intermediate_dates = self.calendar.select_dates(start_date, end_date)

        df = df.unstack().reindex(full_dates).ffill().reindex(intermediate_dates)
        if dates:
            df = df.reset_index(dates)
        return df


class YearlyReportAccountingFactor(AccountingFactor):
    """
    年报数据
    """

    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)
        self.table_name = f'合并{self.table_name}'
        self._check_args(self.table_name, self.factor_names)

    @DateUtils.format_input_dates
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
        buffer = dt.timedelta(days=365 * 2)
        if dates:
            start_date = min(dates)
            end_date = max(end_date)
        buffer_start = start_date - buffer

        data = self.db_interface.read_table(self.table_name, self.factor_names,
                                            start_date=buffer_start, end_date=end_date, report_month=12,
                                            ids=ids)

        storage = []
        tickers = data.index.get_level_values('ID').unique().tolist()
        for ticker in tickers:
            start_index = self.date_cache['YOY'].bisect_left((ticker, start_date)) - 1
            end_index = self.date_cache['YOY'].bisect_left((ticker, end_date))
            for index in range(start_index, end_index):
                val = self.date_cache['YOY'].peekitem(index)[1]
                val[0]

        # return df


class TTMAccountingFactor(AccountingFactor):
    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)
        self.table_name = f'合并单季度{self.table_name}'
        self._check_args(self.table_name, self.factor_names)

    @DateUtils.format_input_dates
    def get_data(self, dates: Sequence[dt.datetime] = None,
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
        buffer = dt.timedelta(days=365 * 2)
        if dates:
            start_date = min(dates)
            end_date = max(end_date)
        buffer_start = start_date - buffer

        data = self.db_interface.read_table(self.table_name, self.factor_names,
                                            start_date=buffer_start, end_date=end_date,
                                            ids=ids).dropna()
        df = self._flatten_data(data, lambda x: x.tail(4).sum(), start_date=start_date, end_date=end_date, dates=dates)

        return df


class LatestAccountingFactor(AccountingFactor):
    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)
        self.table_name = f'合并{self.table_name}'
        self._check_args(self.table_name, self.factor_names)

    @DateUtils.format_input_dates
    def get_data(self, dates: Sequence[dt.datetime] = None,
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
        buffer = dt.timedelta(days=365 * 2)
        if dates:
            start_date = min(dates)
            end_date = max(end_date)
        buffer_start = start_date - buffer

        data = self.db_interface.read_table(self.table_name, self.factor_names,
                                            start_date=buffer_start, end_date=end_date,
                                            ids=ids).dropna()

        df = self._flatten_data(data, lambda x: x.tail(1), start_date=start_date, end_date=end_date, dates=dates)
        return df


class YOYTTMAccountingFactor(AccountingFactor):
    def __init__(self, db_interface: DBInterface, factor_name: str):
        super().__init__(db_interface, factor_name)
        self.table_name = f'合并单季度{self.table_name}'
        self._check_args(self.table_name, self.factor_names)

    @DateUtils.format_input_dates
    def get_data(self, dates: Sequence[dt.datetime] = None,
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
        buffer = dt.timedelta(days=365 * 2)
        if dates:
            start_date = min(dates)
            end_date = max(end_date)
        buffer_start = start_date - buffer

        data = self.db_interface.read_table(self.table_name, self.factor_names,
                                            start_date=buffer_start, end_date=end_date,
                                            ids=ids).dropna()
        tickers = data.index.get_level_values('ID').unique().tolist()
        storage = []
        for ticker in tickers:
            start_index = self.date_cache['YOY'].bisect_left((ticker, start_date)) - 1
            end_index = self.date_cache['YOY'].bisect_left((ticker, end_date))
            pre_year_storage = []
            current_year_storage = []
            for index in range(start_index, end_index):
                val = self.date_cache['YOY'].peekitem(index)[1]
                pre_year_storage.append(data.loc[val[0]])
                current_year_storage.append(data.loc[val[1]])
            pre_year_data = pd.concat(pre_year_storage)
            current_year_data = pd.concat(current_year_storage)
            index = current_year_data.index.droplevel(2)
            tmp = pd.Series(current_year_data.values / pre_year_data.values, index=index)
            storage.append(tmp)

        buffer_start = data.index.get_level_values('DateTime').min()
        full_dates = self.calendar.select_dates(buffer_start, end_date)
        intermediate_dates = self.calendar.select_dates(start_date, end_date)

        df = pd.concat(storage).unstack().reindex(full_dates).ffill().reindex(intermediate_dates)
        if dates:
            df = df.reset_index(dates)
        return df
