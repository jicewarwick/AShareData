import datetime as dt
import numbers
from functools import cached_property, partial
from typing import List, Sequence, Union

import numpy as np
import pandas as pd

from . import algo, constants, DateUtils, utils
from .config import get_db_interface
from .DateUtils import SHSZTradingCalendar
from .DBInterface import DBInterface
from .utils import TickerSelector


class FactorBase(object):
    def __init__(self, factor_name: str = None):
        super().__init__()
        self._factor_name = factor_name
        self.name = factor_name
        self.calendar = SHSZTradingCalendar()

    def set_factor_name(self, name):
        self.name = name
        return self

    def get_data(self, *args, **kwargs) -> Union[pd.Series, List]:
        """获取数据"""
        s = self._get_data(*args, **kwargs)
        if self.name and isinstance(s, pd.Series):
            s.name = self.name
        return s

    def _get_data(self, *args, **kwargs) -> pd.Series:
        """获取数据"""
        raise NotImplementedError()

    def __and__(self, other):
        def sub_get_data(self, **kwargs):
            return self.f1.get_data(**kwargs) & self.f2.get_data(**kwargs)

        Foo = type('', (BinaryFactor,), {'_get_data': sub_get_data})
        return Foo(self, other)

    def __invert__(self):
        def sub_get_data(self, **kwargs):
            return ~self.f.get_data(**kwargs)

        Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
        return Foo(self)

    def __add__(self, other):
        if isinstance(other, (numbers.Number, np.number)):
            def sub_get_data(self, **kwargs):
                return self.f.get_data(**kwargs) + other

            Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
            return Foo(self)
        else:
            def sub_get_data(self, **kwargs):
                return self.f1.get_data(**kwargs) + self.f2.get_data(**kwargs)

            Foo = type('', (BinaryFactor,), {'_get_data': sub_get_data})
            return Foo(self, other)

    def __sub__(self, other):
        if isinstance(other, (numbers.Number, np.number)):
            def sub_get_data(self, **kwargs):
                return self.f.get_data(**kwargs) - other

            Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
            return Foo(self)
        else:
            def sub_get_data(self, **kwargs):
                return self.f1.get_data(**kwargs) - self.f2.get_data(**kwargs)

            Foo = type('', (BinaryFactor,), {'get_data': sub_get_data})
            return Foo(self, other)

    def __mul__(self, other):
        if isinstance(other, (numbers.Number, np.number)):
            def sub_get_data(self, **kwargs):
                return self.f.get_data(**kwargs) * other

            Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
            return Foo(self)
        else:
            def sub_get_data(self, **kwargs):
                return self.f1.get_data(**kwargs) * self.f2.get_data(**kwargs)

            Foo = type('', (BinaryFactor,), {'_get_data': sub_get_data})
            return Foo(self, other)

    def __truediv__(self, other):
        if isinstance(other, (numbers.Number, np.number)):
            def sub_get_data(self, **kwargs):
                return self.f.get_data(**kwargs) / other

            Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
            return Foo(self)
        else:
            def sub_get_data(self, **kwargs):
                return self.f1.get_data(**kwargs) / self.f2.get_data(**kwargs)

            Foo = type('', (BinaryFactor,), {'_get_data': sub_get_data})
            return Foo(self, other)

    def __gt__(self, other):
        if isinstance(other, (numbers.Number, np.number)):
            def sub_get_data(self, **kwargs):
                return self.f.get_data(**kwargs) > other

            Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
            return Foo(self)
        else:
            def sub_get_data(self, **kwargs):
                return self.f1.get_data(**kwargs) > self.f2.get_data(**kwargs)

            Foo = type('', (BinaryFactor,), {'_get_data': sub_get_data})
            return Foo(self, other)

    def __lt__(self, other):
        if isinstance(other, (numbers.Number, np.number)):
            def sub_get_data(self, **kwargs):
                return self.f.get_data(**kwargs) < other

            Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
            return Foo(self)
        else:
            def sub_get_data(self, **kwargs):
                return self.f1.get_data(**kwargs) < self.f2.get_data(**kwargs)

            Foo = type('', (BinaryFactor,), {'_get_data': sub_get_data})
            return Foo(self, other)

    def __ge__(self, other):
        if isinstance(other, (numbers.Number, np.number)):
            def sub_get_data(self, **kwargs):
                return self.f.get_data(**kwargs) >= other

            Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
            return Foo(self)
        else:
            def sub_get_data(self, **kwargs):
                return self.f1.get_data(**kwargs) >= self.f2.get_data(**kwargs)

            Foo = type('', (BinaryFactor,), {'_get_data': sub_get_data})
            return Foo(self, other)

    def __le__(self, other):
        if isinstance(other, (numbers.Number, np.number)):
            def sub_get_data(self, **kwargs):
                return self.f.get_data(**kwargs) <= other

            Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
            return Foo(self)
        else:
            def sub_get_data(self, **kwargs):
                return self.f1.get_data(**kwargs) <= self.f2.get_data(**kwargs)

            Foo = type('', (BinaryFactor,), {'_get_data': sub_get_data})
            return Foo(self, other)

    def __eq__(self, other):
        if isinstance(other, (numbers.Number, np.number)):
            def sub_get_data(self, **kwargs):
                return self.f.get_data(**kwargs) == other

            Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
            return Foo(self)
        else:
            def sub_get_data(self, **kwargs):
                return self.f1.get_data(**kwargs) == self.f2.get_data(**kwargs)

            Foo = type('', (BinaryFactor,), {'_get_data': sub_get_data})
            return Foo(self, other)

    def __ne__(self, other):
        if isinstance(other, (numbers.Number, np.number)):
            def sub_get_data(self, **kwargs):
                return self.f.get_data(**kwargs) != other

            Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
            return Foo(self)
        else:
            def sub_get_data(self, **kwargs):
                return self.f1.get_data(**kwargs) != self.f2.get_data(**kwargs)

            Foo = type('', (BinaryFactor,), {'_get_data': sub_get_data})
            return Foo(self, other)

    def __abs__(self):
        def sub_get_data(self, **kwargs):
            return self.f.get_data(**kwargs).abs()

        Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
        return Foo(self)

    def __neg__(self):
        def sub_get_data(self, **kwargs):
            return -self.f.get_data(**kwargs)

        Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
        return Foo(self)

    def max(self):
        """analogue to max for each ``ID``"""

        def sub_get_data(self, **kwargs):
            return self.f.get_data(**kwargs).unstack().max()

        Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
        return Foo(self)

    def log(self):
        """analogue to numpy.log"""

        def sub_get_data(self, **kwargs):
            return np.log(self.f.get_data(**kwargs))

        Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
        return Foo(self)

    def pct_change(self):
        """analogue to pd.pct_change for each ``ID``"""

        def sub_get_data(self, **kwargs):
            if 'start_date' in kwargs:
                kwargs['start_date'] = self.calendar.offset(kwargs['start_date'], -1)
            return self.f.get_data(**kwargs).unstack().pct_change().stack().dropna()

        Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
        return Foo(self)

    def diff(self):
        """analogue to pd.diff for each ``ID``"""

        def sub_get_data(self, **kwargs):
            if 'start_date' in kwargs:
                kwargs['start_date'] = self.calendar.offset(kwargs['start_date'], -1)
            return self.f.get_data(**kwargs).unstack().diff().stack().dropna()

        Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
        return Foo(self)

    def shift(self, n: int):
        """analogue to pd.shift(n) for each ``ID``"""

        def sub_get_data(self, **kwargs):
            if 'start_date' in kwargs and n > 0:
                kwargs['start_date'] = self.calendar.offset(kwargs['start_date'], n)
            elif 'end_date' in kwargs and n < 0:
                kwargs['end_date'] = self.calendar.offset(kwargs['end_date'], n)
            return self.f.get_data(**kwargs).unstack().shift(n).stack().dropna()

        Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
        return Foo(self)

    def diff_shift(self, n: int):
        """analogue to pd.diff().shift(n) for each ``ID``"""

        def sub_get_data(self, **kwargs):
            if 'start_date' in kwargs and n > 0:
                kwargs['start_date'] = self.calendar.offset(kwargs['start_date'], n)
            elif 'end_date' in kwargs and n < 0:
                kwargs['end_date'] = self.calendar.offset(kwargs['end_date'], n)
            return self.f.get_data(**kwargs).unstack().diff().shift(n).stack().dropna()

        Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
        return Foo(self)

    def pct_change_shift(self, n: int):
        """analogue to pd.pct_change().shift(n) for each ``ID``"""

        def sub_get_data(self, **kwargs):
            if 'start_date' in kwargs and n > 0:
                kwargs['start_date'] = self.calendar.offset(kwargs['start_date'], n)
            elif 'end_date' in kwargs and n < 0:
                kwargs['end_date'] = self.calendar.offset(kwargs['end_date'], n)
            return self.f.get_data(**kwargs).unstack().pct_change().shift(n).stack().dropna()

        Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
        return Foo(self)

    def weight(self):
        def sub_get_data(self, **kwargs):
            data = self.f.get_data(**kwargs)
            return data / data.groupby('DateTime').sum()

        Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
        return Foo(self)

    def corr(self, other):
        pass

    def mean(self, along: str = 'DateTime'):
        assert along in ['DateTime', 'ID'], 'along can only be DateTime or ID'

        def sub_get_data(self, **kwargs):
            data = self.f.get_data(**kwargs)
            return data.groupby(along).mean()

        Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
        return Foo(self)

    def sum(self, along: str = 'DateTime'):
        assert along in ['DateTime', 'ID'], 'along can only be DateTime or ID'

        def sub_get_data(self, **kwargs):
            data = self.f.get_data(**kwargs)
            return data.groupby(along).sum()

        Foo = type('', (UnaryFactor,), {'_get_data': sub_get_data})
        return Foo(self)

    def bind_params(self, ids: Union[str, Sequence[str]] = None):
        if ids:
            self._get_data = partial(self._get_data, ids=ids)
        return self


class UnaryFactor(FactorBase):
    def __init__(self, f: FactorBase):
        super().__init__()
        self.f = f

    def _get_data(self, *args, **kwargs):
        """获取数据"""
        raise NotImplementedError()


class BinaryFactor(FactorBase):
    def __init__(self, f1: FactorBase, f2: FactorBase):
        super().__init__()
        self.f1 = f1
        self.f2 = f2

    def _get_data(self, *args, **kwargs):
        """获取数据"""
        raise NotImplementedError()


class Factor(FactorBase):
    """
    Factor base class
    """

    def __init__(self, table_name: str, factor_name: str, db_interface: DBInterface = None):
        super().__init__(factor_name)
        self.table_name = table_name
        self.db_interface = db_interface if db_interface else get_db_interface()

    def _get_data(self, *args, **kwargs):
        """获取数据"""
        raise NotImplementedError()

    # helper functions
    def _check_args(self, table_name: str, factor_name: str):
        table_name = table_name.lower()
        assert self.db_interface.exist_table(table_name), f'数据库中不存在表 {table_name}'

        if factor_name:
            columns = self.db_interface.get_columns_names(table_name)
            assert factor_name in columns, f'表 {table_name} 中不存在 {factor_name} 列'


class IndexConstitute(Factor):
    """指数成分股权重"""

    def __init__(self, db_interface: DBInterface = None):
        super().__init__('指数成分股权重', '', db_interface)

    def _get_data(self, index_ticker: str, date: DateUtils.DateType):
        date_str = DateUtils.date_type2str(date, '-')
        stm = f'DateTime = (SELECT MAX(DateTime) FROM `{self.table_name}` WHERE DateTime <= "{date_str}")'
        ret = self.db_interface.read_table(self.table_name, ids=index_ticker, text_statement=stm)
        ret.index = pd.MultiIndex.from_product([[date], ret.index.get_level_values('ID')])
        ret.index.names = ['DateTime', 'ID']
        return ret


class NonFinancialFactor(Factor):
    """
    非财报数据
    """

    def __init__(self, table_name: str, factor_name: str = None, db_interface: DBInterface = None):
        super().__init__(table_name, factor_name, db_interface)
        self._check_args(table_name, factor_name)
        assert not any([it in table_name for it in constants.FINANCIAL_STATEMENTS_TYPE]), \
            f'{table_name} 为财报数据, 请使用 FinancialFactor 类!'

    def _get_data(self, *args, **kwargs):
        raise NotImplementedError()


class CompactFactor(NonFinancialFactor):
    """
    数字变动很不平常的特性, 列如复权因子, 行业, 股本 等.

    对应的数据库表格为: {'DateTime', 'ID', 'FactorName'} 该类可以缓存以提升效率
    """

    def __init__(self, table_name: str, db_interface: DBInterface = None):
        super().__init__(table_name, table_name, db_interface)
        self.data = self.db_interface.read_table(table_name)

    def _get_data(self, dates: Union[Sequence[dt.datetime], DateUtils.DateType] = None,
                  start_date: DateUtils.DateType = None, end_date: DateUtils.DateType = None,
                  ids: Union[Sequence[str], str] = None, ticker_selector: TickerSelector = None) -> pd.Series:
        """
        :param start_date: start date
        :param end_date: end date
        :param dates: selected dates
        :param ids: query stocks
        :param ticker_selector: TickerSelector that specifies criteria
        :return: pandas.Series with DateTime as index and stock as column
        """
        if isinstance(dates, dt.datetime):
            dates = [dates]
        data = self.data.copy()
        if ids:
            if isinstance(ids, str):
                ids = [ids]
            data = data.loc[(slice(None), ids)]
        if dates:
            end_date = max(dates)
            start_date = min(dates)
        if end_date is None:
            end_date = dt.datetime.today()
        if start_date is None:
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
        ret = df.stack()
        ret.index.names = ['DateTime', 'ID']
        if ticker_selector:
            index = ticker_selector.generate_index(dates=dates)
            ret = ret.reindex(index)
        return ret


class IndustryFactor(CompactFactor):
    """
    股票行业分类
    """

    def __init__(self, provider: str, level: int, db_interface: DBInterface = None) -> None:
        """
        :param db_interface: DB Interface
        :param provider: Industry classification data provider
        :param level: Level of industry classification
        """
        assert 0 < level <= constants.INDUSTRY_LEVEL[provider], f'{provider}行业没有{level}级'
        table_name = f'{provider}行业'
        super().__init__(table_name, db_interface)
        self.name = f'{provider}{level}级行业'

        if level != constants.INDUSTRY_LEVEL[provider]:
            translation = utils.load_param('industry.json')
            new_translation = {}
            for key, value in translation[table_name].items():
                new_translation[key] = value[f'level_{level}']

            self.data = self.data.map(new_translation)

    def list_constitutes(self, date: DateUtils.DateType, industry: str) -> List[str]:
        """
        获取行业内的股票构成

        :param date: 查询日期
        :param industry: 行业名称
        :return: 行业构成
        """
        date_data = self.get_data(dates=date)
        data = date_data.loc[date_data == industry]
        return data.index.get_level_values('ID').tolist()

    @cached_property
    def all_industries(self) -> List[str]:
        """所有行业列表"""
        return self.data.dropna().unique().tolist()


class OnTheRecordFactor(NonFinancialFactor):
    """
        有记录的数据, 例如涨跌停, 一字板, 停牌等
    """

    def __init__(self, factor_name: str, db_interface: DBInterface = None):
        super().__init__(factor_name, db_interface=db_interface)
        self.factor_name = factor_name

    def _get_data(self, date: DateUtils.DateType, **kwargs) -> List:
        """
        :param date: selected dates
        :return: list of IDs on the record
        """
        tmp = self.db_interface.read_table(self.table_name, dates=date)
        return tmp.index.get_level_values('ID').tolist()

    def get_counts(self, start_date: DateUtils.DateType, end_date: DateUtils.DateType,
                   ids: Sequence[str] = None) -> pd.Series:
        """返回 ``start_date`` 和 ``end_date`` (含)之间的记录次数"""
        tmp = self.db_interface.read_table(self.table_name, start_date=start_date, end_date=end_date, ids=ids)
        tmp = tmp.reset_index().groupby('ID').count().reindex(ids).fillna(0)
        res = tmp.iloc[:, 0]
        res.index = pd.MultiIndex.from_product([[end_date], res.index], names=('DateTime', 'ID'))
        res.name = f'{self.table_name}天数'
        return res


class CompactRecordFactor(NonFinancialFactor):
    def __init__(self, compact_factor: CompactFactor, factor_name: str):
        super().__init__(compact_factor.table_name, compact_factor._factor_name, compact_factor.db_interface)
        self.base_factor = compact_factor
        self.factor_name = factor_name

    def _get_data(self, date: DateUtils.DateType, **kwargs) -> List:
        """
        :param date: selected dates
        :return: list of IDs on the record
        """
        tmp = self.base_factor.get_data(dates=date)
        return tmp.loc[tmp].index.get_level_values('ID').tolist()


class ContinuousFactor(NonFinancialFactor):
    """
    有连续记录的数据
    """

    def __init__(self, table_name: str, factor_name: str, db_interface: DBInterface = None):
        super().__init__(table_name, factor_name, db_interface)

    def _get_data(self, dates: Union[Sequence[dt.datetime], DateUtils.DateType] = None,
                  start_date: DateUtils.DateType = None, end_date: DateUtils.DateType = None,
                  ids: Sequence[str] = None, ticker_selector: TickerSelector = None, **kwargs) \
            -> Union[pd.Series, pd.DataFrame]:
        """
        :param start_date: start date
        :param end_date: end date
        :param dates: selected dates
        :param ids: query stocks
        :param ticker_selector: TickerSelector that specifies criteria
        :return: pandas.DataFrame with DateTime as index and stock as column
        """

        df = self.db_interface.read_table(self.table_name, columns=self._factor_name,
                                          start_date=start_date, end_date=end_date, dates=dates, ids=ids)

        if ticker_selector:
            index = ticker_selector.generate_index(dates=dates)
            df = df.reindex(index)

        return df


class InterestRateFactor(ContinuousFactor):
    def __init__(self, table_name: str, factor_name: str, db_interface: DBInterface = None):
        super().__init__(table_name, factor_name, db_interface)

    def _get_data(self, *args, **kwargs) -> pd.Series:
        data = super()._get_data(*args, **kwargs)
        dates = data.index.get_level_values('DateTime').tolist()
        after_date = pd.Timestamp(self.calendar.offset(data.index.get_level_values('DateTime')[-1], 1))
        days = [(it[1] - it[0]).days for it in zip(dates, dates[1:] + [after_date])]
        data = data / 365 * days / 100
        return data


# TODO
class PriceFactor(FactorBase):
    def __init__(self, factor: FactorBase, db_interface: DBInterface = None):
        super().__init__(factor.name)
        self.factor = factor
        self.db_interface = db_interface if db_interface else get_db_interface()

    def _get_data(self, *args, **kwargs) -> pd.Series:
        return self.factor.get_data(*args, **kwargs)

    def get_return_data(self, *args, **kwargs):
        data = self.factor._get_data(*args, **kwargs)


class AccountingFactor(Factor):
    """
    财报数据
    """

    fields = {}

    def __init__(self, factor_name: str, db_interface: DBInterface = None):
        if len(self.fields) == 0:
            fields_data = utils.load_param('db_schema.json')
            for statement_type in constants.FINANCIAL_STATEMENTS_TYPE:
                statement = fields_data[statement_type]
                for key, _ in statement.items():
                    self.fields[key] = statement_type
        assert factor_name in self.fields.keys(), f'{factor_name} 非财报关键字'

        table_name = self.fields[factor_name]
        super().__init__(f'合并{table_name}', factor_name, db_interface)
        self.report_month = None
        self.buffer_length = 365 * 2
        self.offset_strs = None

    def _get_data(self, dates: Union[dt.datetime, Sequence[dt.datetime]] = None,
                  start_date: DateUtils.DateType = None, end_date: DateUtils.DateType = None,
                  ids: Sequence[str] = None, ticker_selector: TickerSelector = None) -> Union[pd.Series, pd.DataFrame]:
        """
        :param start_date: start date
        :param end_date: end date
        :param dates: selected dates
        :param ids: query stocks
        :param ticker_selector: TickerSelector that specifies criteria
        :return: pandas.DataFrame with DateTime as index and stock as column
        """
        buffer = dt.timedelta(days=self.buffer_length)
        if dates:
            if isinstance(dates, dt.datetime):
                dates = [dates]
            db_start_date, db_end_date = min(dates), max(dates)
        else:
            db_start_date, db_end_date = start_date, end_date
            dates = self.calendar.select_dates(start_date, end_date)
        buffer_start = db_start_date - buffer

        db_columns = [self._factor_name]
        if self.offset_strs:
            db_columns.extend(self.offset_strs)
        data = self.db_interface.read_table(self.table_name, columns=db_columns,
                                            start_date=buffer_start, end_date=end_date, report_month=self.report_month,
                                            ids=ids)
        if isinstance(data, pd.Series):
            data = data.to_frame()
        tickers = data.index.get_level_values('ID').unique().tolist()
        storage = []
        for ticker in tickers:
            ticker_data = data.loc[(slice(None), ticker, slice(None)), :]
            related_dates = algo.get_less_or_equal_of_a_in_b(a=dates,
                                                             b=ticker_data.index.get_level_values('DateTime').tolist())
            relevant_rec = ticker_data.loc[ticker_data.index.get_level_values('DateTime').isin(related_dates.values())]
            relevant_rec = relevant_rec.groupby('DateTime').tail(1)

            pre_data = self.gather_data(ticker_data, relevant_rec, self.offset_strs)
            calc_data = self.func(pre_data)
            calc_data.index = calc_data.index.get_level_values('DateTime')

            res_index = pd.MultiIndex.from_product([list(related_dates.keys()), [ticker]], names=('DateTime', 'ID'))
            val = calc_data.loc[related_dates.values()].values
            content = pd.DataFrame(val, index=res_index)
            storage.append(content)

        ret = pd.concat(storage)
        if ret.shape[1] == 1:
            ret = ret.iloc[:, 0]
            ret.name = self._factor_name

        if ticker_selector:
            index = ticker_selector.generate_index(dates=dates)
            ret = ret.reindex(index)
        return ret

    @staticmethod
    def func(data: pd.DataFrame) -> np.float:
        raise NotImplementedError()

    def gather_data(self, ticker_data: pd.DataFrame, relevant_rec: pd.DataFrame,
                    offset_strs: List[str]) -> pd.DataFrame:
        if not offset_strs:
            relevant_rec.columns = ['q0']
            return relevant_rec
        storage = [self.loc_pre_data(ticker_data, relevant_rec, offset_str).iloc[:, 0].values for offset_str in
                   offset_strs]
        storage.append(relevant_rec.iloc[:, 0].values)
        col_names = offset_strs + ['q0']
        return pd.DataFrame(np.stack(storage, axis=1), index=relevant_rec.index, columns=col_names)

    @staticmethod
    def loc_pre_data(ticker_data: pd.DataFrame, relevant_rec: pd.DataFrame, offset_str: str) -> pd.DataFrame:
        pre_date = [DateUtils.ReportingDate.offset(it, offset_str) for it in relevant_rec.index.get_level_values('报告期')]
        ticker = ticker_data.index.get_level_values('ID')[0]
        pre_index = pd.MultiIndex.from_arrays([relevant_rec[offset_str], [ticker] * relevant_rec.shape[0], pre_date])
        pre_data = ticker_data.reindex(pre_index)
        return pre_data


class QuarterlyFactor(AccountingFactor):
    """
    季报数据
    """

    def __init__(self, factor_name: str, db_interface: DBInterface = None):
        super().__init__(factor_name, db_interface)
        self.func = self.balance_sheet_func if self.table_name == '合并资产负债表' else self.cash_flow_or_profit_func
        self.name = f'季度{self._factor_name}'

    @staticmethod
    def balance_sheet_func(data: pd.DataFrame) -> np.float:
        raise NotImplementedError()

    @staticmethod
    def cash_flow_or_profit_func(data: pd.DataFrame) -> np.float:
        raise NotImplementedError()

    @staticmethod
    def func(data: pd.DataFrame) -> np.float:
        pass


class LatestAccountingFactor(AccountingFactor):
    """最新财报数据"""

    def __init__(self, factor_name: str, db_interface: DBInterface = None):
        super().__init__(factor_name, db_interface)
        self.name = f'最新{self._factor_name}'

    @staticmethod
    def func(data: pd.DataFrame) -> np.float:
        return data.q0


class LatestQuarterAccountingFactor(QuarterlyFactor):
    """最新财报季度数据"""

    def __init__(self, factor_name: str, db_interface: DBInterface = None):
        super().__init__(factor_name, db_interface)
        self.offset_strs = ['q1'] if self.table_name == '合并资产负债表' else ['q1', 'q5']
        self.name = f'最新季度{self._factor_name}'

    @staticmethod
    def cash_flow_or_profit_func(data: pd.DataFrame) -> np.float:
        val = (data.q0 - data.q1) / (data.q4 - data.q5) - 1
        return val

    @staticmethod
    def balance_sheet_func(data: pd.DataFrame) -> np.float:
        val = data.q0 / data.q1 - 1
        return val


class YearlyReportAccountingFactor(AccountingFactor):
    """
    年报数据
    """

    def __init__(self, factor_name: str, db_interface: DBInterface = None):
        super().__init__(factor_name, db_interface)
        self.report_month = 12
        self.offset_strs = ['y1']
        self.name = f'最新年报{self._factor_name}'

    @staticmethod
    def func(data: pd.DataFrame) -> np.float:
        return data.y1


class QOQAccountingFactor(QuarterlyFactor):
    """环比数据"""

    def __init__(self, factor_name: str, db_interface: DBInterface = None):
        super().__init__(factor_name, db_interface)
        self.offset_strs = ['q1'] if self.table_name == '合并资产负债表' else ['q1', 'q2']
        self.name = f'{self._factor_name}环比'

    @staticmethod
    def cash_flow_or_profit_func(data: pd.DataFrame) -> np.float:
        val = (data.q0 - data.q1) / (data.q1 - data.q2) - 1
        return val

    @staticmethod
    def balance_sheet_func(data: pd.DataFrame) -> np.float:
        val = data.q0 / data.q1 - 1
        return val


class YOYPeriodAccountingFactor(AccountingFactor):
    """同比数据"""

    def __init__(self, factor_name: str, db_interface: DBInterface = None):
        super().__init__(factor_name, db_interface)
        self.offset_strs = ['q1', 'q4']
        self.name = f'{self._factor_name}同比'

    @staticmethod
    def func(data: pd.DataFrame) -> np.float:
        val = data.q0 / data.q4 - 1
        return val


class YOYQuarterAccountingFactor(QuarterlyFactor):
    """季度同比数据"""

    def __init__(self, factor_name: str, db_interface: DBInterface = None):
        super().__init__(factor_name, db_interface)
        self.offset_strs = ['q4'] if self.table_name == '合并资产负债表' else ['q1', 'q4', 'q5']
        self.name = f'年度{self._factor_name}增长率'

    @staticmethod
    def cash_flow_or_profit_func(data: pd.DataFrame) -> np.float:
        val = (data.q0 - data.q1) / (data.q4 - data.q5) - 1
        return val

    @staticmethod
    def balance_sheet_func(data: pd.DataFrame) -> np.float:
        val = data.q0 / data.q4 - 1
        return val


class TTMAccountingFactor(AccountingFactor):
    """TTM数据"""

    def __init__(self, factor_name: str, db_interface: DBInterface = None):
        super().__init__(factor_name, db_interface)
        self.offset_strs = ['q4', 'y1']
        self.name = f'{self._factor_name}TTM'

    @staticmethod
    def func(data: pd.DataFrame) -> np.float:
        val = data.q0 - data.q4 + data.y1
        return val


class CachedFactor(FactorBase):
    def __init__(self, data: Union[pd.Series, pd.DataFrame], factor_name: str = None):
        super().__init__(factor_name)
        self.data = data

    def _get_data(self, *args, **kwargs):
        return self.data


class BetaFactor(FactorBase):
    def __init__(self, market_ret: FactorBase = None, rf_rate: InterestRateFactor = None,
                 db_interface: DBInterface = None):
        """ Stock Beta

        :param market_ret: 市场收益
        :param rf_rate: 无风险收益
        :param db_interface: Database Interface
        """
        super().__init__('Beta')
        if db_interface is None:
            db_interface = get_db_interface()
        stock_ret = ContinuousFactor('股票日行情', '收盘价', db_interface) * CompactFactor('复权因子', db_interface)
        self.stock_ret = stock_ret.pct_change()
        if market_ret is None:
            market_ret = ContinuousFactor('指数日行情', '收盘点位', db_interface).pct_change()
            market_ret.bind_params(ids='000300.SH')
        self.market_ret = market_ret
        if rf_rate is None:
            self.rf_rate = InterestRateFactor('shibor利率数据', '3个月', db_interface)

    def _get_data(self, dates: Sequence[dt.datetime],
                  ids: Union[str, Sequence[str]] = None, ticker_selector: TickerSelector = None,
                  look_back_period: int = 60, min_trading_days: int = 40) -> pd.Series:
        storage = []
        for date in dates:
            if ticker_selector:
                ids = ticker_selector.ticker(date)
            start_date = self.calendar.offset(date, -look_back_period - 1)
            end_date = self.calendar.offset(date, -1)
            stock_data = self.stock_ret.get_data(ids=ids, start_date=start_date, end_date=end_date).reset_index()
            market_data = self.market_ret.get_data(start_date=start_date, end_date=end_date).droplevel(
                'ID').reset_index()
            rf_data = self.rf_rate.get_data(start_date=start_date, end_date=end_date).reset_index()
            combined_data = pd.merge(stock_data, market_data, on='DateTime')
            combined_data = pd.merge(combined_data, rf_data, on='DateTime')
            combined_data.iloc[:, 2] = combined_data.iloc[:, 2] - combined_data.iloc[:, -1]
            combined_data.iloc[:, 3] = combined_data.iloc[:, 3] - combined_data.iloc[:, -1]

            for ID, group in combined_data.groupby('ID'):
                if group.shape[0] < min_trading_days:
                    beta = np.nan
                else:
                    cov_matrix = np.cov(group.iloc[:, 2], group.iloc[:, 3])
                    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                data = pd.Series(beta, index=pd.MultiIndex.from_tuples([(date, ID)], names=('DateTime', 'ID')))
                storage.append(data)

        ret = pd.concat(storage).sort_index()
        return ret
