from typing import Sequence, List

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine.url import URL

from utils import DateType, select_dates


class Factor(object):
    def __init__(self, df: pd.DataFrame, name: str = None) -> None:
        self._name = name
        self._data = df

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def __mul__(self, other):
        return Factor(self.data * other.data)

    # def fill_na(self):


class FinancialFactor(Factor):
    def __init__(self, df: pd.DataFrame, name: str = None) -> None:
        super().__init__(df, name)


class SQLDBReader(object):
    def __init__(self, ip: str, port: int, username: str, password: str, db_name: str,
                 driver: str = 'mysql+pymysql') -> None:
        """
        SQL Database Reader

        :param ip: server ip
        :param port: server port
        :param username: username
        :param password: password
        :param db_name: database name
        :param driver: MySQL driver
        """
        url = URL(drivername=driver, username=username, password=password, host=ip, port=port, database=db_name)
        self.engine = sa.create_engine(url)

        calendar_df = pd.read_sql_table('交易日历', self.engine)
        self._calendar = calendar_df['交易日期'].dt.to_pydatetime().tolist()
        stock_list_df = pd.read_sql_table('股票代码', self.engine)
        self._stock_list = stock_list_df['证券代码'].values.tolist()

    def get_factor(self, table_name: str, factor_name: str,
                   start_date: DateType = None, end_date: DateType = None,
                   stock_list: Sequence[str] = None) -> Factor:
        primary_keys = self._check_args_and_get_primary_keys(table_name, [factor_name])

        query_columns = primary_keys + [factor_name]
        series = pd.read_sql_table(table_name, self.engine, index_col=primary_keys, columns=query_columns)
        series.sort_index(inplace=True)
        # todo: drop_level?
        df = series.unstack().loc[:, factor_name]

        df = self._reindex_df(df, start_date, end_date, stock_list)
        factor = Factor(df, factor_name)
        return factor

    def get_financial_factor(self, table_name: str, factor_name: str,
                             start_date: DateType = None, end_date: DateType = None,
                             stock_list: Sequence[str] = None) -> Factor:
        query_columns = ['报告期', factor_name]
        primary_keys = self._check_args_and_get_primary_keys(table_name, query_columns)
        series = pd.read_sql_table(table_name, self.engine, index_col=primary_keys, columns=query_columns)
        series.sort_index(inplace=True)
        df = series.unstack().loc[:, factor_name]

        df = self._reindex_df(df, start_date, end_date, stock_list)
        factor = Factor(df, factor_name)
        return factor

    # helper functions
    def _check_args_and_get_primary_keys(self, table_name: str, factor_names: Sequence[str]) -> List[str]:
        meta = sa.MetaData(bind=self.engine)
        meta.reflect()
        assert table_name in meta.tables.keys(), f'数据库中不存在表 {table_name}'
        columns = [it.name for it in meta.tables[table_name].c]
        for factor_name in factor_names:
            assert factor_name in columns, f'表 {table_name} 中不存在 {factor_name} 列'
        primary_keys = sorted(list({'DateTime', 'ID'} & set(columns)))
        return primary_keys

    def _reindex_df(self, df, start_date: DateType = None, end_date: DateType = None,
                    stock_list: Sequence[str] = None) -> pd.DataFrame:
        date_list = select_dates(self._calendar, start_date, end_date)
        df.reindex(date_list)
        if not stock_list:
            stock_list = self._stock_list
        df.reindex(stock_list, axis=1)
        return df
