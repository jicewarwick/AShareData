import sqlalchemy as sa
from sqlalchemy.engine.url import URL
import pandas as pd

class Factor(object):
    def __init__(self, df: pd.DataFrame, name: str = None):
        self._name = name
        self._data = df

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def __mul__(self, other):
        return Factor(self.data * other.data)

    # def fill_na(self):


class FinancialFactor(Factor):
    def __init__(self, df: pd.DataFrame, name: str = None):
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

    def get_factor(self, table_name: str, factor_name: str) -> Factor:
        meta = sa.MetaData(bind=self.engine)
        meta.reflect()
        assert table_name in meta.tables.keys(), f'数据库中不存在表 {table_name}'
        columns = [it.name for it in meta.tables[table_name].c]
        assert factor_name in columns, f'表 {table_name} 中不存在 {factor_name} 列'
        primary_keys = sorted(list({'DateTime', 'ID'} & set(columns)))

        series = pd.read_sql_table(table_name, self.engine, index_col=primary_keys, columns=primary_keys + [factor_name])
        series.sort_index(inplace=True)
        df = series.unstack().loc[:, factor_name]
        factor = Factor(df, factor_name)
        return factor


