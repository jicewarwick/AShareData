import datetime as dt
import json
import logging
from importlib.resources import open_text
from typing import List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import Boolean, Column, DateTime, Float, func, Integer, Table, Text, VARCHAR
from sqlalchemy.dialects.mysql import DOUBLE, insert
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker

import AShareData.utils as utils


class DBInterface(object):
    def __init__(self):
        pass

    def create_table(self, table_name: str, table_info: Mapping[str, str]) -> None:
        raise NotImplementedError()

    def drop_all_tables(self) -> None:
        raise NotImplementedError()

    def purge_table(self, table_name: str) -> None:
        raise NotImplementedError()

    def update_df(self, df: pd.DataFrame, table_name: str) -> None:
        raise NotImplementedError()

    def update_compact_df(self, df, table_name: str) -> None:
        raise NotImplementedError()

    def get_latest_timestamp(self, table_name: str) -> Optional[dt.datetime]:
        raise NotImplementedError()

    def read_table(self, table_name: str, columns: Sequence[str] = None) -> Union[pd.Series, pd.DataFrame]:
        raise NotImplementedError()

    def get_all_id(self, table_name: str) -> Optional[List[str]]:
        raise NotImplementedError()

    def exist_table(self, table_name: str) -> bool:
        raise NotImplementedError()

    def get_columns_names(self, table_name: str) -> List[str]:
        raise NotImplementedError()

    def get_table_primary_keys(self, table_name: str) -> Optional[List[str]]:
        raise NotImplementedError()


def prepare_engine(config_loc: str) -> sa.engine.Engine:
    with open(config_loc, 'r') as f:
        config = json.load(f)
    url = URL(drivername=config['driver'], host=config['host'], port=config['port'], database=config['database'],
              username=config['username'], password=config['password'],
              query={'charset': 'utf8mb4'})
    return sa.create_engine(url)


class MySQLInterface(DBInterface):
    type_mapper = {
        'datetime': DateTime,
        'float': Float,
        'double': DOUBLE,
        'str': Text,
        'int': Integer,
        'varchar': VARCHAR(20),
        'boolean': Boolean
    }

    def __init__(self, engine: sa.engine.Engine, init: bool = False, db_schema_loc: str = None) -> None:
        """ DataFrame to MySQL Database Writer

        Write pd.DataFrame to MySQL server. Feature:
        - Auto Create Table Set 'DateTime' and 'ID' as primary key and index if they are available as DataFrame index.
        - Add new columns when necessary
        - Handle datetime insertion
        - Handle nan insertion
        - Insert new or update old records using on_duplicate_key_update()

        :param engine: sqlalchemy engine
        """
        super().__init__()
        assert engine.name == 'mysql', 'This class is MySQL database ONLY!!!'
        self.engine = engine

        if init:
            self._create_db_schema_tables(db_schema_loc)
        self.meta = sa.MetaData(bind=self.engine)
        self.meta.reflect()

    def _create_db_schema_tables(self, db_schema_loc):
        if db_schema_loc is None:
            f = open_text('AShareData.data', 'db_schema.json')
        else:
            f = open(db_schema_loc, 'r', encoding='utf-8')
        with f:
            self._db_parameters = json.load(f)
        for special_item in ['资产负债表', '现金流量表', '利润表']:
            tmp_item = self._db_parameters.pop(special_item)
            for prefix in ['合并', '母公司']:
                for yearly in ['', '单季度']:
                    self._db_parameters[prefix + yearly + special_item] = tmp_item
        for entry in ['合并单季度资产负债表', '母公司单季度资产负债表']:
            del self._db_parameters[entry]
        for table_name, table_schema in self._db_parameters.items():
            self.create_table(table_name, table_schema)

    def create_table(self, table_name: str, table_schema: Mapping[str, str]) -> None:
        """
        创建表

        :param table_name: 表名
        :param table_schema: dict{字段名: 类型}
        """
        col_names = list(table_schema.keys())
        col_types = [self.type_mapper[it] for it in table_schema.values()]
        primary_keys = [it for it in ['DateTime', 'ID', '报告期', 'IndexCode'] if it in col_names]
        existing_tables = [it.lower() for it in self.meta.tables]
        if table_name.lower() in existing_tables:
            logging.debug(f'表 {table_name} 已存在.')
            return

        new_table = Table(table_name, self.meta,
                          *(Column(col_name, col_type) for col_name, col_type in zip(col_names, col_types)),
                          sa.PrimaryKeyConstraint(*primary_keys))
        new_table.create()
        self.meta.reflect()
        logging.info(f'表 {table_name} 创建成功.')

    def drop_all_tables(self) -> None:
        """删除database内所有的表, 谨慎使用!!!"""
        logging.debug('DROPPING ALL TABLES')
        for table in self.meta.tables.values():
            table.drop()
        self.meta.reflect()

    def purge_table(self, table_name: str) -> None:
        """删除表中的所有数据, 谨慎使用!!!"""
        assert table_name in self.meta.tables.keys(), f'数据库中无名为 {table_name} 的表'
        table = self.meta.tables[table_name]
        conn = self.engine.connect()
        conn.execute(table.delete())
        logging.info(f'table {table_name} purged')

    def update_df(self, df: pd.DataFrame, table_name: str) -> None:
        """ 将DataFrame写入数据库"""
        if df.empty:
            return

        metadata = sa.MetaData(self.engine)
        metadata.reflect()
        table = metadata.tables[table_name.lower()]
        flat_df = df.reset_index()

        date_cols = flat_df.select_dtypes(np.datetime64).columns.values.tolist()
        for col in date_cols:
            flat_df[col] = flat_df[col].apply(self.date2str)

        # replace nan to None so that insert will not error out
        # it seems that this operation changes dtypes. so do it last
        for col in flat_df.columns:
            flat_df[col] = flat_df[col].where(flat_df[col].notnull(), other=None)
        for _, row in flat_df.iterrows():
            insert_statement = insert(table).values(**row.to_dict())
            statement = insert_statement.on_duplicate_key_update(**row.to_dict())
            self.engine.execute(statement)

    def update_compact_df(self, df, table_name: str) -> None:
        if df.empty:
            return

        existing_data = self.read_table(table_name, index_col=['DateTime', 'ID'])
        if existing_data.empty:
            self.update_df(df, table_name)
        else:
            current_date = df.index.get_level_values('DateTime').to_pydatetime()[0]
            existing_data = existing_data.loc[existing_data.index.get_level_values('DateTime') < current_date, :]
            new_info = utils.compute_diff(df, existing_data)
            self.update_df(new_info, table_name)

    def get_latest_timestamp(self, table_name: str) -> Optional[dt.datetime]:
        """
        返回数据库表中最新的时间戳

        :param table_name: 表名
        :return: 最新时间
        """
        table_name = table_name.lower()
        assert table_name in self.meta.tables.keys(), f'数据库中无名为 {table_name} 的表'

        table = self.meta.tables[table_name]
        session_maker = sessionmaker(bind=self.engine)
        session = session_maker()
        if 'DateTime' in table.columns.keys():
            logging.debug(f'{table_name} 表中找到时间列')
            latest_time = session.query(func.max(table.c.DateTime)).one()[0]
            return latest_time

    def get_all_id(self, table_name: str) -> Optional[List[str]]:
        """
        返回数据库表中的所有股票代码

        :param table_name: 表名
        :return: 证券代码列表
        """
        assert table_name in self.meta.tables.keys(), f'数据库中无名为 {table_name} 的表'
        table = self.meta.tables[table_name]
        session_maker = sessionmaker(bind=self.engine)
        session = session_maker()
        if 'ID' in table.columns.keys():
            logging.debug(f'{table_name} 表中找到ID列')
            cache = session.query(table.c.ID).all()
            return sorted(list(set([it[0] for it in cache])))

    # todo: TBD
    def clean_db(self, table_name: str) -> None:
        """清理表中多余的数据. 未实现"""
        metadata = sa.MetaData(self.engine)
        metadata.reflect()
        assert table_name in metadata.tables.keys(), f'数据库中无名为 {table_name} 的表'
        table = metadata.tables[table_name]
        session_maker = sessionmaker(bind=self.engine)
        session = session_maker()
        data = self.read_table(table_name).unstack()

    @staticmethod
    def date2str(date) -> Optional[str]:
        if isinstance(date, pd.Timestamp):
            return date.strftime('%Y-%m-%d %H:%M:%S')

    def read_table(self, table_name: str, columns: Union[str, Sequence[str]] = None,
                   where_clause: str = None) -> Union[pd.Series, pd.DataFrame]:
        """ 读取数据库中的表

        :param table_name: 表名
        :param columns: 所需的列名
        :param where_clause: 所需数据条件
        :return:
        """
        if isinstance(columns, str):
            columns = [columns]
        index_col = self.get_table_primary_keys(table_name)
        if where_clause is None:
            ret = pd.read_sql_table(table_name=table_name, con=self.engine, index_col=index_col, columns=columns)
        else:
            # construct sql command
            sql = f'SELECT {", ".join(columns)} FROM {table_name} WHERE {where_clause}'
            ret = pd.read_sql(sql, con=self.engine, index_col=index_col, columns=columns)
        if ret.shape[1] == 1:
            ret = ret.iloc[:, 0]
        return ret

    def exist_table(self, table_name: str) -> bool:
        """ 数据库中是否存在该表"""
        return table_name in self.meta.tables.keys()

    def get_columns_names(self, table_name: str) -> List[str]:
        """ 数据库中表中的所有列"""
        table = sa.Table(table_name, self.meta)
        return [it.name for it in table.columns]

    def get_table_primary_keys(self, table_name: str) -> Optional[List[str]]:
        table = self.meta.tables[table_name]
        primary_key = [it.name for it in table.primary_key]
        if primary_key:
            return primary_key


def _get_stock_list_info(db_interface: DBInterface, date: utils.DateType = None) -> pd.DataFrame:
    stock_list_df = db_interface.read_table('股票上市退市')
    if date:
        date = utils.date_type2datetime(date)
        stock_list_df = stock_list_df.loc[stock_list_df.index.get_level_values('DateTime') <= date, :]
    return stock_list_df


def get_stocks(db_interface: DBInterface, date: utils.DateType = None) -> List[str]:
    stock_list_df = _get_stock_list_info(db_interface, date)
    return sorted(stock_list_df.index.get_level_values('ID').unique().tolist())


def get_listed_stocks(db_interface: DBInterface, date: utils.DateType = None) -> List[str]:
    stock_list_df = _get_stock_list_info(db_interface, date)
    tmp = stock_list_df.reset_index().groupby('ID').tail(1)
    return sorted(tmp.loc[tmp['上市状态'] == 1, 'ID'].tolist())
