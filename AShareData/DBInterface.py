import datetime as dt
import json
import logging
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import Boolean, Column, DateTime, Float, func, Integer, Table, Text, VARCHAR
from sqlalchemy.dialects.mysql import DOUBLE, insert
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker

from AShareData.utils import compute_diff


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

    def get_progress(self, table_name: str) -> Tuple[Optional[dt.datetime], Optional[str]]:
        raise NotImplementedError()

    def read_table(self, table_name: str,
                   index_col: Sequence[str] = None, columns: Sequence[str] = None) -> pd.DataFrame:
        raise NotImplementedError()

    def get_all_id(self, table_name: str) -> Optional[List[str]]:
        raise NotImplementedError()

    def exist_table(self, table_name: str) -> bool:
        raise NotImplementedError()

    def get_table_columns_names(self, table_name: str) -> List[str]:
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

    def __init__(self, engine: sa.engine.Engine) -> None:
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
        self.meta = sa.MetaData(bind=self.engine)
        self.meta.reflect()

    def create_table(self, table_name: str, table_info: Mapping[str, str]) -> None:
        """
        创建表

        :param table_name: 表名
        :param table_info: dict{字段名: 类型}
        """
        col_names = list(table_info.keys())
        col_types = [self.type_mapper[it] for it in table_info.values()]
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
        metadata = sa.MetaData(self.engine)
        metadata.reflect()
        assert table_name in metadata.tables.keys(), f'数据库中无名为 {table_name} 的表'
        table = metadata.tables[table_name]
        conn = self.engine.connect()
        conn.execute(table.delete())
        logging.info(f'table {table_name} purged')

    def update_df(self, df: pd.DataFrame, table_name: str) -> None:
        """ 将DataFrame写入数据库"""
        if df.shape[0] == 0:
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
        if df.shape[0] == 0:
            return
        current_data = self.read_table(table_name, index_col=['DateTime', 'ID'])
        if current_data.shape[0] == 0:
            self.update_df(df, table_name)
        else:
            current_date = df.index.get_level_values(0).to_pydatetime()[0]
            current_data = current_data.loc[current_data.index.get_level_values(0) < current_date, :]
            new_info = compute_diff(df, current_data)
            self.update_df(new_info, table_name)

    def get_progress(self, table_name: str) -> Tuple[Optional[dt.datetime], Optional[str]]:
        """
        返回数据库中最新的时间和证券代码

        :param table_name: 表名
        :return: (最新时间, 最大证券代码)
        """
        table_name = table_name.lower()
        latest_time = None
        latest_id = None

        assert table_name in self.meta.tables.keys(), f'数据库中无名为 {table_name} 的表'
        table = self.meta.tables[table_name]
        session_maker = sessionmaker(bind=self.engine)
        session = session_maker()
        if 'DateTime' in table.columns.keys():
            logging.debug(f'{table_name} 表中找到时间列')
            latest_time = session.query(func.max(table.c.DateTime)).one()[0]
        if 'ID' in table.columns.keys():
            logging.debug(f'{table_name} 表中找到ID列')
            latest_id = session.query(func.max(table.c.ID)).one()[0]

        return latest_time, latest_id

    def get_all_id(self, table_name: str) -> Optional[List[str]]:
        """
        返回数据库中的所有股票代码

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

    def read_table(self, table_name: str, index_col: Sequence[str] = None,
                   columns: Sequence[str] = None, where_clause: str = None) -> pd.DataFrame:
        """ 读取数据库中的表

        :param table_name: 表名
        :param index_col: 需设为输出的索引名
        :param columns: 所需的列名
        :param where_clause: 所需数据条件
        :return:
        """
        if where_clause is None:
            return pd.read_sql_table(table_name=table_name, con=self.engine, index_col=index_col, columns=columns)
        else:
            # construct sql command
            sql = f'SELECT {", ".join(columns)} FROM {table_name} WHERE {where_clause}'
            return pd.read_sql(sql, con=self.engine, index_col=index_col, columns=columns)

    def exist_table(self, table_name: str) -> bool:
        """ 数据库中是否存在该表"""
        return table_name in self.meta.tables.keys()

    def get_table_columns_names(self, table_name: str) -> List[str]:
        """ 数据库中表中的所有列"""
        table = sa.Table(table_name, self.meta)
        return [it.name for it in table.columns]


def get_stocks(db_interface: DBInterface) -> List[str]:
    stock_list_df = db_interface.read_table('股票上市退市')
    return sorted(stock_list_df['ID'].unique().tolist())
