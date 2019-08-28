import datetime as dt
import logging
from typing import Mapping, Tuple, Optional

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import DateTime, Float, Text, Integer, VARCHAR, Table, Column, func
from sqlalchemy.dialects.mysql import DOUBLE, insert
from sqlalchemy.orm import sessionmaker


class DBWriter(object):
    def __init__(self):
        pass

    def create_table(self, table_name: str, table_info: Mapping[str, str]) -> None:
        raise NotImplementedError()

    def drop_all_tables(self) -> None:
        raise NotImplementedError()

    def update_df(self, df: pd.DataFrame, table_name: str) -> None:
        raise NotImplementedError()

    def get_progress(self, table_name: str) -> Tuple[Optional[dt.datetime], Optional[str]]:
        raise NotImplementedError()


class DataFrameMySQLWriter(DBWriter):
    type_mapper = {
        'datetime': DateTime,
        'float': Float,
        'double': DOUBLE,
        'str': Text,
        'int': Integer,
        'varchar': VARCHAR(20)
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
        # assert 'mysql' in driver, 'This class is MySQL database ONLY!!!'
        self.engine = engine

    def create_table(self, table_name: str, table_info: Mapping[str, str]) -> None:
        """
        创建表

        :param table_name: 表名
        :param table_info: dict{字段名: 类型}
        """
        meta = sa.MetaData(bind=self.engine)
        meta.reflect()
        col_names = list(table_info.keys())
        col_types = [self.type_mapper[it] for it in table_info.values()]
        primary_keys = [it for it in ['DateTime', 'ID', '报告期'] if it in col_names]
        existing_tables = [it.lower() for it in meta.tables]
        if table_name.lower() in existing_tables:
            logging.debug(f'表 {table_name} 已存在.')
            return

        new_table = Table(table_name, meta,
                          *(Column(col_name, col_type) for col_name, col_type in zip(col_names, col_types)),
                          sa.PrimaryKeyConstraint(*primary_keys))
        new_table.create()
        logging.info(f'表 {table_name} 创建成功.')

    def drop_all_tables(self) -> None:
        metadata = sa.MetaData(self.engine)
        logging.debug('DROPPING ALL TABLES')
        for table in metadata.tables.values():
            table.drop()

    def update_df(self, df: pd.DataFrame, table_name: str) -> None:
        """ 将DataFrame写入数据库"""
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
            flat_df[col] = np.where(flat_df[col].isnull(), None, flat_df[col])
        for _, row in flat_df.iterrows():
            insert_statement = insert(table).values(**row.to_dict())
            statement = insert_statement.on_duplicate_key_update(**row.to_dict())
            self.engine.execute(statement)

    def get_progress(self, table_name: str) -> Tuple[Optional[dt.datetime], Optional[str]]:
        """
        返回数据库中最新的PRIMARY KEY

        :param table_name: 表名
        :return: (最新时间, 最大证券代码)
        """
        latest_time = None
        latest_id = None

        metadata = sa.MetaData(self.engine)
        metadata.reflect()
        assert table_name in metadata.tables.keys(), f'数据库中无名为 {table_name} 的表'
        table = metadata.tables[table_name]
        session_maker = sessionmaker(bind=self.engine)
        session = session_maker()
        if 'DateTime' in table.columns.keys():
            logging.debug(f'{table_name} 表中找到时间列')
            latest_time = session.query(func.max(table.c.DateTime)).one()[0]
        if 'ID' in table.columns.keys():
            logging.debug(f'{table_name} 表中找到ID列')
            latest_id = session.query(func.max(table.c.ID)).one()[0]

        return latest_time, latest_id

    @staticmethod
    def date2str(date) -> Optional[str]:
        if isinstance(date, pd.Timestamp):
            return date.strftime('%Y-%m-%d %H:%M:%S')
