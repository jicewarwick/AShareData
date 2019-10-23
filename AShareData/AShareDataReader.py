import datetime as dt
import logging
from typing import Callable, Dict, List, Sequence

import pandas as pd
from cached_property import cached_property

from . import utils
from .constants import FINANCIAL_STATEMENTS, FINANCIAL_STATEMENTS_TYPE, INDUSTRY_LEVEL
from .DBInterface import DBInterface, get_listed_stocks, get_stocks
from .TradingCalendar import TradingCalendar


class AShareDataReader(object):
    """ AShare Data Reader"""

    def __init__(self, db_interface: DBInterface) -> None:
        """
        AShare Data Reader

        :param db_interface: DBInterface
        """
        self.db_interface = db_interface

    @cached_property
    def calendar(self) -> TradingCalendar:
        """Trading Calendar"""
        return TradingCalendar(self.db_interface)

    @cached_property
    def stocks(self) -> List[str]:
        """All the stocks ever listed"""
        return get_stocks(self.db_interface)

    def listed_stock(self, date: utils.DateType = dt.date.today()) -> List[str]:
        """Get stocks still listed at ``date``"""
        return get_listed_stocks(self.db_interface, date)

    # common factors
    def get_factor(self, table_name: str, factor_name: str, ffill: bool = False,
                   start_date: utils.DateType = None, end_date: utils.DateType = None,
                   stock_list: Sequence[str] = None) -> [pd.Series, pd.DataFrame]:
        assert not (table_name in FINANCIAL_STATEMENTS), '财报数据请使用 query_financial_data 或 get_financial_factor 查询!'
        table_name = table_name.lower()
        self._check_args(table_name, factor_name)

        logging.debug('开始读取数据.')
        df = self.db_interface.read_table(table_name, columns=factor_name)
        logging.debug('数据读取完成.')
        if isinstance(df.index, pd.MultiIndex):
            df = self._conform_df(df.unstack(), ffill=ffill,
                                  start_date=start_date, end_date=end_date, stock_list=stock_list)
            # name may not survive pickling
            df.name = factor_name
        return df

    def get_snapshot(self, table_name: str, factor_name: str, date: utils.DateType = dt.date.today()) -> pd.Series:
        """Get ``factor_name`` from ``table_name`` at ``date``"""
        assert not (table_name in FINANCIAL_STATEMENTS), '财报数据请使用 query_financial_data 或 get_financial_factor 查询!'
        table_name = table_name.lower()
        self._check_args(table_name, factor_name)

        logging.debug('开始读取数据.')
        series = self.db_interface.read_table(table_name, columns=factor_name)
        logging.debug('数据读取完成.')

        df = series.reset_index()
        if date:
            date = utils.date_type2datetime(date)
            timestamp = date
            df = df.loc[df['DateTime'] <= date, :]
        else:
            timestamp = df['DateTime'].max()

        listed_stocks = get_listed_stocks(self.db_interface, timestamp)
        content = df.groupby('ID').tail(1).loc[df['ID'].isin(listed_stocks), ['ID', factor_name]]
        content['DateTime'] = timestamp
        return content.set_index(series.index.names).sort_index().iloc[:, 0]

    # industry
    def get_industry(self, provider: str, level: int, translation_json_loc: str = None,
                     start_date: utils.DateType = None, end_date: utils.DateType = None,
                     stock_list: Sequence[str] = None) -> pd.DataFrame:
        """Get industry factor

        :param provider: Industry classification data provider
        :param level: Level of industry classification
        :param translation_json_loc: custom dict specifying maximum industry classification level to lower level
        :param start_date: start date
        :param end_date: end date
        :param stock_list: query stocks
        :return: industry classification pandas.DataFrame with DateTime as index and stock as column
        """
        assert 0 < level <= INDUSTRY_LEVEL[provider], f'{provider}行业没有{level}级'

        table_name = f'{provider}行业'
        logging.debug('开始读取数据.')
        series = self.db_interface.read_table(table_name, columns='行业名称')
        logging.debug('数据读取完成.')

        if level != INDUSTRY_LEVEL[provider]:
            new_translation = self._get_industry_translation_dict(table_name, level, translation_json_loc)
            series = series.map(new_translation)

        df = self._conform_df(series.unstack(), True, start_date, end_date, stock_list)
        df.name = f'{provider}{level}级行业'
        return df

    def get_industry_snapshot(self, provider: str, level: int, translation_json_loc: str = None,
                              date: utils.DateType = dt.date.today()) -> pd.Series:
        """Get industry info at ``date``

        :param provider: Industry classification data provider
        :param level: Level of industry classification
        :param translation_json_loc: custom dict specifying maximum industry classification level to lower level
        :param date:
        :return: industry classification pandas.DataFrame with DateTime as index and stock as column
        """
        table_name = f'{provider}行业'
        factor_name = '行业名称'
        industry = self.get_snapshot(table_name, factor_name, date)
        if level != INDUSTRY_LEVEL[provider]:
            new_translation = self._get_industry_translation_dict(table_name, level, translation_json_loc)
            industry = industry.map(new_translation)
        industry.name = f'{provider}{level}级行业'
        return industry

    # financial statements
    def query_financial_statements(self, table_type: str, factor_name: str, report_period: utils.DateType,
                                   combined: bool = True, quarterly: bool = False) -> pd.Series:
        """Query financial statements

        :param table_type: type of financial statements
        :param factor_name: factor name
        :param report_period: report date
        :param combined: if query combined statements
        :param quarterly: if query quarterly date
        :return: factor series
        """
        assert table_type in FINANCIAL_STATEMENTS_TYPE, '非财报数据请使用 get_factor 等函数查询!'
        table_name = self._gen_table_name(table_type, combined, quarterly)
        self._check_args(table_name, factor_name)

        report_period = utils.date_type2datetime(report_period)

        logging.debug('开始读取数据.')
        df = self.db_interface.read_table(table_name, columns=factor_name, report_period=report_period)
        logging.debug('数据读取完成.')
        df = df.groupby(df.index.get_level_values('ID')).tail(1)
        return df

    def get_financial_snapshot(self, table_type: str, factor_name: str, date: utils.DateType = dt.date.today(),
                               combined: bool = True, quarterly: bool = False, yearly=False) -> pd.Series:
        """Query financial latest info until ``date``

        :param table_type: type of financial statements
        :param factor_name: factor name
        :param date: query date
        :param combined: if query combined statements
        :param quarterly: if query quarterly date
        :param yearly: if only query data on yearly report
        :return: factor series
        """
        assert not (quarterly & yearly), 'quarterly 和 yearly 不能同时为 True'
        assert table_type in FINANCIAL_STATEMENTS_TYPE, '非财报数据请使用 get_factor 等函数查询!'
        table_name = self._gen_table_name(table_type, combined, quarterly)
        self._check_args(table_name, factor_name)

        pre_date_str = utils.date_type2str(utils.date_type2datetime(date) - dt.timedelta(365 * 2), '-')
        date = utils.date_type2datetime(date)
        date_str = utils.date_type2str(date, '-')
        where_clause = f'DateTime >= "{pre_date_str}" AND DateTime <= "{date_str}"'

        logging.debug('开始读取数据.')
        df = self.db_interface.read_table(table_name, columns=factor_name, where_clause=where_clause)
        stock_list = get_listed_stocks(self.db_interface, date)
        logging.debug('数据读取完成.')
        if yearly:
            df = df.loc[df.index.get_level_values('报告期').month == 12, :]
        df = df.groupby('ID').tail(1).droplevel(['DateTime', '报告期']).reindex(stock_list).to_frame()
        df['DateTime'] = date
        df = df.set_index('DateTime', append=True).swaplevel().iloc[:, 0]
        return df

    # todo: TBD
    def get_financial_factor(self, table_name: str, factor_name: str, agg_func: Callable,
                             start_date: utils.DateType = None, end_date: utils.DateType = None,
                             stock_list: Sequence[str] = None, yearly: bool = True) -> pd.DataFrame:
        assert table_name in FINANCIAL_STATEMENTS, '非财报数据请使用 get_factor 等函数查询!'
        table_name = table_name.lower()
        self._check_args(table_name, factor_name)

        data = self.db_interface.read_table(table_name, columns=factor_name)
        if yearly:
            data = data.loc[data.index.get_level_values('报告期').month == 12, :]

        storage = []
        all_secs = set(data.ID.unique().tolist())
        if stock_list:
            all_secs = all_secs & set(stock_list)
        for sec_id in all_secs:
            id_data = data.loc[data.ID == sec_id, :]
            dates = id_data.DateTime.dt.to_pydatetime().tolist()
            dates = sorted(list(set(dates)))
            for date in dates:
                date_id_data = id_data.loc[data.DateTime <= date, :]
                each_date_data = date_id_data.groupby('报告期', as_index=False).last()
                each_date_data.set_index(['DateTime', 'ID', '报告期'], inplace=True)
                output_data = each_date_data.apply({factor_name: agg_func})
                output_data.index = pd.MultiIndex.from_tuples([(date, sec_id)], names=['DateTime', 'ID'])
                storage.append(output_data)

        df = pd.concat(storage)
        df = df.unstack().droplevel(None, axis=1)
        df = self._conform_df(df, False, start_date, end_date, stock_list)
        # name may not survive pickling
        df.name = factor_name
        return df

    # helper functions
    def _check_args(self, table_name: str, factor_name: str):
        table_name = table_name.lower()
        assert self.db_interface.exist_table(table_name), f'数据库中不存在表 {table_name}'

        columns = self.db_interface.get_columns_names(table_name)
        assert factor_name in columns, f'表 {table_name} 中不存在 {factor_name} 列'

    @staticmethod
    def _get_industry_translation_dict(table_name: str, level: int, translation_json_loc: str = None) -> Dict[str, str]:
        translation = utils.load_param('industry.json', translation_json_loc)
        new_translation = {}
        for key, value in translation[table_name].items():
            new_translation[key] = value[f'level_{level}']
        return new_translation

    @staticmethod
    def _gen_table_name(table_type: str, combined: bool, quarterly: bool) -> str:
        combined = '合并' if combined else '母公司'
        quarterly = '单季度' if (quarterly & (table_type != '资产负债表')) else ''
        table_name = f'{combined}{quarterly}{table_type}'
        return table_name

    def _conform_df(self, df, ffill: bool = False,
                    start_date: utils.DateType = None, end_date: utils.DateType = None,
                    stock_list: Sequence[str] = None) -> pd.DataFrame:
        if ffill:
            first_timestamp = df.index.get_level_values(0).min()
            date_list = self.calendar.select_dates(first_timestamp, end_date)
            df = df.reindex(date_list[:-1]).ffill()
            df = df.loc[start_date:, :]
        else:
            date_list = self.calendar.select_dates(start_date, end_date)
            df = df.reindex(date_list[:-1])

        if stock_list is None:
            stock_list = self.stocks
        df = df.reindex(stock_list, axis=1)
        return df
