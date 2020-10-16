import logging
from functools import cached_property
from typing import Callable

from .Factor import *
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
    def sec_name(self):
        return CompactFactor(self.db_interface, '证券名称')

    @cached_property
    def risk_warned_stocks(self) -> CompactRecordFactor:
        tmp = CompactFactor(self.db_interface, '证券名称')
        tmp.data = tmp.data.map(lambda x: 'PT' in x or 'ST' in x or '退' in x)
        compact_record_factor = CompactRecordFactor(tmp, '风险警示股')
        return compact_record_factor

    @cached_property
    def paused_stocks(self) -> OnTheRecordFactor:
        return OnTheRecordFactor(self.db_interface, '股票停牌')

    @cached_property
    def adj_factor(self) -> CompactFactor:
        return CompactFactor(self.db_interface, '复权因子')

    @cached_property
    def free_a_shares(self) -> CompactFactor:
        return CompactFactor(self.db_interface, 'A股流通股本')

    @cached_property
    def const_limit(self) -> OnTheRecordFactor:
        return OnTheRecordFactor(self.db_interface, '一字涨跌停')

    def close(self) -> ContinuousFactor:
        return ContinuousFactor(self.db_interface, '股票日行情', '收盘价')

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
        table_name = self._generate_table_name(table_type, combined, quarterly)
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
        table_name = self._generate_table_name(table_type, combined, quarterly)
        self._check_args(table_name, factor_name)

        logging.debug('开始读取数据.')
        df = self.db_interface.read_table(table_name, columns=factor_name,
                                          start_date=utils.date_type2datetime(date) - dt.timedelta(365 * 2),
                                          end_date=date)
        stock_list = get_listed_stocks(self.db_interface, date)
        logging.debug('数据读取完成.')
        if yearly:
            df = df.loc[df.index.get_level_values('报告期').month == 12, :]
        df = df.groupby('ID').tail(1).droplevel(['DateTime', '报告期']).reindex(stock_list).to_frame()
        df['DateTime'] = date
        df = df.set_index('DateTime', append=True).swaplevel().iloc[:, 0]
        return df

    def get_financial_factor(self, table_type: str, factor_name: str, agg_func: Callable, combined: bool = True,
                             quarterly: bool = False, yearly: bool = True, start_date: utils.DateType = None,
                             end_date: utils.DateType = None, stock_list: Sequence[str] = None) -> pd.DataFrame:
        """ Query ``factor_name`` from ``start_date`` to ``end_date``
        WARNING: This function takes a very long time to process all the data. Please cache your results.

        :param table_type: type of financial statements
        :param factor_name: factor name
        :param agg_func: target function to compute result. eg: lambda x: tail(1) to get the latest data
        :param start_date: start date
        :param end_date: end data
        :param stock_list: returned stocks
        :param combined: if query combined statements
        :param quarterly: if query quarterly date
        :param yearly: if only query data on yearly report
        :return: factor dataframe
        """
        assert not (quarterly & yearly), 'quarterly 和 yearly 不能同时为 True'
        assert table_type in FINANCIAL_STATEMENTS_TYPE, '非财报数据请使用 get_factor 等函数查询!'
        table_name = self._generate_table_name(table_type, combined, quarterly)
        self._check_args(table_name, factor_name)

        query_start_date = utils.date_type2datetime(start_date) - dt.timedelta(365 * 2) if start_date else None
        data = self.db_interface.read_table(table_name, factor_name, start_date=query_start_date, end_date=end_date)
        if yearly:
            data = data.loc[data.index.get_level_values('报告期').month == 12, :]

        storage = []
        all_secs = set(data.index.get_level_values('ID').unique().tolist())
        if stock_list:
            all_secs = all_secs & set(stock_list)
        for sec_id in all_secs:
            id_data = data.loc[data.index.get_level_values('ID') == sec_id, :]
            dates = id_data.index.get_level_values('DateTime').to_pydatetime().tolist()
            dates = sorted(list(set(dates)))
            for date in dates:
                date_id_data = id_data.loc[id_data.index.get_level_values('DateTime') <= date, :]
                each_date_data = date_id_data.groupby('报告期').last()
                output_data = each_date_data.apply({factor_name: agg_func})
                output_data.index = pd.MultiIndex.from_tuples([(date, sec_id)], names=['DateTime', 'ID'])
                storage.append(output_data)

        df = pd.concat(storage)
        df = self._conform_df(df.unstack(), start_date, end_date, stock_list, False)
        # name may not survive pickling
        df.name = factor_name
        return df

    @staticmethod
    def _generate_table_name(table_type: str, combined: bool, quarterly: bool) -> str:
        combined = '合并' if combined else '母公司'
        quarterly = '单季度' if (quarterly & (table_type != '资产负债表')) else ''
        table_name = f'{combined}{quarterly}{table_type}'
        return table_name
