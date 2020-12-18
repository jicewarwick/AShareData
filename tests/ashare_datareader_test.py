import datetime as dt
import unittest

from AShareData.AShareDataReader import AShareDataReader
from AShareData.DBInterface import MySQLInterface, prepare_engine


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        engine = prepare_engine(config_loc)
        self.db = AShareDataReader(MySQLInterface(engine))

    def test_warned_stocks(self):
        start_date = dt.date(2010, 1, 8)
        end_date = dt.date(2019, 12, 27)
        self.db.risk_warned_stocks(start_date=start_date, end_date=end_date)

    def test_pause_stocks(self):
        start_date = dt.date(2010, 1, 8)
        end_date = dt.date(2019, 12, 27)
        self.db.paused_stocks(start_date=start_date, end_date=end_date)

    def test_calendar(self):
        print(self.db.calendar.calendar)

    def test_adj_factor(self):
        start_date = dt.date(2018, 5, 10)
        end_date = dt.date(2018, 7, 10)
        dates = [start_date, end_date]
        ids = ['000001.SZ', '600000.SH', '000002.SZ']
        print(self.db.adj_factor(start_date=start_date, end_date=end_date, ids=ids))
        print(self.db.adj_factor(start_date=start_date, ids=ids))
        print(self.db.adj_factor(end_date=end_date, ids=ids))
        print(self.db.adj_factor(dates=dates, ids=ids))

    def test_stocks(self):
        print(self.db.stocks)

    def test_get_factor(self):
        factor_name = '证券名称'
        table_name = '股票曾用名'
        print(self.db.get_factor(table_name, factor_name).tail())

    def test_get_financial_factor(self):
        self.fail()

    def test_industry(self):
        print(self.db.get_industry('中信', 3).tail())
        print(self.db.get_industry('中证', 3).tail())

    def test_snapshot(self):
        factor_name = '证券名称'
        table_name = '股票曾用名'
        print(self.db.get_snapshot(table_name, factor_name))

    def test_industry_snapshot(self):
        print(self.db.get_industry_snapshot('中信', 3))
        print(self.db.get_industry_snapshot('中证', 3))

    def test_financial_query(self):
        print(self.db.query_financial_statements('资产负债表', '资产总计', '20181231'))

    def test_financial_snapshot(self):
        print(self.db.get_financial_snapshot('资产负债表', '资产总计', yearly=True))
        print(self.db.get_financial_snapshot('资产负债表', '资产总计', quarterly=True))

    def test_index_constitute(self):
        print(self.db.index_constitute('000300.SH', '20201130'))


if __name__ == '__main__':
    unittest.main()
