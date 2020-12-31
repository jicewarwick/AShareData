import datetime as dt
import unittest

from AShareData.AShareDataReader import AShareDataReader
from AShareData.DBInterface import MySQLInterface
from AShareData.config import prepare_engine


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        engine = prepare_engine(config_loc)
        self.db = AShareDataReader(MySQLInterface(engine))

    def test_calendar(self):
        print(self.db.calendar.calendar)

    def test_adj_factor(self):
        start_date = dt.date(2018, 5, 10)
        end_date = dt.date(2018, 7, 10)
        dates = [start_date, end_date]
        ids = ['000001.SZ', '600000.SH', '000002.SZ']
        print(self.db.adj_factor.get_data(start_date=start_date, end_date=end_date, ids=ids))
        print(self.db.adj_factor.get_data(start_date=start_date, ids=ids))
        print(self.db.adj_factor.get_data(end_date=end_date, ids=ids))
        print(self.db.adj_factor.get_data(dates=dates, ids=ids))

    def test_stocks(self):
        print(self.db.stocks)

    def test_get_sec_name(self):
        start_date = dt.date(2018, 5, 10)
        print(self.db.sec_name.get_data(dates=[start_date]))

    def test_get_financial_factor(self):
        self.fail()

    def test_industry(self):
        start_date = dt.date(2018, 5, 10)
        print(self.db.industry('中信', 3).get_data(dates=[start_date]))
        print(self.db.industry('中证', 3).get_data(dates=[start_date]))
    #
    # def test_financial_query(self):
    #     print(self.db.query_financial_statements('资产负债表', '资产总计', '20181231'))
    #
    # def test_financial_snapshot(self):
    #     print(self.db.get_financial_snapshot('资产负债表', '资产总计', yearly=True))
    #     print(self.db.get_financial_snapshot('资产负债表', '资产总计', quarterly=True))

    def test_index_constitute(self):
        print(self.db.index_constitute.get_data('000300.SH', '20201130'))


if __name__ == '__main__':
    unittest.main()
