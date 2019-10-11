import logging
import unittest

from AShareData.AShareDataReader import AShareDataReader
from AShareData.DBInterface import MySQLInterface, prepare_engine

logging.basicConfig(format='%(asctime)s  %(name)s  %(levelname)s: %(message)s', level=logging.DEBUG)


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        engine = prepare_engine(config_loc)
        self.db = AShareDataReader(MySQLInterface(engine))

    def test_calendar(self):
        print(self.db.calendar.calendar)

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


if __name__ == '__main__':
    unittest.main()
