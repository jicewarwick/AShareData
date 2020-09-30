import datetime as dt
import unittest

from AShareData.DBInterface import MySQLInterface, prepare_engine
from AShareData.Ticker import *


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        engine = prepare_engine(config_loc)
        db_interface = MySQLInterface(engine)
        self.stock_ticker = StockTicker(db_interface)
        self.future_ticker = FutureTicker(db_interface)
        self.option_ticker = OptionTicker(db_interface)
        self.etf_ticker = ETFTicker(db_interface)

    def test_stock_ticker(self):
        self.stock_ticker.all_ticker()
        self.stock_ticker.ticker(dt.date(2020, 1, 1))

    def test_future_ticker(self):
        print(self.future_ticker.all_ticker())
        self.future_ticker.ticker(dt.date(2020, 1, 1))

    def test_option_ticker(self):
        self.option_ticker.all_ticker()
        self.option_ticker.ticker(dt.date(2020, 1, 1))

    def test_etf_ticker(self):
        self.etf_ticker.all_ticker()
        self.etf_ticker.ticker(dt.date(2020, 1, 1))


if __name__ == '__main__':
    unittest.main()
