import unittest

from AShareData.DBInterface import MySQLInterface, prepare_engine
from AShareData.Tickers import *


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        engine = prepare_engine(config_loc)
        self.db_interface = MySQLInterface(engine)

    @staticmethod
    def ticker_test(ticker_obj):
        ticker_obj.all_ticker()
        tickers = (ticker_obj.ticker(dt.date(2020, 9, 30)))
        print(tickers)
        print(len(tickers))

    def test_stock_ticker(self):
        stock_ticker = StockTickers(self.db_interface)
        self.ticker_test(stock_ticker)

    def test_future_ticker(self):
        future_ticker = FutureTickers(self.db_interface)
        self.ticker_test(future_ticker)

    def test_etf_option_ticker(self):
        etf_option_ticker = ETFOptionTickers(self.db_interface)
        self.ticker_test(etf_option_ticker)

    def test_etf_ticker(self):
        etf_ticker = ETFTickers(self.db_interface)
        self.ticker_test(etf_ticker)

    def test_stock_etf_ticker(self):
        stock_etf = ExchangeStockETFTickers(self.db_interface)
        self.ticker_test(stock_etf)

    def test_active_stock_ticker(self):
        ticker = ActiveManagedOTCStockFundTickers(self.db_interface)
        self.ticker_test(ticker)

    def test_exchange_fund_ticker(self):
        ticker = ExchangeFundTickers(self.db_interface)
        self.ticker_test(ticker)

    def test_option_ticker(self):
        ticker = OptionTickers(self.db_interface)
        self.ticker_test(ticker)


if __name__ == '__main__':
    unittest.main()
