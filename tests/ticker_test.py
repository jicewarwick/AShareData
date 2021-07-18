import unittest

from AShareData.config import set_global_config
from AShareData.tickers import *
from AShareData.utils import StockSelectionPolicy


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_global_config('config.json')
        self.db_interface = get_db_interface()

    @staticmethod
    def ticker_test(ticker_obj):
        ticker_obj.all_ticker()
        tickers = (ticker_obj.ticker(dt.date(2020, 9, 30)))
        print(tickers)
        print(len(tickers))

    def test_stock_ticker(self):
        stock_ticker = StockTickers(self.db_interface)
        self.ticker_test(stock_ticker)
        stock_ticker.get_list_date('000001.SZ')

        start_date = dt.datetime(2018, 1, 1)
        end_date = dt.datetime(2018, 12, 1)
        print(stock_ticker.new_ticker(start_date=start_date, end_date=end_date))

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

    def test_bond_etf_ticker(self):
        stock_etf = BondETFTickers(self.db_interface)
        self.ticker_test(stock_etf)

    def test_active_stock_ticker(self):
        ticker = ActiveManagedStockFundTickers(True, self.db_interface)
        self.ticker_test(ticker)

    def test_exchange_fund_ticker(self):
        ticker = ExchangeFundTickers(self.db_interface)
        self.ticker_test(ticker)

    def test_option_ticker(self):
        ticker = OptionTickers(self.db_interface)
        self.ticker_test(ticker)

    def test_ticker_selection(self):
        policy = StockSelectionPolicy()
        policy.ignore_new_stock_period = 360
        policy.select_st = False
        policy.max_pause_days = (2, 5)
        selector = StockTickerSelector(policy=policy, db_interface=self.db_interface)
        dates = [dt.datetime(2020, 1, 7), dt.datetime(2020, 12, 28)]
        ret = selector.generate_index(dates=dates)
        print(ret)

    def test_new_ticker_selection(self):
        policy = StockSelectionPolicy()
        policy.ignore_new_stock_period = 60
        policy.select_new_stock_period = 90
        policy.select_st = False
        selector = StockTickerSelector(policy=policy, db_interface=self.db_interface)
        dates = [dt.datetime(2020, 1, 7), dt.datetime(2020, 12, 28)]
        ret = selector.generate_index(dates=dates)
        print(ret)


if __name__ == '__main__':
    unittest.main()
