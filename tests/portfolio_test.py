import datetime as dt
import unittest

from AShareData import set_global_config, TradingCalendar
from AShareData.PortfolioAnalysis import ASharePortfolioAnalysis, CrossSectionalPortfolioAnalysis
from AShareData.Tickers import StockTickerSelector
from AShareData.utils import StockSelectionPolicy


class MyTestCase(unittest.TestCase):
    def setUp(self):
        set_global_config('config.json')
        self.portfolio_analysis = ASharePortfolioAnalysis()
        self.data_reader = self.portfolio_analysis.data_reader

    def test_summary_statistics(self):
        start_date = dt.date(2008, 1, 1)
        end_date = dt.date(2020, 1, 1)
        price = self.data_reader.get_factor('股票日行情', '收盘价', start_date=start_date, end_date=end_date)
        self.portfolio_analysis.summary_statistics(price)

    def test_cross_sectional_portfolio_analysis(self):
        forward_return = self.data_reader.forward_return
        factors = self.data_reader.log_cap
        ticker_selector = StockTickerSelector(StockSelectionPolicy())
        industry = None,
        market_cap = None
        start_date = dt.datetime(2019, 1, 1)
        end_date = dt.datetime(2021, 2, 1)
        dates = TradingCalendar().first_day_of_month(start_date, end_date)

        p = CrossSectionalPortfolioAnalysis(forward_return, factors=factors, dates=dates,
                                            ticker_selector=ticker_selector)
        self = p


if __name__ == '__main__':
    unittest.main()
