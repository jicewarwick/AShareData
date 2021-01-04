import datetime as dt
import unittest

from AShareData import set_global_config
from AShareData.PortfolioAnalysis import ASharePortfolioAnalysis


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


if __name__ == '__main__':
    unittest.main()
