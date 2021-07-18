import unittest

from AShareData import set_global_config, SHSZTradingCalendar
from AShareData.analysis.holding import *
from AShareData.model.fama_french_3_factor_model import FamaFrench3FactorModel
from AShareData.portfolio_analysis import *
from AShareData.tickers import StockTickerSelector
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


class CrossSectionTesting(unittest.TestCase):
    def setUp(self):
        set_global_config('config.json')
        self.data_reader = AShareDataReader()
        forward_return = self.data_reader.forward_return
        factors = self.data_reader.log_cap
        ticker_selector = StockTickerSelector(StockSelectionPolicy())
        market_cap = self.data_reader.stock_free_floating_market_cap
        start_date = dt.datetime(2020, 8, 1)
        end_date = dt.datetime(2021, 2, 1)
        dates = SHSZTradingCalendar().first_day_of_month(start_date, end_date)

        self.t = CrossSectionalPortfolioAnalysis(forward_return, factors=factors, dates=dates, market_cap=market_cap,
                                                 ticker_selector=ticker_selector)
        self.t.cache()

    def test_single_sort(self):
        self.t.single_factor_sorting('BM')
        self.t.returns_results(cap_weighted=True)
        self.t.returns_results(cap_weighted=False)
        self.t.summary_statistics('BM')

    def test_independent_double_sort(self):
        self.t.two_factor_sorting(factor_names=('BM', '市值对数'), quantile=10, separate_neg_vals=True, independent=True)
        self.t.returns_results(cap_weighted=True)
        self.t.returns_results(cap_weighted=False)
        self.t.summary_statistics('BM')

    def test_dependent_double_sort(self):
        self.t.two_factor_sorting(factor_names=('BM', '市值对数'), quantile=10, separate_neg_vals=True, independent=False)
        self.t.returns_results(cap_weighted=True)
        self.t.returns_results(cap_weighted=False)
        self.t.summary_statistics('BM')


class PortfolioExposureTest(unittest.TestCase):
    def setUp(self):
        set_global_config('config.json')

    @staticmethod
    def test_case():
        date = dt.datetime(2021, 3, 8)
        model = FamaFrench3FactorModel()
        exposure = ASharePortfolioExposure(model=model)
        ticker = '000002.SZ'
        fh = FundHolding()
        portfolio_weight = fh.portfolio_stock_weight(date, 'ALL')
        exposure.get_stock_exposure(ticker)
        exposure.get_portfolio_exposure(portfolio_weight)


if __name__ == '__main__':
    unittest.main()
