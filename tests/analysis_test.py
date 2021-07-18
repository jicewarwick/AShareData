import unittest

from AShareData import *
from AShareData.analysis.fund_nav_analysis import *
from AShareData.analysis.holding import *
from AShareData.analysis.public_fund_holding import *
# from AShareData.analysis.trading import *
from AShareData.analysis.return_analysis import *
from AShareData.factor import ContinuousFactor


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_global_config('config.json')
        self.target = ContinuousFactor('自合成指数', '收益率')
        self.target.bind_params(ids='ST.IND')
        self.benchmark = ContinuousFactor('自合成指数', '收益率')
        self.benchmark.bind_params(ids='全市场.IND')
        self.start = dt.datetime(2012, 1, 1)
        self.end = dt.datetime(2020, 1, 1)

    def test_max_drawdown(self):
        returns = self.target.get_data(start_date=self.start, end_date=self.end).unstack().iloc[:, 0]
        print(locate_max_drawdown(returns))
        returns = self.benchmark.get_data().unstack().iloc[:, 0]
        print(locate_max_drawdown(returns))

    def test_aggregate_return(self):
        print(aggregate_returns(target=self.target, convert_to='monthly', benchmark_factor=self.benchmark))

    @staticmethod
    def test_holding():
        h = FundHolding()
        date = dt.datetime(2021, 3, 8)
        print(h.get_holding(date))
        print(h.get_holding(date, fund='指增1号 - 东财 - 普通户'))
        print(h.get_holding(date, fund='ALL'))

    def test_fund_nav_analysis(self):
        fund_nav_analysis = FundNAVAnalysis('110011.OF')
        fund_nav_analysis.compute_correlation('399006.SZ')
        model = FamaFrench3FactorModel()
        fund_nav_analysis.compute_exposure(model)
        fund_nav_analysis.get_latest_published_portfolio_holding()

    def test_public_fund_holding(self):
        ticker = '000001.SZ'
        date = dt.datetime(2020, 12, 31)
        rec = PublicFundHoldingRecords(ticker, date)
        self = rec


if __name__ == '__main__':
    unittest.main()
