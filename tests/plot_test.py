import unittest

from AShareData.config import set_global_config
from AShareData.factor import ContinuousFactor
from AShareData.plot import plot_index


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_global_config('config.json')

    def test_plot_factor_portfolio_return(self):
        factor_name = 'Beta'
        weight = True
        industry_neutral = True
        bins = 5
        start_date = None
        end_date = None
        db_interface = None

    def test_plot_index(self):
        index_factor = ContinuousFactor('自合成指数', '收益率')
        index_factor.bind_params(ids='ST.IND')
        benchmark_factor = ContinuousFactor('自合成指数', '收益率')
        benchmark_factor.bind_params(ids='全市场.IND')
        start_date = None
        end_date = None
        ids = 'ST.IND'
        plot_index(index_factor)
        plot_index(index_factor, benchmark_factor=benchmark_factor)


if __name__ == '__main__':
    unittest.main()
