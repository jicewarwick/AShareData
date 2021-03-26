import datetime as dt
import unittest

from AShareData import set_global_config
from AShareData.model import *


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_global_config('config.json')

    def test_something(self):
        self.assertEqual(True, False)

    @staticmethod
    def test_FF3factor_return():
        model = FamaFrench3FactorModel()
        smb = SMBandHMLCompositor(model)
        date = dt.datetime(2021, 3, 9)
        pre_date = dt.datetime(2021, 3, 8)
        pre_month_date = dt.datetime(2021, 2, 26)
        smb.compute_factor_return(balance_date=pre_date, pre_date=pre_date, date=date,
                                  rebalance_marker='D', period_marker='D')
        smb.compute_factor_return(balance_date=pre_month_date, pre_date=pre_date, date=date,
                                  rebalance_marker='M', period_marker='D')
        smb.compute_factor_return(balance_date=pre_month_date, pre_date=pre_month_date, date=date,
                                  rebalance_marker='M', period_marker='M')

    @staticmethod
    def test_FFC4_factor_return():
        model = FamaFrenchCarhart4FactorModel()
        umd = UMDCompositor(model)
        date = dt.datetime(2021, 3, 9)
        pre_date = dt.datetime(2021, 3, 8)
        pre_month_date = dt.datetime(2021, 2, 26)
        umd.compute_factor_return(balance_date=pre_date, pre_date=pre_date, date=date,
                                  rebalance_marker='D', period_marker='D')
        umd.compute_factor_return(balance_date=pre_month_date, pre_date=pre_date, date=date,
                                  rebalance_marker='M', period_marker='D')
        umd.compute_factor_return(balance_date=pre_month_date, pre_date=pre_month_date, date=date,
                                  rebalance_marker='M', period_marker='M')


if __name__ == '__main__':
    unittest.main()
