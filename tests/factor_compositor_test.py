import unittest

import AShareData.DateUtils
from AShareData.config import set_global_config
from AShareData.FactorCompositor import *


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_global_config('config.json')
        self.factor_compositor = FactorCompositor()

    def test_market_return(self):
        ticker: str = '000001.IND'
        ignore_new_stock_period: dt.timedelta = dt.timedelta(days=252)
        unit_base: str = '自由流通股本'
        start_date: AShareData.DateUtils.DateType = dt.datetime(1999, 5, 4)


if __name__ == '__main__':
    unittest.main()
