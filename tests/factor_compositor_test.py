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

        self.factor_compositor.update_market_return(ticker, ignore_st=True, ignore_const_limit=True, ignore_pause=True,
                                                    ignore_new_stock_period=ignore_new_stock_period,
                                                    unit_base=unit_base,
                                                    start_date=start_date)


if __name__ == '__main__':
    unittest.main()
