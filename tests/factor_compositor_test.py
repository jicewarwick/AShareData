import unittest

import AShareData.DateUtils
from AShareData.DBInterface import MySQLInterface, prepare_engine
from AShareData.FactorCompositor import *


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        engine = prepare_engine(config_loc)
        self.factor_compositor = FactorCompositor(MySQLInterface(engine))

    def test_market_return(self):
        ticker: str = '000001.IND'
        ignore_new_stock_period: dt.timedelta = dt.timedelta(days=252)
        unit_base: str = '自由流通股本'
        start_date: AShareData.DateUtils.DateType = dt.datetime(1999, 5, 4)

        self.factor_compositor.update_market_return(ticker, ignore_st=True, ignore_const_limit=True, ignore_pause=True,
                                                    ignore_new_stock_period=ignore_new_stock_period,
                                                    unit_base=unit_base,
                                                    start_date=start_date)

    def test_accounting_cache(self):
        cache = AccountingDateCacheCompositor(self.factor_compositor.db_interface)
        cache.update()


if __name__ == '__main__':
    unittest.main()
