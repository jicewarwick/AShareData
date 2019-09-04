import logging
import unittest

from AShareData.FactorZoo import FactorZoo
from AShareData.SQLDBReader import SQLDBReader
from AShareData.utils import prepare_engine

logging.basicConfig(format='%(asctime)s  %(name)s  %(levelname)s: %(message)s', level=logging.DEBUG)


class FactorZooTest(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = '../config.json'
        db = SQLDBReader(prepare_engine(config_loc))
        self.factor_zoo = FactorZoo(db)

    def test_close_bfq(self):
        factor = self.factor_zoo.close
        self.assertAlmostEqual(factor.loc['2009-08-03', '000001.SZ'], 25.85)

    def test_close_hfq(self):
        hfq_close = self.factor_zoo.hfq_close
        # todo: wind gives 928.16!!!
        self.assertAlmostEqual(hfq_close.loc['2009-08-03', '000001.SZ'], 928.17, delta=0.01)

    def test_name_factor(self):
        names = self.factor_zoo.names
        print(names.data.fillna(method='ffill').tail())

    def test_latest_year_equity(self):
        output = self.db.get_financial_factor('合并资产负债表', '股东权益合计(不含少数股东权益)',
                                              agg_func=lambda x: x.tail(1), yearly=True)
        print(output.data.fillna(method='ffill').tail())

    def test_zx_industry(self):
        output = self.factor_zoo.zx_industry(4)
        print(output.tail(1).T)

    def test_shibor(self):
        print(self.factor_zoo.shibor_rate())

    def test_list_days(self):
        print(self.factor_zoo.listed_more_than_n_days(21))

    def test_excess_return_factor(self):
        self.fail()

    def test_financial_ttm(self):
        self.fail()

    def test_standardized(self):
        self.fail()

    def test_winsorized(self):
        self.fail()

    def test_rank(self):
        self.fail()

    def test_rolling_func(self):
        self.fail()

    def test_expanding_func(self):
        self.fail()


if __name__ == '__main__':
    unittest.main()
