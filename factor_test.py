import json
import unittest

from Factor import SQLDBReader


class TestSQLDBReader(unittest.TestCase):
    def setUp(self):
        config_loc = 'config.json'
        with open(config_loc, 'r') as f:
            config = json.load(f)
        self.db = SQLDBReader(config['ip'], config['port'], config['username'], config['password'], config['db_name'])

    def test_get_factor(self):
        factor = self.db.get_factor('股票日行情', '收盘价')
        self.assertAlmostEqual(factor.data.loc['2009-08-03', '000001.SZ'], 25.85)

    def test_multiply_factors(self):
        close = self.db.get_factor('股票日行情', '收盘价')
        adj_factor = self.db.get_factor('股票日行情', '复权因子')
        hfq_close = close * adj_factor
        # todo: wind gives 928.16!!!
        self.assertAlmostEqual(hfq_close.data.loc['2009-08-03', '000001.SZ'], 928.17, delta=0.01)

    def test_name_factor(self):
        name = self.db.get_factor('股票曾用名', '证券名称')
        print(name.data.fillna(method='ffill').tail())

    def test_latest_year_equity(self):
        output = self.db.get_financial_factor('合并资产负债表', '股东权益合计(不含少数股东权益)',
                                              agg_func=lambda x: x.tail(1), yearly=True)
        print(output.data.fillna(method='ffill').tail())

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
