import unittest

from AShareData.industry import IndustryComparison
from AShareData import prepare_engine, MySQLInterface

import pandas as pd
import datetime as dt


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        engine = prepare_engine(config_loc)
        db_interface = MySQLInterface(engine)
        self.industry_obj = IndustryComparison(db_interface)

    def test_something(self):
        holding = pd.read_csv('holding.csv', index_col=0)
        print(self.industry_obj.industry_ratio_comparison(holding, date=dt.datetime(2020, 11, 30), index='000905.SH'))


if __name__ == '__main__':
    unittest.main()
