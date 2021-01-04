import datetime as dt
import unittest

from AShareData import get_db_interface, IndustryComparison, set_global_config


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_global_config('config.json')
        db_interface = get_db_interface()
        self.industry_obj = IndustryComparison(index='000905.SH', industry_provider='中信', industry_level=2)

    def test_something(self):
        holding = self.industry_obj.import_holding('holding.xlsx', date=dt.datetime(2020, 12, 18))
        print(self.industry_obj.holding_comparison(holding))


if __name__ == '__main__':
    unittest.main()
