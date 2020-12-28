import datetime as dt
import unittest

from AShareData import generate_db_interface_from_config, IndustryComparison


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        db_interface = generate_db_interface_from_config(config_loc)
        self.industry_obj = IndustryComparison(db_interface, index='000905.SH', industry_provider='中信',
                                               industry_level=2)

    def test_something(self):
        holding = self.industry_obj.import_holding('holding.xlsx', date=dt.datetime(2020, 12, 18))
        print(self.industry_obj.holding_comparison(holding))


if __name__ == '__main__':
    unittest.main()
