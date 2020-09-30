import unittest
from AShareData.Factor import *
from AShareData.DBInterface import MySQLInterface, prepare_engine


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        engine = prepare_engine(config_loc)
        self.db_interface = MySQLInterface(engine)

    def test_yearly_financial_data(self):
        factor = YearlyReportFinancialFactor(self.db_interface, '合并资产负债表', '商誉')
        factor.get_data(start_date=dt.date(2018, 1, 1), end_date=dt.date(2020, 8, 1))


if __name__ == '__main__':
    unittest.main()
