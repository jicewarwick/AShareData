import unittest

from AShareData.DBInterface import MySQLInterface, prepare_engine
from AShareData.Factor import *


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        engine = prepare_engine(config_loc)
        self.db_interface = MySQLInterface(engine)

    def test_yearly_financial_data(self):
        factor = YearlyReportAccountingFactor(self.db_interface, '合并资产负债表', '商誉')
        factor.get_data(start_date=dt.date(2018, 1, 1), end_date=dt.date(2020, 8, 1))

    def test_compact_record_factor(self):
        compact_factor = CompactFactor(self.db_interface, '证券名称')
        compact_factor.data = compact_factor.data.map(lambda x: 'PT' in x or 'ST' in x or '退' in x)
        compact_record_factor = CompactRecordFactor(compact_factor, 'ST')
        print(compact_record_factor.get_data(date=dt.datetime(2015, 5, 15)))

    def test_compact_factor(self):
        compact_factor = CompactFactor(self.db_interface, '证券名称')
        print(compact_factor.get_data(dates=[dt.datetime(2015, 5, 15)]))

    def test_industry(self):
        print('')
        industry_factor = IndustryFactor(self.db_interface, '中信', 2)
        print(industry_factor.list_constitutes(dt.datetime(2019, 1, 7), '白酒'))
        print('')
        print(industry_factor.all_industries())


if __name__ == '__main__':
    unittest.main()
