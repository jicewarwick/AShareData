import unittest

from AShareData import generate_db_interface_from_config
from AShareData.Factor import *
from AShareData.Tickers import *


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        self.db_interface = generate_db_interface_from_config(config_loc)
        self.start_date = dt.datetime(2005, 1, 1)
        self.end_date = dt.datetime(2010, 1, 1)
        # self.ids = ['000001.SZ']
        self.ids = StockTickers(self.db_interface).ticker(dt.date(2005, 1, 1))

    def test_compact_record_factor(self):
        compact_factor = CompactFactor('证券名称', self.db_interface)
        compact_factor.data = compact_factor.data.map(lambda x: 'PT' in x or 'ST' in x or '退' in x)
        compact_record_factor = CompactRecordFactor(compact_factor, 'ST')
        print(compact_record_factor.get_data(date=dt.datetime(2015, 5, 15)))

    def test_compact_factor(self):
        compact_factor = CompactFactor('证券名称', self.db_interface)
        print(compact_factor.get_data(dates=[dt.datetime(2015, 5, 15)]))

    def test_industry(self):
        print('')
        industry_factor = IndustryFactor('中信', 2, self.db_interface)
        print(industry_factor.list_constitutes(dt.datetime(2019, 1, 7), '白酒'))
        print('')
        print(industry_factor.all_industries)

    def test_latest_accounting_factor(self):
        f = LatestAccountingFactor('未分配利润', self.db_interface)
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(a)

    def test_latest_quarter_report_factor(self):
        f = LatestQuarterAccountingFactor('未分配利润', self.db_interface)
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(a)

    def test_yearly_report_factor(self):
        f = YearlyReportAccountingFactor('未分配利润', self.db_interface)
        ids = list(set(self.ids) - set(['600087.SH', '600788.SH', '600722.SH']))
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=ids)
        print(a)

    def test_qoq_report_factor(self):
        f = QOQAccountingFactor('未分配利润', self.db_interface)
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(a)

    def test_yoy_period_report_factor(self):
        f = YOYPeriodAccountingFactor('未分配利润', self.db_interface)
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(a)

    def test_yoy_quarter_factor(self):
        f = YOYQuarterAccountingFactor('未分配利润', self.db_interface)
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(a)

    def test_ttm_factor(self):
        f = TTMAccountingFactor('营业总收入', self.db_interface)
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(a)

    def test_index_constitute(self):
        index_constitute = IndexConstitute(self.db_interface, '指数成分股权重')
        print(index_constitute.get_data('000300.SH', '20200803'))


if __name__ == '__main__':
    unittest.main()
