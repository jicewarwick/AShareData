import datetime as dt
import unittest

from AShareData.ashare_data_reader import AShareDataReader
from AShareData.config import set_global_config


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_global_config('config.json')
        self.db = AShareDataReader()
        self.start_date = dt.datetime(2018, 5, 10)
        self.end_date = dt.datetime(2018, 7, 10)
        self.ids = ['000001.SZ', '600000.SH', '000002.SZ']
        self.dates = [self.start_date, self.end_date]

    def test_calendar(self):
        print(self.db.calendar.calendar)

    def test_adj_factor(self):
        print(self.db.adj_factor.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids))
        print(self.db.adj_factor.get_data(start_date=self.start_date, ids=self.ids))
        print(self.db.adj_factor.get_data(end_date=self.end_date, ids=self.ids))
        print(self.db.adj_factor.get_data(dates=self.dates, ids=self.ids))

    def test_stocks(self):
        print(self.db.stocks)

    def test_get_sec_name(self):
        start_date = dt.date(2018, 5, 10)
        print(self.db.sec_name.get_data(dates=start_date))

    def test_industry(self):
        start_date = dt.date(2018, 5, 10)
        print(self.db.industry('中信', 3).get_data(dates=start_date))
        print(self.db.industry('中证', 3).get_data(dates=start_date))

    def test_index_constitute(self):
        print(self.db.index_constitute.get_data('000300.SH', '20201130'))

    def test_ttm(self):
        print(self.db.earning_ttm.get_data(dates=self.dates, ids=self.ids))
        print(self.db.stock_market_cap.get_data(dates=self.dates, ids=self.ids))
        print(self.db.pe_ttm.get_data(dates=self.dates, ids=self.ids))
        print(self.db.pb_after_close.get_data(dates=self.dates, ids=self.ids))

    def test_cap_weight(self):
        print(self.db.free_floating_cap_weight.get_data(dates=[self.start_date, self.end_date], ids=self.ids))


if __name__ == '__main__':
    unittest.main()
