import unittest

from AShareData import set_global_config, TushareData


class Tushare2MySQLTest(unittest.TestCase):
    def setUp(self) -> None:
        set_global_config('config.json')
        self.downloader = TushareData()

    def test_calendar(self):
        print(self.downloader.calendar.calendar)

    def test_financial(self):
        self.downloader.get_financial('300146.SZ')

    def test_index(self):
        self.downloader.get_index_daily()

    def test_ipo_info(self):
        self.downloader.get_ipo_info()

    def test_all_past_names(self):
        self.downloader.init_stock_names()

    def test_past_names(self):
        self.downloader.update_stock_names()

    def test_company_info(self):
        self.downloader.get_company_info()

    def test_daily_hq(self):
        self.downloader.get_daily_hq(start_date='2010917')

    def test_all_dividend(self):
        self.downloader.get_all_dividend()

    def test_routine(self):
        # self.downloader.update_routine()
        pass

    def test_hs_const(self):
        self.downloader.get_hs_constitute()

    def test_shibor(self):
        self.downloader.get_shibor(end_date='20111010')

    def test_index_weight(self):
        self.downloader.get_index_weight(start_date='20050101')


if __name__ == '__main__':
    unittest.main()
