import json
import logging
import unittest

from AShareData.TushareData import TushareData
from AShareData.utils import prepare_engine

logging.basicConfig(format='%(asctime)s  %(name)s  %(levelname)s: %(message)s', level=logging.DEBUG)


class Tushare2MySQLTest(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        with open(config_loc, 'r') as f:
            config = json.load(f)

        tushare_token = config['tushare_token']
        self.downloader = TushareData(tushare_token, engine=prepare_engine(config_loc))

    def test_calendar(self):
        print(self.downloader.calendar)

    def test_all_stocks(self):
        print(self.downloader.all_stocks)

    def test_financial(self):
        self.downloader.get_financial(['300146.SZ', '000001.SZ'])

    def test_index(self):
        self.downloader.get_index_daily()

    def test_ipo_info(self):
        self.downloader.get_ipo_info()

    def test_all_past_names(self):
        self.downloader.get_all_past_names()

    def test_company_info(self):
        self.downloader.get_company_info()

    def test_daily_hq(self):
        self.downloader.get_daily_hq(start_date='20090803')

    def test_dividend(self):
        self.downloader.get_dividend()

    def test_routine(self):
        self.downloader.update_routine()

    def test_hs_const(self):
        self.downloader.get_hs_const()

    def test_shibor(self):
        self.downloader.get_shibor(end_date='20111010')


if __name__ == '__main__':
    unittest.main()
