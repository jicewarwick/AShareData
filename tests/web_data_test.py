import datetime as dt
import logging
import unittest

from AShareData.DBInterface import prepare_engine
from AShareData.WebData import WebDataCrawler

logging.basicConfig(format='%(asctime)s  %(name)s  %(levelname)s: %(message)s', level=logging.DEBUG)


class WebDataSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        self.web_crawler = WebDataCrawler(prepare_engine(config_loc))

    def test_sw_industry(self):
        self.web_crawler.get_sw_industry()

    def test_zx_industry(self):
        print(self.web_crawler.get_zx_industry(dt.date(2019, 9, 2)))


if __name__ == '__main__':
    unittest.main()
