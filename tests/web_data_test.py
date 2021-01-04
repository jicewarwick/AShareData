import datetime as dt
import unittest

from AShareData.config import set_global_config
from AShareData.data_source.WebData import WebDataCrawler
from AShareData.DateUtils import TradingCalendar


class WebDataSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        set_global_config('config.json')
        self.web_crawler = WebDataCrawler()
        self.calendar = TradingCalendar()

    def test_sw_industry(self):
        self.web_crawler.get_sw_industry()

    def test_zx_industry(self):
        self.web_crawler.get_zz_industry(self.calendar.offset(dt.date.today(), -1))


if __name__ == '__main__':
    unittest.main()
