import datetime as dt
import unittest

import AShareData as asd


class WebDataSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        asd.set_global_config('config.json')
        self.web_crawler = asd.WebDataCrawler()
        self.calendar = asd.SHSZTradingCalendar()

    def test_sw_industry(self):
        self.web_crawler.get_sw_industry()

    def test_zx_industry(self):
        self.web_crawler.get_zz_industry(self.calendar.offset(dt.date.today(), -1))


if __name__ == '__main__':
    unittest.main()
