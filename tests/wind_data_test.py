import logging
import unittest

from AShareData.DBInterface import MySQLInterface, prepare_engine
from AShareData.WindData import WindData

logging.basicConfig(format='%(asctime)s  %(name)s  %(levelname)s: %(message)s', level=logging.DEBUG)


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        engine = prepare_engine(config_loc)
        self.wind_data = WindData(MySQLInterface(engine))

    def test_get_industry_func(self):
        wind_code = '000019.SZ'
        start_date = '20161212'
        end_date = '20190905'
        provider = '中证'
        start_data = '软饮料'
        end_data = '食品经销商'
        print(self.wind_data._find_industry(wind_code, provider, start_date, start_data, end_date, end_data))

    def test_update_zz_industry(self):
        self.wind_data.update_industry('中证')

    def test_update_sw_industry(self):
        self.wind_data.update_industry('申万')

    def test_update_wind_industry(self):
        self.wind_data.update_industry('Wind')

    def test_minutes_data(self):
        self.assertRaises(AssertionError, self.wind_data.get_minutes_data, '20191001')
        # print(self.wind_data.get_minutes_data('20161017'))

    def test_update_minutes_data(self):
        self.wind_data.update_minutes_data()


if __name__ == '__main__':
    unittest.main()
