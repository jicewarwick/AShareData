import json
import logging
import unittest

from Tushare2MySQL import Tushare2MySQL


class Tushare2MySQLTest(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        config_loc = 'config.json'
        with open(config_loc, 'r') as f:
            config = json.load(f)

        tushare_token = config['tushare_token']
        ip, port, db_name = config['ip'], config['port'], config['db_name']
        username, password = config['username'], config['password']

        tushare_parameters_db = 'param.json'
        self.downloader = Tushare2MySQL(tushare_token, param_json=tushare_parameters_db)
        self.downloader.add_mysql_db(ip, port, username, password, db_name=db_name)

    def test_db_initializer(self):
        self.downloader._initialize_db_table()

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

    def test_routine(self):
        self.downloader.update_routine()


if __name__ == '__main__':
    unittest.main()
