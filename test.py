import json
import logging
from unittest import TestCase, main

from SQLDBReader import SQLDBReader
from TushareData import TushareData
from WebData import WebDataCrawler


class TestSQLDBReader(TestCase):
    def setUp(self):
        config_loc = 'config.json'
        with open(config_loc, 'r') as f:
            config = json.load(f)
        self.db = SQLDBReader(config['ip'], config['port'], config['username'], config['password'], config['db_name'])

    def test_get_factor(self):
        factor = self.db.get_factor('股票日行情', '收盘价')
        self.assertAlmostEqual(factor.data.loc['2009-08-03', '000001.SZ'], 25.85)

    def test_multiply_factors(self):
        close = self.db.get_factor('股票日行情', '收盘价')
        adj_factor = self.db.get_factor('股票日行情', '复权因子')
        hfq_close = close * adj_factor
        # todo: wind gives 928.16!!!
        self.assertAlmostEqual(hfq_close.data.loc['2009-08-03', '000001.SZ'], 928.17, delta=0.01)

    def test_name_factor(self):
        name = self.db.get_factor('股票曾用名', '证券名称')
        print(name.data.fillna(method='ffill').tail())

    def test_latest_year_equity(self):
        output = self.db.get_financial_factor('合并资产负债表', '股东权益合计(不含少数股东权益)',
                                              agg_func=lambda x: x.tail(1), yearly=True)
        print(output.data.fillna(method='ffill').tail())

    def test_excess_return_factor(self):
        self.fail()

    def test_financial_ttm(self):
        self.fail()

    def test_standardized(self):
        self.fail()

    def test_winsorized(self):
        self.fail()

    def test_rank(self):
        self.fail()

    def test_rolling_func(self):
        self.fail()

    def test_expanding_func(self):
        self.fail()


class Tushare2MySQLTest(TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        config_loc = 'config.json'
        with open(config_loc, 'r') as f:
            config = json.load(f)

        tushare_token = config['tushare_token']
        ip, port, db_name = config['ip'], config['port'], config['db_name']
        username, password = config['username'], config['password']

        tushare_parameters_db = 'param.json'
        self.downloader = TushareData(tushare_token, param_json=tushare_parameters_db)
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


class WebDataSourceTest(TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        config_loc = 'config.json'
        with open(config_loc, 'r') as f:
            config = json.load(f)

        ip, port, db_name = config['ip'], config['port'], config['db_name']
        username, password = config['username'], config['password']

        tushare_parameters_db = 'param.json'
        self.web_crawler = WebDataCrawler(tushare_parameters_db, ip, port, username, password, db_name)

    def test_industry(self):
        self.web_crawler.get_sw_industry()


if __name__ == '__main__':
    main()
