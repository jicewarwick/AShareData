import json
import logging
from unittest import TestCase, main

from FactorZoo import FactorZoo
from SQLDBReader import SQLDBReader
from TushareData import TushareData
from WebData import WebDataCrawler
from utils import prepare_engine

logging.basicConfig(format='%(asctime)s  %(name)s  %(levelname)s: %(message)s', level=logging.DEBUG)


class FactorZooTest(TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        db = SQLDBReader(prepare_engine(config_loc))
        self.factor_zoo = FactorZoo(db)

    def test_close_bfq(self):
        factor = self.factor_zoo.close
        self.assertAlmostEqual(factor.loc['2009-08-03', '000001.SZ'], 25.85)

    def test_close_hfq(self):
        hfq_close = self.factor_zoo.hfq_close
        # todo: wind gives 928.16!!!
        self.assertAlmostEqual(hfq_close.loc['2009-08-03', '000001.SZ'], 928.17, delta=0.01)

    def test_name_factor(self):
        names = self.factor_zoo.names
        print(names.data.fillna(method='ffill').tail())

    def test_latest_year_equity(self):
        output = self.db.get_financial_factor('合并资产负债表', '股东权益合计(不含少数股东权益)',
                                              agg_func=lambda x: x.tail(1), yearly=True)
        print(output.data.fillna(method='ffill').tail())

    def test_shibor(self):
        print(self.factor_zoo.shibor_rate())

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
        config_loc = 'config.json'
        with open(config_loc, 'r') as f:
            config = json.load(f)

        tushare_token = config['tushare_token']

        tushare_parameters_db = 'param.json'
        db_schema = 'db_schema.json'
        self.downloader = TushareData(tushare_token,
                                      param_json=tushare_parameters_db, db_schema=db_schema,
                                      engine=prepare_engine(config_loc))

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

    def test_dividend(self):
        self.downloader.get_dividend()

    def test_routine(self):
        self.downloader.update_routine()

    def test_hs_const(self):
        self.downloader.get_hs_const()

    def test_shibor(self):
        self.downloader.get_shibor(end_date='20111010')


class WebDataSourceTest(TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        tushare_parameters_db = 'param.json'
        self.web_crawler = WebDataCrawler(prepare_engine(config_loc), tushare_parameters_db)

    def test_industry(self):
        self.web_crawler.get_sw_industry()


if __name__ == '__main__':
    main()
