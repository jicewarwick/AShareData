import datetime as dt
import unittest

from AShareData import constants, set_global_config
from AShareData.data_source.wind_data import WindData, WindWrapper


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        set_global_config(config_loc)
        self.wind_data = WindData.from_config(config_loc)

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
        self.assertRaises(AssertionError, self.wind_data.get_stock_minutes_data, '20191001')
        # print(self.wind_data.get_minutes_data('20161017'))

    def test_update_minutes_data(self):
        self.wind_data.update_stock_minutes_data()

    def test_stock_daily_data(self):
        self.wind_data.get_stock_daily_data(trade_date=dt.date(2019, 12, 27))


class WindWrapperTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.w = WindWrapper()
        self.w.connect()

    def test_wsd(self):
        stock = '000001.SZ'
        stocks = ['000001.SZ', '000002.SZ']
        start_date = dt.datetime(2019, 10, 23)
        end_date = dt.datetime(2019, 10, 24)
        indicator = 'high'
        indicators = 'high,low'
        provider = '中信'

        print(self.w.wsd(stock, indicator, start_date, start_date, ''))
        print(self.w.wsd(stock, indicators, start_date, start_date, ''))
        print(self.w.wsd(stocks, indicator, start_date, start_date, ''))
        print(self.w.wsd(stock, indicator, start_date, end_date, ''))
        print(self.w.wsd(stock, indicators, start_date, end_date, ''))
        print(self.w.wsd(stocks, indicator, start_date, end_date, ''))

        print(self.w.wsd(stocks, f'industry_{constants.INDUSTRY_DATA_PROVIDER_CODE_DICT[provider]}',
                         start_date, end_date, industryType=constants.INDUSTRY_LEVEL[provider]))

    def test_wss(self):
        # data = self.w.wss(['000001.SZ', '000002.SZ', '000005.SZ'], ['SHARE_RTD_STATE', 'SHARE_RTD_STATEJUR'],
        #                   trade_date='20190715', unit='1')
        # print('\n')
        # print(data)

        data = self.w.wss(['000001.SZ', '000002.SZ', '000005.SZ'], 'open,low,high,close,volume,amt',
                          date='20190715',
                          priceAdj='U', cycle='D')
        print('\n')
        print(data)

        # data = self.w.wss("000001.SH,000002.SZ", "grossmargin,operateincome", "unit=1;rptDate=20191231")
        # print('\n')
        # print(data)

    def test_wset(self):
        data = self.w.wset('futurecc', startdate='2019-07-29', enddate='2020-07-29', wind_code='A.DCE')
        print('\n')
        print(data)

        start_date = dt.date(2020, 6, 30).strftime('%Y-%m-%d')
        end_date = dt.date(2020, 7, 30).strftime('%Y-%m-%d')
        exchange = 'sse'
        wind_code = '510050.SH'
        status = 'all'
        field = 'wind_code,trade_code,sec_name'
        data = self.w.wset('optioncontractbasicinfo', options=f'field={field}', startdate=start_date, enddate=end_date,
                           status=status, windcode=wind_code, exchange=exchange)
        print('\n')
        print(data)

    def test_wsq(self):
        data = self.w.wsq('002080.SZ,000002.SZ', 'rt_latest,rt_vol')
        print('\n')
        print(data)
        data = self.w.wsq('000002.SZ', 'rt_latest,rt_vol')
        print('\n')
        print(data)
        data = self.w.wsq('002080.SZ,000002.SZ', 'rt_latest')
        print('\n')
        print(data)
        data = self.w.wsq('000002.SZ', 'rt_latest')
        print('\n')
        print(data)

    def test_index_constitute(self):
        hs300_constitute = self.w.get_index_constitute(index='000300.SH')
        print(hs300_constitute)


if __name__ == '__main__':
    unittest.main()
