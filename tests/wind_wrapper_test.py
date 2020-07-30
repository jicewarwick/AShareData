import datetime as dt
import unittest

from AShareData.WindData import WindWrapper


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.w = WindWrapper()
        self.w.connect()

    def test_index_constitute(self):
        hs300_constitute = self.w.get_index_constitute(index='000300.SH')
        print(hs300_constitute)

    def test_wsd(self):
        rnd_close = self.w.wsd(['000001.SZ', '000002.SZ'], 'close', '2019-07-01', '2019-07-10', '')
        print('\n')
        print(rnd_close)

    def test_wss(self):
        # data = self.w.wss(['000001.SZ', '000002.SZ', '000005.SZ'], ['SHARE_RTD_STATE', 'SHARE_RTD_STATEJUR'],
        #                   trade_date='20190715', unit='1')
        # print('\n')
        # print(data)

        data = self.w.wss(['000001.SZ', '000002.SZ', '000005.SZ'], "open,low,high,close,volume,amt",
                          trade_date='20190715',
                          priceAdj='U', cycle='D')
        print('\n')
        print(data)

        # data = self.w.wss("000001.SH,000002.SZ", "grossmargin,operateincome", "unit=1;rptDate=20191231")
        # print('\n')
        # print(data)

    def test_wset(self):
        data = self.w.wset("futurecc", startdate='2019-07-29', enddate='2020-07-29', wind_code='A.DCE')
        print('\n')
        print(data)

        start_date = dt.date(2020, 6, 30).strftime('%Y-%m-%d')
        end_date = dt.date(2020, 7, 30).strftime('%Y-%m-%d')
        exchange = 'sse'
        wind_code = '510050.SH'
        status = 'all'
        field = 'wind_code,trade_code,sec_name'
        data = self.w.wset("optioncontractbasicinfo", options=f'field={field}', startdate=start_date, enddate=end_date,
                           status=status, windcode=wind_code, exchange=exchange)
        print('\n')
        print(data)


if __name__ == '__main__':
    unittest.main()
