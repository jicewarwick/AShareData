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
        rnd_close = self.w.wsd(['000001.SZ', '000002.SZ'], 'close', '2019-06-11', '2019-07-10', '')
        print(rnd_close)

    def test_wss(self):
        data = self.w.wss(['000001.SZ', '000002.SZ', '000005.SZ'], ['share_rtd_state', 'share_rtd_statejur'], unit=1,
                          tradeDate='20190714')
        print(data)

    def test_wss_fail(self):
        self.assertRaises(ValueError, self.w.wss, ['000001.SZ'], ['share_rtd_state1'], unit=1, tradeDate='20190714')

    def test_days(self):
        print(self.w.tdayscount("2019-06-15", "2019-07-15", ""))
        print(self.w.tdaysoffset(-5, "2019-07-15", ""))


if __name__ == '__main__':
    unittest.main()
