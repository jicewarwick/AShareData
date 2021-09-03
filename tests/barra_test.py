import unittest

import AShareData as asd
from AShareData.factor_compositor.barra.descriptors import *


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        asd.set_global_config('config.json')
        self.data_reader = BarraDataReader()

    def test_data_reader(self):
        print(self.data_reader.excess_market_return.get_data(dates=dt.datetime(2021, 8, 31)))
        print(self.data_reader.excess_return.get_data(dates=dt.datetime(2021, 8, 31)))


if __name__ == '__main__':
    unittest.main()
