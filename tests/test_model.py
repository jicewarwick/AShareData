import datetime as dt
import unittest

from AShareData import set_global_config
from AShareData.model.fama_french_3_factor_model import FamaFrench3FactorModel


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_ff_model():
        set_global_config('config.json')
        date = dt.datetime(2020, 3, 3)
        model = FamaFrench3FactorModel()
        # self = model
        print(model.compute_daily_factor_return(date))


if __name__ == '__main__':
    unittest.main()
