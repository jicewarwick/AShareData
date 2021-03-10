import unittest
from AShareData.algo import *
from AShareData.utils import *


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_get_less_or_equal_of_a_in_b():
        a = [5, 9, 16, 25, 60]
        a2 = [-2]
        a3 = [-2, 54]
        a4 = []
        b = list(range(20)) + [24, 55, 56]
        print(get_less_or_equal_of_a_in_b(a, b))
        print(get_less_or_equal_of_a_in_b(a2, b))
        print(get_less_or_equal_of_a_in_b(a3, b))
        print(get_less_or_equal_of_a_in_b(a4, b))

    def test_is_stock_ticker(self):
        self.assertEqual(get_stock_board_name('000001.SZ'), '主板')
        self.assertEqual(get_stock_board_name('001979.SZ'), '主板')
        self.assertEqual(get_stock_board_name('600000.SH'), '主板')
        self.assertEqual(get_stock_board_name('605500.SH'), '主板')

        self.assertEqual(get_stock_board_name('002594.SZ'), '中小板')
        self.assertEqual(get_stock_board_name('300498.SZ'), '创业板')
        self.assertEqual(get_stock_board_name('688688.SH'), '科创板')

        self.assertEqual(get_stock_board_name('0196.HK'), '非股票')
        self.assertEqual(get_stock_board_name('IF1208.CFE'), '非股票')
        self.assertEqual(get_stock_board_name('300498'), '非股票')
        self.assertEqual(get_stock_board_name(300498), '非股票')


if __name__ == '__main__':
    unittest.main()
