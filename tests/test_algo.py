import unittest
from AShareData.algo import *


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


if __name__ == '__main__':
    unittest.main()
