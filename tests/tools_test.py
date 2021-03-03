import unittest

from AShareData import set_global_config
from AShareData.tools.tools import IndexHighlighter, MajorIndustryConstitutes


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_major_industry_constitute():
        set_global_config('config.json')
        provider = '申万'
        level = 2
        name = '景点'
        obj = MajorIndustryConstitutes(provider=provider, level=level)
        print(obj.get_major_constitute(name))

    @staticmethod
    def test_index_highlighter():
        set_global_config('config.json')
        obj = IndexHighlighter()
        obj.summary()


if __name__ == '__main__':
    unittest.main()
