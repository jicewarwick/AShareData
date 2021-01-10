import unittest

from AShareData import JQData, set_global_config


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_global_config('config.json')
        self.jq_data = JQData()

    def test_jq_login(self):
        self.jq_data.update_convertable_bond_list()


if __name__ == '__main__':
    unittest.main()
