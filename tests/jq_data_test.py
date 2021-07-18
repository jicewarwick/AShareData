import unittest

import AShareData as asd


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        asd.set_global_config('config.json')
        self.jq_data = asd.JQData()

    def test_jq_login(self):
        pass


if __name__ == '__main__':
    unittest.main()
