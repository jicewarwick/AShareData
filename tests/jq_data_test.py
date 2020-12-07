import json
import unittest

from AShareData import JQData, MySQLInterface, prepare_engine


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        with open(config_loc, 'r') as f:
            config = json.load(f)
        config_loc = 'config.json'
        engine = prepare_engine(config_loc)
        db_interface = MySQLInterface(engine)
        self.jq_data = JQData(db_interface, config['jq_mobile'], config['jq_password'])

    def test_jq_login(self):
        self.jq_data.update_convertable_bond_list()


if __name__ == '__main__':
    unittest.main()
