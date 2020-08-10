import json
import logging

from AShareData import TushareData
from AShareData import MySQLInterface, prepare_engine, WindData

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    config_loc = './tests/config.json'
    with open(config_loc, 'r') as f:
        config = json.load(f)

    tushare_token = config['tushare_token']
    engine = prepare_engine(config_loc)
    db_interface = MySQLInterface(engine, init=True)

    # downloader = TushareData(tushare_token, db_interface=db_interface, init=True)
    # downloader.update_routine()

    wind_data = WindData(db_interface)
    wind_data.update_routine()
