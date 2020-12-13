import json
import sys

from AShareData import MySQLInterface, prepare_engine, WindData

if __name__ == '__main__':
    config_loc = sys.argv[1]
    with open(config_loc, 'r') as f:
        config = json.load(f)

    engine = prepare_engine(config_loc)
    db_interface = MySQLInterface(engine, init=True)
    self = db_interface

    with WindData(db_interface) as wind_data:
        wind_data.get_stock_rt_price()
