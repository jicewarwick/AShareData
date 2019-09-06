import json
import logging

from AShareData.TushareData import TushareData
from AShareData.WindData import WindData
from AShareData.utils import prepare_engine

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_loc = './tests/config.json'
    with open(config_loc, 'r') as f:
        config = json.load(f)

    tushare_token = config['tushare_token']
    engine = prepare_engine(config_loc)

    downloader = TushareData(tushare_token, engine=engine)
    downloader.update_routine()

    wind_data = WindData(engine)
    wind_data.update_industry('中证')
