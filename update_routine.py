import json
import logging

from TushareData import TushareData
from utils import prepare_engine

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_loc = 'config.json'
    with open(config_loc, 'r') as f:
        config = json.load(f)

    tushare_token = config['tushare_token']

    tushare_parameters_db = 'param.json'
    db_schema = 'db_schema.json'
    downloader = TushareData(tushare_token, param_json=tushare_parameters_db, db_schema=db_schema,
                             engine=prepare_engine(config_loc))
    downloader.update_routine()
