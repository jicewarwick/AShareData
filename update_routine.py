import json
import logging

from TushareData import TushareData

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_loc = 'config.json'
    with open(config_loc, 'r') as f:
        config = json.load(f)

    tushare_token = config['tushare_token']
    ip, port, db_name = config['ip'], config['port'], config['db_name']
    username, password = config['username'], config['password']

    tushare_parameters_db = 'param.json'
    downloader = TushareData(tushare_token, param_json=tushare_parameters_db)
    downloader.add_mysql_db(ip, port, username, password, db_name=db_name)
    downloader.update_routine()
