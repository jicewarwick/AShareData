import json
from Tushare2MySQL import Tushare2MySQL

if __name__ == '__main__':
    config_loc = 'config.json'
    with open(config_loc, 'r') as f:
        config = json.load(f)

    tushare_token = config['tushare_token']
    ip, port = config['ip'], config['port']
    username, password = config['username'], config['password']

    tushare_parameters_db = 'param.json'
    downloader = Tushare2MySQL(tushare_token, param_json=tushare_parameters_db)
    downloader.add_mysql_db(ip, port, username, password)
    # downloader.get_daily_hq(start_date='20080103', end_date='20181231')
    # downloader.update_index_daily()
    # downloader.get_all_stocks()
    # downloader.get_calendar()
    # downloader.get_all_past_names()
    # downloader.get_company_info()

