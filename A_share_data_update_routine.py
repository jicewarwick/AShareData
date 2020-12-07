import json

from AShareData import TushareData
from AShareData import MySQLInterface, prepare_engine, WindData

if __name__ == '__main__':
    config_loc = './tests/config.json'
    with open(config_loc, 'r') as f:
        config = json.load(f)

    tushare_token = config['tushare_token']
    engine = prepare_engine(config_loc)
    db_interface = MySQLInterface(engine, init=True)

    tushare_crawler = TushareData(tushare_token, db_interface=db_interface)
    tushare_crawler.update_base_info()
    # tushare_crawler.update_dividend()
    tushare_crawler.get_shibor()
    tushare_crawler.update_fund_dividend()
    # downloader.update_fund_daily()
    # downloader.update_routine()

