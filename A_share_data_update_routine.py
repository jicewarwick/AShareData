import json
import datetime as dt
import logging

from AShareData.TushareData import TushareData
from AShareData.WebData import WebDataCrawler
from AShareData.utils import prepare_engine

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_loc = 'config.json'
    with open(config_loc, 'r') as f:
        config = json.load(f)

    tushare_token = config['tushare_token']

    downloader = TushareData(tushare_token, engine=prepare_engine(config_loc))
    downloader.update_routine()

    web_crawler = WebDataCrawler(prepare_engine(config_loc))
    web_crawler.get_zx_industry(dt.date(2019, 1, 2))
