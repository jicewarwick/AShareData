import json
import logging

from AShareData.DBInterface import MySQLInterface, prepare_engine
from AShareData.TushareData import TushareData
from AShareData.WindData import WindData

logging.basicConfig(format='%(asctime)s  %(levelname)s: %(message)s', level=logging.INFO)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_loc = './tests/config.json'
    with open(config_loc, 'r') as f:
        config = json.load(f)

    tushare_token = config['tushare_token']
    engine = prepare_engine(config_loc)
    db_interface = MySQLInterface(engine)

    downloader = TushareData(tushare_token, db_interface=db_interface)
    wind_data = WindData(db_interface)

    # downloader.update_routine()

    # update industry
    # db_reader = DataFrameMySQLWriter(engine)
    # needed_update_provider = []
    # for provider in INDUSTRY_DATA_PROVIDER:
    #     timestamp, _ = db_reader.get_progress(f'{provider}行业')
    #     if timestamp > dt.datetime.now() - dt.timedelta(days=30):
    #         needed_update_provider.append(provider)
    # if needed_update_provider:
    #     wind_data = WindData(engine)
    #     for provider in needed_update_provider:
    #         wind_data.update_industry(provider)

    wind_data.update_minutes_data()
