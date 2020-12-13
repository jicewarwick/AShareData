import datetime as dt
import json
import sys

from DingTalkMessageBot import DingTalkMessageBot

from AShareData import JQData, MySQLInterface, prepare_engine, TushareData

if __name__ == '__main__':
    config_loc = sys.argv[1]
    date = dt.date.today()
    with open(config_loc, 'r') as f:
        config = json.load(f)
    engine = prepare_engine(config_loc)
    db_interface = MySQLInterface(engine)

    tushare_crawler = TushareData(config['tushare_token'], db_interface=db_interface)
    tushare_crawler.get_ipo_info()

    token = '076314e3a582ce36705d53dc0b822493c046447b82e3cfb1571850debe500b15'
    secret = 'SEC2931f95f8d27145c579b8f37eb479fc587a3caec441829b5a4999b90271286c4'
    messenger = DingTalkMessageBot(token, secret)
    try:
        with JQData(db_interface, config['jq_mobile'], config['jq_password']) as jq_data:
            jq_data.stock_open_auction_data(date)
            messenger.send_message(f'{date} 集合竞价数据已下载.')
    except:
        messenger.send_message(f'{date} 集合竞价数据下载失败.')
