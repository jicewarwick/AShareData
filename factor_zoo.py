import json
import QuantStudio.api as QS


if __name__ == '__main__':
    config_loc = 'config.json'
    with open(config_loc, 'r') as f:
        config = json.load(f)

    config = {'数据库类型': 'MySQL',
              '数据库名': config['db_name'],
              'IP地址': config['ip'],
              '端口': config['port'],
              '用户名': config['username'],
              '密码': config['password'],
              '表名前缀': '',
              '字符集': 'utf8',
              '连接器': 'default'
              }
    sql_db = QS.FactorDB.SQLDB(config)
    sql_db._Prefix = ''
    sql_db.connect()

    price_table = sql_db.getTable('股票日行情')
    dts = price_table.getDateTime()
    secs = price_table.getID()
    # a = price_table.readData(['收盘价'], ['000001.SZ'], dts)
    close = price_table.getFactor('收盘价')
    adj_factor = price_table.getFactor('复权因子')
    hfq_close = QS.FactorDB.Factorize(close * adj_factor, '后复权收盘价')
    hfq_close.readData(secs, dts)
