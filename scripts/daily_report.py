import datetime as dt
import sys

import pandas as pd

import AShareData as asd

if __name__ == '__main__':
    pd.set_option('precision', 2)

    asd.set_global_config(sys.argv[1])
    db_interface = asd.get_db_interface()
    pre_date = dt.datetime.combine(dt.date.today(), dt.time()) - dt.timedelta(days=7)

    data = db_interface.read_table('市场汇总', start_date=pre_date)
    data['换手率'] = data['成交额'] / data['自由流通市值'] * 100
    data.iloc[:, :4] = data.iloc[:, :4] / 1e12
    print('市场成交和估值:')
    print(data)

    print('')
    print('自编指数收益:')
    asd.IndexHighlighter().summary()

    print('')
    print('主要指数估值:')
    print(asd.major_index_valuation())

    print('')
    print('股指贴水情况:')
    print(asd.StockIndexFutureBasis().compute())

    data_reader = asd.AShareDataReader()
    model_factor_ret = data_reader.model_factor_return.bind_params(ids=['FF3_SMB_DD', 'FF3_HML_DD', 'Carhart_UMD_DD'])
    date = dt.datetime.combine(dt.date.today(), dt.time())
    print('')
    print('因子收益率:')
    print(pd.concat([data_reader.market_return.get_data(dates=date), model_factor_ret.get_data(dates=date)]))
