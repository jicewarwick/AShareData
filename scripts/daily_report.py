import datetime as dt
import sys

import pandas as pd

from AShareData import AShareDataReader, IndexHighlighter, major_index_valuation, set_global_config

if __name__ == '__main__':
    set_global_config(sys.argv[1])
    IndexHighlighter().summary()
    print(major_index_valuation())

    data_reader = AShareDataReader()
    model_factor_ret = data_reader.model_factor_return.bind_params(ids=['FF3_SMB_DD', 'FF3_HML_DD', 'Carhart_UMD_DD'])
    date = dt.datetime.combine(dt.date.today(), dt.time())
    print(pd.concat([data_reader.market_return.get_data(dates=date), model_factor_ret.get_data(dates=date)]))
