import sys

import AShareData as asd
from update_routine import daily_routine

if __name__ == '__main__':
    config_loc = sys.argv[1]
    db_interface = asd.generate_db_interface_from_config(config_loc, init=True)
    asd.set_global_config(config_loc)

    with asd.TushareData() as tushare_data:
        tushare_data.init_db()
        tushare_data.init_accounting_data()

    daily_routine(config_loc)

    asd.model.SMBandHMLCompositor(asd.FamaFrench3FactorModel()).update()
    asd.model.UMDCompositor(asd.FamaFrenchCarhart4FactorModel()).update()
