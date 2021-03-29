import sys

from AShareData import generate_db_interface_from_config, set_global_config, TushareData
from AShareData.model import FamaFrench3FactorModel, FamaFrenchCarhart4FactorModel, SMBandHMLCompositor, UMDCompositor
from update_routine import daily_routine

if __name__ == '__main__':
    config_loc = sys.argv[1]
    db_interface = generate_db_interface_from_config(config_loc, init=True)
    set_global_config(config_loc)

    with TushareData() as tushare_data:
        tushare_data.init_db()
        tushare_data.init_accounting_data()

    daily_routine(config_loc)

    SMBandHMLCompositor(FamaFrench3FactorModel()).update()
    UMDCompositor(FamaFrenchCarhart4FactorModel()).update()
