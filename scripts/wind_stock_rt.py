import sys

from AShareData import WindData, set_global_config

if __name__ == '__main__':
    config_loc = sys.argv[1]
    set_global_config(config_loc)

    with WindData() as wind_data:
        wind_data.get_stock_rt_price()
