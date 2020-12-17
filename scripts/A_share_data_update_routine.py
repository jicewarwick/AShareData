import json
import sys

from AShareData import ConstLimitStockFactorCompositor, MySQLInterface, prepare_engine, TushareData, WindData

if __name__ == '__main__':
    config_loc = sys.argv[1]
    with open(config_loc, 'r') as f:
        config = json.load(f)

    tushare_token = config['tushare_token']
    engine = prepare_engine(config_loc)
    db_interface = MySQLInterface(engine, init=True)

    tushare_crawler = TushareData(tushare_token, db_interface=db_interface)
    tushare_crawler.update_base_info()
    tushare_crawler.get_shibor()

    tushare_crawler.get_ipo_info()
    tushare_crawler.get_company_info()
    tushare_crawler.update_hs_holding()
    tushare_crawler.get_hs_constitute()

    tushare_crawler.update_stock_names()
    tushare_crawler.update_dividend()

    tushare_crawler.update_index_daily()

    tushare_crawler.update_hk_stock_daily()

    tushare_crawler.update_fund_daily()
    tushare_crawler.update_fund_dividend()

    with WindData(db_interface) as wind_data:
        wind_data.update_stock_daily_data()
        wind_data.update_stock_adj_factor()
        wind_data.update_stock_units()
        wind_data.update_industry()
        wind_data.update_pause_stock_info()

        wind_data.update_convertible_bond_daily_data()
        wind_data.update_future_daily_data()
        wind_data.update_stock_option_daily_data()

        wind_data.update_minutes_data()

    ConstLimitStockFactorCompositor(db_interface).update()
