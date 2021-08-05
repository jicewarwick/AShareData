import sys

import AShareData as asd


def daily_routine(config_loc: str):
    asd.set_global_config(config_loc)

    with asd.TushareData() as tushare_crawler:
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
        tushare_crawler.update_financial_data()

    with asd.WindData() as wind_data:
        wind_data.update_stock_daily_data()
        wind_data.update_stock_adj_factor()
        wind_data.update_stock_units()
        wind_data.update_industry()
        wind_data.update_pause_stock_info()

        wind_data.update_convertible_bond_daily_data()
        wind_data.update_cb_convertible_price()
        wind_data.update_future_daily_data()
        wind_data.update_fund_info()
        wind_data.update_stock_option_daily_data()

    with asd.JQData() as jq_data:
        jq_data.update_stock_morning_auction_data()

    with asd.TDXData() as tdx_data:
        # tdx_data.update_stock_minute()
        tdx_data.update_convertible_bond_minute()

    # compute data
    asd.ConstLimitStockFactorCompositor().update()
    asd.NegativeBookEquityListingCompositor().update()
    asd.IndexUpdater().update()
    asd.MarketSummaryCompositor().update()

    # model data
    asd.model.SMBandHMLCompositor().update()
    asd.model.UMDCompositor().update()


if __name__ == '__main__':
    daily_routine(sys.argv[1])
