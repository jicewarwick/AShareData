import empyrical

import AShareData as asd

if __name__ == '__main__':
    config_loc = './config.json'
    asd.set_global_config(config_loc)
    data_reader = asd.AShareDataReader()
    cal = asd.SHSZTradingCalendar()
    pre_date = cal.offset(cal.today(), -150)

    short_turn_bond_fund = asd.tickers.InvestmentStyleFundTicker(['短期纯债型基金'], otc=True)
    hfq_nav = data_reader.hfq_fund_nav

    tickers = list(set(short_turn_bond_fund.ticker()) & set(short_turn_bond_fund.ticker(pre_date)))
    data = hfq_nav.get_data(ids=tickers)
    data = data.dropna()
    nav_table = data.dropna().unstack()
    nav_table.to_excel('short_turn_bond_fund_adj_nav.xlsx')
    full_period = data.groupby('ID').apply(bond_fund_annual_return)
    cal = asd.SHSZTradingCalendar()
    start_date = cal.offset(cal.today(), -220)
    y1_data = data.loc[data.index.get_level_values('DateTime') >= start_date]
    y1 = y1_data.groupby('ID').agg(annual_ret=bond_fund_annual_return, annual_vol=bond_fund_annual_volatility,
                                   sharpe=bond_fund_sharpe_ratio)
    y1 = y1.sort_values('annual_ret')

    empyrical.annual_return(nav_table.loc[:, '000084.OF'].dropna().pct_change().dropna())

    data = data.loc[data.index.get_level_values('ID').isin(tickers), :]
