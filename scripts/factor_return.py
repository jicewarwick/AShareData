import datetime as dt
import sys

import AShareData as asd
from AShareData.factor_compositor import FactorPortfolio, FactorPortfolioPolicy
from AShareData.utils import StockSelectionPolicy

if __name__ == '__main__':
    asd.set_global_config(sys.argv[1])

    data_reader = asd.AShareDataReader()
    stock_selection_policy = StockSelectionPolicy()
    stock_selection_policy.ignore_new_stock_period = 244
    stock_selection_policy.ignore_st = True
    stock_selection_policy.ignore_pause = True

    policy = FactorPortfolioPolicy()
    policy.bins = [5, 10]
    policy.stock_selection_policy = stock_selection_policy
    policy.start_date = dt.datetime(2010, 1, 1)
    policy.industry = data_reader.industry('申万', 1)
    policy.weight = data_reader.stock_free_floating_market_cap

    policy.name = data_reader.beta.name
    policy.factor = data_reader.beta

    sub_port = FactorPortfolio(factor_portfolio_policy=policy)
    sub_port.update()
