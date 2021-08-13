import math

import empyrical
import pandas as pd

from AShareData.date_utils import SHSZTradingCalendar

DAYS_IN_YEAR = 365


def annual_return(prices: pd.Series):
    prices = prices.dropna()
    if prices.shape[0] <= 1:
        return 0
    dates = prices.index.get_level_values('DateTime')
    days = (dates[-1].date() - dates[0].date()).days
    pct_change = prices[-1] / prices[0]
    years = days / DAYS_IN_YEAR
    return pow(pct_change, 1 / years) - 1


def annual_volatility(prices: pd.Series):
    dates = prices.index.get_level_values('DateTime')
    cal = SHSZTradingCalendar()
    date_index = cal.select_dates(start_date=dates[0], end_date=dates[-1])
    prices = prices.droplevel('ID').reindex(date_index).interpolate()
    return prices.pct_change().std() * math.sqrt(DAYS_IN_YEAR)


def sharpe_ratio(prices: pd.Series):
    return annual_return(prices) / annual_volatility(prices)


def bond_fund_annual_return(prices: pd.Series, threshold: float = 0.005):
    prices = prices.dropna()
    if prices.shape[0] <= 1:
        return 0
    dates = prices.index.get_level_values('DateTime')
    days = (dates[-1].date() - dates[0].date()).days
    pct = prices.pct_change()
    pct_change = (1 + pct.loc[pct < threshold]).prod()
    years = days / DAYS_IN_YEAR
    return pow(pct_change, 1 / years) - 1


def bond_fund_annual_volatility(prices: pd.Series, threshold: float = 0.005):
    dates = prices.index.get_level_values('DateTime')
    cal = SHSZTradingCalendar()
    date_index = cal.select_dates(start_date=dates[0], end_date=dates[-1])
    prices = prices.droplevel('ID').reindex(date_index).interpolate()
    pct = prices.pct_change()
    std = pct.loc[pct < threshold].std()
    return std * math.sqrt(DAYS_IN_YEAR)


def bond_fund_sharpe_ratio(prices: pd.Series):
    if prices.shape[0] < 20:
        return 0
    return bond_fund_annual_return(prices) / bond_fund_annual_volatility(prices)


def max_drawdown(prices: pd.Series) -> float:
    return empyrical.max_drawdown(prices.pct_change())
