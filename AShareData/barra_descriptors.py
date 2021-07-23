import pandas as pd


class BarraDescriptor(object):
    def __init__(self, factor_zoo):
        self.factor_zoo = factor_zoo

    def nature_log_of_market_cap(self):
        return self.factor_zoo.stock_market_cap.log()

    def beta(self, window: int, half_life: int):
        y = self.factor_zoo.excess_return()
        pass

    def relative_strength(self, window: int = 504, lag: int = 21, half_life: int = 126) -> pd.DataFrame:
        tmp = self.factor_zoo.log_return().sub(self.factor_zoo.log_shibor_return(), axis=0)
        exp_weight = self.factor_zoo.exponential_weight(window, half_life)
        tmp2 = tmp * exp_weight
        return tmp2.rolling(window, min_periods=window).sum().shift(lag)

    def daily_standard_deviation(self):
        pass

    def cumulative_range(self):
        pass

    def historical_sigma(self):
        pass

    def cube_of_size(self):
        pass

    def book_to_price_ratio(self):
        pass

    def share_turnover_one_month(self):
        pass

    def average_share_turnover_trailing_3_month(self):
        pass

    def average_share_turnover_trailing_12_months(self):
        pass

    def predicted_earning_to_price_ratio(self):
        pass

    def cash_earning_to_price_ratio(self):
        pass

    def trailing_earnings_to_price_ratio(self):
        pass

    def long_term_predicted_earning_growth(self):
        pass

    def short_term_predicted_earning_growth(self):
        pass

    def earnings_growth_trailing_5_years(self):
        pass

    def sales_growth_trailing_5_years(self):
        pass

    def market_leverage(self):
        pass

    def debt_to_assets(self):
        pass

    def book_leverage(self):
        pass
