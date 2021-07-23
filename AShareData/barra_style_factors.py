from .barra_descriptors import BarraDescriptor


class BarraStyleFactors(object):
    def __init__(self, descriptors: BarraDescriptor):
        self.descriptors = descriptors

    def size(self):
        return self.descriptors.nature_log_of_market_cap()

    def beta(self):
        window = 252
        half_life = 63
        return self.descriptors.beta(window, half_life)

    def momentum(self):
        return self.descriptors.relative_strength()

    def residual_volatility(self):
        return 0.74 * self.descriptors.daily_standard_deviation() + \
               0.16 * self.descriptors.cumulative_range() + \
               0.1 * self.descriptors.historical_sigma()

    def non_linear_size(self):
        return self.descriptors.cube_of_size()

    def book_to_price(self):
        return self.descriptors.book_to_price_ratio()

    def liquidity(self):
        return 0.35 * self.descriptors.share_turnover_one_month() + \
               0.35 * self.descriptors.average_share_turnover_trailing_3_month() + \
               0.3 * self.descriptors.average_share_turnover_trailing_12_months()

    def earning_yield(self):
        return 0.68 * self.descriptors.predicted_earning_to_price_ratio() + \
               0.21 * self.descriptors.cash_earning_to_price_ratio() + \
               0.11 * self.descriptors.trailing_earnings_to_price_ratio()

    def growth(self):
        return 0.18 * self.descriptors.long_term_predicted_earning_growth() + \
               0.11 * self.descriptors.short_term_predicted_earning_growth() + \
               0.24 * self.descriptors.earnings_growth_trailing_5_years() + \
               0.47 * self.descriptors.sales_growth_trailing_5_years()

    def leverage(self):
        return 0.38 * self.descriptors.market_leverage() + \
               0.35 * self.descriptors.debt_to_assets() + \
               0.27 * self.descriptors.book_leverage()
