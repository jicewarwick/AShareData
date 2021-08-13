import unittest

from AShareData import set_global_config
from AShareData.factor import *
from AShareData.tickers import *
from AShareData.utils import StockSelectionPolicy


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_global_config('config.json')
        self.db_interface = get_db_interface()
        self.calendar = SHSZTradingCalendar()
        self.start_date = dt.datetime(2002, 3, 1)
        self.end_date = dt.datetime(2002, 3, 30)
        self.ids = ['000001.SZ', '000002.SZ']
        self.close = ContinuousFactor('股票日行情', '收盘价', self.db_interface)
        self.adj = CompactFactor('复权因子', self.db_interface)

    def test_compact_record_factor(self):
        compact_factor = CompactFactor('证券名称', self.db_interface)
        compact_factor.data = compact_factor.data.map(lambda x: 'PT' in x or 'ST' in x or '退' in x)
        compact_record_factor = CompactRecordFactor(compact_factor, 'ST')
        print(compact_record_factor.get_data(date=dt.datetime(2015, 5, 15)))

    def test_compact_factor(self):
        compact_factor = CompactFactor('证券名称', self.db_interface)
        print(compact_factor.get_data(dates=dt.datetime(2015, 5, 15)))
        policy = StockSelectionPolicy(select_st=True)
        print(compact_factor.get_data(dates=dt.datetime(2015, 5, 15), ticker_selector=StockTickerSelector(policy)))

    def test_industry(self):
        print('')
        industry_factor = IndustryFactor('中信', 3, self.db_interface)
        print(industry_factor.list_constitutes(dt.datetime(2019, 1, 7), '白酒'))
        print('')
        print(industry_factor.all_industries)

    def test_pause_stocks(self):
        pause_stock = OnTheRecordFactor('股票停牌', self.db_interface)
        start_date = dt.datetime(2021, 1, 1)
        end_date = dt.datetime(2021, 2, 4)
        print(pause_stock.get_data(date=end_date))
        print(pause_stock.get_counts(start_date=start_date, end_date=end_date, ids=self.ids + ['000662.SZ']))

    def test_latest_accounting_factor(self):
        f = LatestAccountingFactor('期末总股本', self.db_interface)
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(a)

    def test_latest_quarter_report_factor(self):
        f = LatestQuarterAccountingFactor('期末总股本', self.db_interface)
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(a)

    def test_yearly_report_factor(self):
        f = YearlyReportAccountingFactor('期末总股本', self.db_interface)
        ids = list(set(self.ids) - {'600087.SH', '600788.SH', '600722.SH'})
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=ids)
        print(a)

    def test_qoq_report_factor(self):
        f = QOQAccountingFactor('期末总股本', self.db_interface)
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(a)

    def test_yoy_period_report_factor(self):
        f = YOYPeriodAccountingFactor('期末总股本', self.db_interface)
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(a)

    def test_yoy_quarter_factor(self):
        f = YOYQuarterAccountingFactor('期末总股本', self.db_interface)
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(a)

    def test_ttm_factor(self):
        f = TTMAccountingFactor('期末总股本', self.db_interface)
        a = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(a)

    def test_index_constitute(self):
        index_constitute = IndexConstitute(self.db_interface)
        print(index_constitute.get_data('000300.SH', '20200803'))

    def test_sum_factor(self):
        sum_hfq = self.close + self.adj
        sum_hfq_close_data = sum_hfq.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(sum_hfq_close_data)
        uni_sum = self.close + 1
        print(uni_sum.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids))

    def test_mul_factor(self):
        hfq = self.close * self.adj
        hfq_close_data = hfq.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(hfq_close_data)

    def test_factor_pct_change(self):
        hfq = self.close * self.adj
        hfq_chg = hfq.pct_change()
        pct_chg_data = hfq_chg.get_data(start_date=self.start_date, end_date=self.end_date)
        print(pct_chg_data)

    def test_factor_max(self):
        f = self.adj.max()
        f_max = f.get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids)
        print(f_max)

    def test_beta_factor(self):
        ids: Union[str, Sequence[str]] = ['000001.SZ', '600000.SH']
        dates: Sequence[dt.datetime] = [dt.datetime(2020, 1, 15), dt.datetime(2020, 5, 13)]
        look_back_period: int = 60
        min_trading_days: int = 40

        policy = StockSelectionPolicy(ignore_new_stock_period=365, ignore_st=True)
        ticker_selector = StockTickerSelector(policy)

        beta_factor = BetaFactor(db_interface=self.db_interface)
        print(beta_factor.get_data(dates, ids, look_back_period=look_back_period, min_trading_days=min_trading_days))
        print(beta_factor.get_data(dates, ticker_selector=ticker_selector, look_back_period=look_back_period,
                                   min_trading_days=min_trading_days))

    def test_interest_rate(self):
        print('')
        interest_rate = InterestRateFactor('shibor利率数据', '6个月', self.db_interface).set_factor_name('6个月shibor')
        start_date = dt.datetime(2021, 1, 1)
        end_date = dt.datetime(2021, 3, 1)
        data = interest_rate.get_data(start_date=start_date, end_date=end_date)
        print(data)

    def test_mean_and_average(self):
        print(self.close.mean('DateTime').get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids))
        print(self.close.mean('ID').get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids))
        print(self.close.sum('DateTime').get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids))
        print(self.close.sum('ID').get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids))

    def test_diff(self):
        print(self.close.diff().get_data(start_date=self.start_date, end_date=self.end_date, ids=self.ids))

    def test_latest_update_factor(self):
        latest_update_factor = LatestUpdateFactor('场外基金规模', '资产净值', self.db_interface)
        print(latest_update_factor.get_data(ids=['008864.OF', '000001.OF']))
        print(latest_update_factor.get_data(ids='008864.OF'))


if __name__ == '__main__':
    unittest.main()
