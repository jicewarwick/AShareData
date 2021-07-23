import unittest

from AShareData.config import set_global_config
from AShareData.date_utils import *


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_global_config('config.json')
        self.calendar = SHSZTradingCalendar()

    def test_is_trading_day(self):
        self.assertFalse(self.calendar.is_trading_date(dt.date(2019, 10, 1)))
        self.assertTrue(self.calendar.is_trading_date(dt.date(2019, 10, 16)))

    def test_days_count(self):
        start = dt.datetime(2019, 1, 4)
        end = dt.datetime(2019, 1, 7)
        self.assertEqual(self.calendar.days_count(start, end), 1)
        self.assertEqual(self.calendar.days_count(end, start), -1)
        self.assertEqual(self.calendar.days_count(start, start), 0)

        self.assertEqual(self.calendar.days_count(dt.datetime(2015, 9, 30), dt.datetime(2015, 10, 8)), 1)
        self.assertEqual(self.calendar.days_count(dt.datetime(2015, 10, 1), dt.datetime(2015, 10, 8)), 1)

    def test_first_day_of_month(self):
        start = dt.datetime(2019, 3, 2)
        end = dt.datetime(2019, 4, 2)
        self.assertEqual(self.calendar.first_day_of_month(start, end)[0], dt.datetime(2019, 4, 1))

    def test_last_day_of_month(self):
        start = dt.datetime(2019, 3, 2)
        end = dt.datetime(2019, 4, 2)
        self.assertEqual(self.calendar.last_day_of_month(start, end)[0], dt.datetime(2019, 3, 29))

    def test_last_day_of_year(self):
        start = dt.datetime(2018, 3, 2)
        end = dt.datetime(2019, 4, 2)
        self.assertEqual(self.calendar.last_day_of_year(start, end)[0], dt.datetime(2018, 12, 28))

    def test_select_dates(self):
        start = dt.datetime(2019, 9, 2)
        end = dt.datetime(2019, 9, 3)
        self.assertEqual(self.calendar.select_dates(start, end), [start, end])

        start = dt.datetime(2020, 11, 2)
        end = dt.datetime(2020, 11, 7)
        dates = self.calendar.select_dates(start, end)
        self.assertEqual(dates[0], start)
        self.assertEqual(dates[-1], dt.datetime(2020, 11, 6))

        start = dt.datetime(2020, 11, 1)
        end = dt.datetime(2020, 11, 6)
        dates = self.calendar.select_dates(start, end)
        self.assertEqual(dates[0], dt.datetime(2020, 11, 2))
        self.assertEqual(dates[-1], dt.datetime(2020, 11, 6))

        start = dt.datetime(2020, 11, 1)
        end = dt.datetime(2020, 11, 7)
        dates = self.calendar.select_dates(start, end)
        self.assertEqual(dates[0], dt.datetime(2020, 11, 2))
        self.assertEqual(dates[-1], dt.datetime(2020, 11, 6))

    def test_offset(self):
        start_date = dt.datetime(2020, 11, 2)
        self.assertEqual(self.calendar.offset(start_date, 1), dt.datetime(2020, 11, 3))
        self.assertEqual(self.calendar.offset(start_date, 0), dt.datetime(2020, 11, 2))
        self.assertEqual(self.calendar.offset(start_date, -1), dt.datetime(2020, 10, 30))

        start_date = dt.datetime(2020, 11, 1)
        self.assertEqual(self.calendar.offset(start_date, 1), dt.datetime(2020, 11, 2))
        self.assertEqual(self.calendar.offset(start_date, 0), dt.datetime(2020, 11, 2))
        self.assertEqual(self.calendar.offset(start_date, -1), dt.datetime(2020, 10, 30))

    def test_begin_and_end(self):
        self.assertEqual(self.calendar.month_begin(2021, 3), dt.datetime(2021, 3, 1))
        self.assertEqual(self.calendar.month_begin(2021, 1), dt.datetime(2021, 1, 4))
        self.assertEqual(self.calendar.month_begin(2020, 10), dt.datetime(2020, 10, 9))

        self.assertEqual(self.calendar.month_end(2021, 3), dt.datetime(2021, 3, 31))
        self.assertEqual(self.calendar.month_end(2021, 1), dt.datetime(2021, 1, 29))
        self.assertEqual(self.calendar.month_end(2020, 1), dt.datetime(2020, 1, 23))

        self.assertEqual(self.calendar.pre_month_end(2021, 4), dt.datetime(2021, 3, 31))
        self.assertEqual(self.calendar.pre_month_end(2021, 2), dt.datetime(2021, 1, 29))
        self.assertEqual(self.calendar.pre_month_end(2020, 2), dt.datetime(2020, 1, 23))

    @staticmethod
    def test_format_dt():
        @dtlize_input_dates
        def func(date, dates=None):
            print(date)
            print(dates)

        func(dt.date(2000, 1, 1), dates=dt.date(2010, 1, 1))

    def test_report_date_offset(self):
        self.assertEqual(ReportingDate.quarterly_offset(dt.datetime(2020, 3, 31), -1), dt.datetime(2019, 12, 31))
        self.assertEqual(ReportingDate.quarterly_offset(dt.datetime(2020, 3, 31), -2), dt.datetime(2019, 9, 30))
        self.assertEqual(ReportingDate.quarterly_offset(dt.datetime(2020, 3, 31), -3), dt.datetime(2019, 6, 30))
        self.assertEqual(ReportingDate.quarterly_offset(dt.datetime(2020, 3, 31), -4), dt.datetime(2019, 3, 31))
        self.assertEqual(ReportingDate.quarterly_offset(dt.datetime(2020, 3, 31), -5), dt.datetime(2018, 12, 31))

        self.assertEqual(ReportingDate.offset(dt.datetime(2020, 3, 31), 'q1'), dt.datetime(2019, 12, 31))
        self.assertEqual(ReportingDate.offset(dt.datetime(2020, 3, 31), 'y1'), dt.datetime(2019, 12, 31))


if __name__ == '__main__':
    unittest.main()
