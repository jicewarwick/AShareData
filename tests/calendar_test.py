import unittest

from AShareData.DateUtils import *
from AShareData.DBInterface import MySQLInterface, prepare_engine


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_loc = 'config.json'
        engine = prepare_engine(config_loc)
        self.calendar = TradingCalendar(MySQLInterface(engine))

    def test_is_trading_day(self):
        self.assertFalse(self.calendar.is_trading_date(dt.date(2019, 10, 1)))
        self.assertTrue(self.calendar.is_trading_date(dt.date(2019, 10, 16)))

    def test_days_count(self):
        start = dt.datetime(2019, 1, 4)
        end = dt.datetime(2019, 1, 7)
        self.assertEqual(self.calendar.days_count(start, end), 1)
        self.assertEqual(self.calendar.days_count(end, start), -1)
        self.assertEqual(self.calendar.days_count(start, start), 0)

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

    @staticmethod
    def test_format_dt():
        @format_input_dates
        def func(date, dates=None):
            print(date)
            print(dates)

        func(dt.date(2000, 1, 1), dates=dt.date(2010, 1, 1))


if __name__ == '__main__':
    unittest.main()
