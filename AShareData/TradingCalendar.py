import datetime as dt
from typing import List

from AShareData.DBInterface import DBInterface
from AShareData.utils import date_type2datetime, DateType


class TradingCalendar(object):
    def __init__(self, db_interface: DBInterface):
        calendar_df = db_interface.read_table('交易日历')
        self.calendar = calendar_df['交易日期'].dt.to_pydatetime().tolist()

    def select_dates(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        calendar = self.calendar.copy()
        if start_date:
            start_date = date_type2datetime(start_date)
            calendar = [it for it in calendar if it >= start_date]

        end_date = date_type2datetime(end_date) if end_date else dt.datetime.now()
        return [it for it in calendar if it <= end_date]

    def get_first_day_of_month(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        calendar = self.select_dates(start_date, end_date)
        storage = []
        for i in range(len(calendar) - 1):
            if calendar[i].month != calendar[i + 1].month:
                storage.append(calendar[i + 1])
        return storage

    def get_last_day_of_month(self, start_date: DateType = None, end_date: DateType = None) -> List[dt.datetime]:
        calendar = self.select_dates(start_date, end_date)
        storage = []
        for i in range(len(calendar) - 1):
            if calendar[i].month != calendar[i + 1].month:
                storage.append(calendar[i])
        return storage

    def offset(self, date: DateType, days: int) -> dt.datetime:
        date = date_type2datetime(date)
        return self.calendar[self.calendar.index(date) + days]

    def middle(self, start_date: DateType, end_date: DateType) -> dt.datetime:
        start_date, end_date = date_type2datetime(start_date), date_type2datetime(end_date)
        return self.calendar[int((self.calendar.index(start_date) + self.calendar.index(end_date)) / 2.0)]

    def days_count(self, start_date: DateType, end_date: DateType) -> int:
        start_date, end_date = date_type2datetime(start_date), date_type2datetime(end_date)
        ind = 1 if start_date <= end_date else -1
        return ind * abs(self.calendar.index(end_date) - self.calendar.index(start_date) + 1)
