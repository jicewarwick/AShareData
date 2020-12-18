import pandas as pd
from . import DBInterface, AShareDataReader, DateUtils


class IndustryComparison(object):
    def __init__(self, db_interface: DBInterface):
        self.data_reader = AShareDataReader(db_interface)

    def industry_ratio_comparison(self, holding: pd.Series, date: DateUtils.DateType, index: str):
        pass
