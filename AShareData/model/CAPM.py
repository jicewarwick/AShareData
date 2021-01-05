from dataclasses import dataclass
from typing import Sequence
import datetime as dt

# import statsmodels.api as sm

from ..AShareDataReader import AShareDataReader
from ..config import get_db_interface
from ..DBInterface import DBInterface
from ..utils import StockSelectionPolicy


@dataclass
class CAPMParams(object):
    adjust_dates: Sequence[dt.datetime]
    start_date: dt.datetime
    end_date: dt.datetime
    benchmark: str
    stock_selection_policy: StockSelectionPolicy


class CAPMModel(object):
    def __init__(self, db_interface: DBInterface = None):
        if not db_interface:
            db_interface = get_db_interface()
        self.data_reader = AShareDataReader(db_interface)

    def check_model(self, params: CAPMParams):
        pass
