import datetime as dt
from typing import Sequence

import linearmodels
import pandas as pd

from ..AShareDataReader import AShareDataReader
from ..config import get_db_interface
from ..DBInterface import DBInterface
from ..Tickers import StockTickerSelector
from ..utils import StockSelectionPolicy


class EquityModel(object):
    pass


class FamaFrench3(EquityModel):
    def __init__(self, stock_selection_policy: StockSelectionPolicy, db_interface: DBInterface = None):
        super().__init__()
        if db_interface is None:
            db_interface = get_db_interface()
        self.db_interface = db_interface
        self.stock_selector = StockTickerSelector(stock_selection_policy)
        self.data_reader = AShareDataReader(db_interface)

    def run(self, dates: Sequence[dt.datetime]) -> linearmodels.panel.model.FamaMacBethResults:
        log_market_cap = self.data_reader.log_cap.get_data(dates=dates, ticker_selector=self.stock_selector)
        beta = self.data_reader.beta.get_data(dates=dates, ticker_selector=self.stock_selector)
        bm = self.data_reader.bm.get_data(dates=dates, ticker_selector=self.stock_selector)

        forward_ret = self.data_reader.log_return.get_data(dates=dates, ticker_selector=self.stock_selector)
        exog = pd.concat([beta, log_market_cap, bm], axis=1)

        ret_reg = forward_ret.copy()
        ret_reg.index = ret_reg.index.swaplevel()
        exog_reg = exog.copy()
        exog_reg.index = exog_reg.index.swaplevel()
        exog_reg = exog_reg.reindex(ret_reg.index)

        fm = linearmodels.panel.model.FamaMacBeth(dependent=ret_reg, exog=exog_reg)
        return fm.fit()
