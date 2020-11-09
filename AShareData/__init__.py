import importlib.util
import logging

from .AShareDataReader import AShareDataReader
from .DateUtils import TradingCalendar
from .DBInterface import MySQLInterface, prepare_engine
from .FactorCompositor import AccountingDateCacheCompositor, ConstLimitStockFactorCompositor, IndexCompositor
from .TushareData import TushareData
from .WebData import WebDataCrawler

wind_spec = importlib.util.find_spec("WindPy")
if wind_spec:
    from .WindData import WindData

    logging.info('WindPy Found')
else:
    logging.debug('WindPy not found!!')
