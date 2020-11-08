import importlib.util
import logging

from .AShareDataReader import AShareDataReader
from .DateUtils import TradingCalendar
from .DBInterface import MySQLInterface, prepare_engine
from .FactorCompositor import FactorCompositor
from .TushareData import TushareData
from .WebData import WebDataCrawler

spam_spec = importlib.util.find_spec("WindPy")
if spam_spec is not None:
    from .WindData import WindData

    logging.info('WindPy Found')
else:
    logging.debug('WindPy not found!!')
