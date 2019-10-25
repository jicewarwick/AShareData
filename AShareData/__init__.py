import importlib
import logging

from .AShareDataReader import AShareDataReader
from .DBInterface import DBInterface, MySQLInterface, prepare_engine
from .TradingCalendar import TradingCalendar
from .TushareData import TushareData
from .WebData import WebDataCrawler

spam_spec = importlib.util.find_spec("WindPy")
windpy_installed = spam_spec is not None

if windpy_installed:
    from .WindData import WindData
    logging.info('WindPy Found')
else:
    logging.debug('WindPy not found!!')
