import importlib.util
import logging

from .AShareDataReader import AShareDataReader
from .DateUtils import TradingCalendar
from .DBInterface import MySQLInterface, prepare_engine
from .FactorCompositor import AccountingDateCacheCompositor, ConstLimitStockFactorCompositor, IndexCompositor
from .TushareData import TushareData
from .WebData import WebDataCrawler

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))

logger = logging.getLogger(__name__)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

wind_spec = importlib.util.find_spec("WindPy")
if wind_spec:
    from .WindData import WindData
    logging.getLogger(__name__).info('WindPy found')
else:
    logging.getLogger(__name__).debug('WindPy not found!!')

jq_spec = importlib.util.find_spec("jqdatasdk")
if jq_spec:
    from .JQData import JQData
    logging.getLogger(__name__).info('JQData found')
else:
    logging.getLogger(__name__).debug('JQData not found!!')
