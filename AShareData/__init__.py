import importlib.util
import logging

from .analysis.holding import IndustryComparison
from .analysis.trading import TradingAnalysis
from .AShareDataReader import AShareDataReader
from .config import get_db_interface, get_global_config, set_global_config
from .data_source.TushareData import TushareData
from .data_source.WebData import WebDataCrawler
from .DateUtils import TradingCalendar
from .DBInterface import MySQLInterface
from .FactorCompositor import ConstLimitStockFactorCompositor, IndexCompositor

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))

logger = logging.getLogger(__name__)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

wind_spec = importlib.util.find_spec("WindPy")
if wind_spec:
    from .data_source.WindData import WindData

    logging.getLogger(__name__).info('WindPy found')
else:
    logging.getLogger(__name__).debug('WindPy not found!!')

jq_spec = importlib.util.find_spec("jqdatasdk")
if jq_spec:
    from .data_source.JQData import JQData

    logging.getLogger(__name__).info('JQData found')
else:
    logging.getLogger(__name__).debug('JQData not found!!')
