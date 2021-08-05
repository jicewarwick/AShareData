import importlib.util
import logging

from .analysis import IndustryComparison, TradingAnalysis
from .ashare_data_reader import AShareDataReader
from .config import generate_db_interface_from_config, get_db_interface, get_global_config, set_global_config
from .data_source import JQData, TDXData, TushareData, WebDataCrawler
from .database_interface import MySQLInterface
from .date_utils import SHSZTradingCalendar
from .factor_compositor import ConstLimitStockFactorCompositor, IndexCompositor, IndexUpdater, MarketSummaryCompositor, \
    NegativeBookEquityListingCompositor
from .model import FamaFrench3FactorModel, FamaFrenchCarhart4FactorModel
from .tools import IndexHighlighter, major_index_valuation, MajorIndustryConstitutes, StockIndexFutureBasis

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))

logger = logging.getLogger(__name__)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

if importlib.util.find_spec('WindPy'):
    from .data_source import WindData
