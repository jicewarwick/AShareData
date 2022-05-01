import importlib.util
import logging

from .analysis import IndustryComparison, TradingAnalysis
from .config import generate_db_interface_from_config, get_db_interface, get_global_config, set_global_config
from .data_reader import (ConvertibleBondDataReader, DataReader, ExchangeFundDataReader, FutureDataReader,
                          IndexDataReader, ModelDataReader, OTCFundDataReader, SHIBORDataReader, StockDataReader)
from .data_source import EastMoneyCrawler, JQData, TDXData, TushareData, WebDataCrawler
from .database_interface import DBInterface, MySQLInterface
from .date_utils import SHSZTradingCalendar
from .factor_compositor import (ConstLimitStockFactorCompositor, FundAdjFactorCompositor, IndexCompositor, IndexUpdater,
                                MarketSummaryCompositor, NegativeBookEquityListingCompositor)
from .factor_portfolio import FactorPortfolio
from .model import FamaFrench3FactorModel, FamaFrenchCarhart4FactorModel
from .tools import IndexHighlighter, MajorIndustryConstitutes, StockIndexFutureBasis, major_index_valuation

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))

logger = logging.getLogger(__name__)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

if importlib.util.find_spec('WindPy'):
    from .data_source import WindData
