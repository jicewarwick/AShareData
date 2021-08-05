import importlib.util
import logging

from .data_source import DataSource
from .jq_data import JQData
from .tdx_data import TDXData
from .tushare_data import TushareData
from .web_data import WebDataCrawler

if importlib.util.find_spec('WindPy'):
    from .wind_data import WindData

    logging.getLogger(__name__).info('WindPy found')
else:
    logging.getLogger(__name__).debug('WindPy not found!!')
