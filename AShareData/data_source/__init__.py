import importlib.util
import logging

from .DataSource import DataSource
from .JQData import JQData
from .TDXData import TDXData
from .TushareData import TushareData
from .WebData import WebDataCrawler

if importlib.util.find_spec("WindPy"):
    from .WindData import WindData

    logging.getLogger(__name__).info('WindPy found')
else:
    logging.getLogger(__name__).debug('WindPy not found!!')
