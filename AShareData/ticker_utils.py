from typing import Optional, Tuple, Union

from .config import get_db_interface
from .database_interface import DBInterface


class TickerFormatter(object):
    pass


class StockTickerFormatter(TickerFormatter):
    @staticmethod
    def stock_num2ticker(stock_num: Union[str, int]) -> str:
        if isinstance(stock_num, str):
            stock_num = int(stock_num)
        if stock_num < 600000:
            return f'{stock_num:06d}.SZ'
        else:
            return f'{stock_num:06d}.SH'

    @staticmethod
    def ticker2stock_num(ts_code: str) -> str:
        return ts_code.split()[0]


class FutureTickerFormatter(TickerFormatter):
    @staticmethod
    def format_czc_ticker(ticker: str) -> str:
        c = ticker[1] if ticker[1].isnumeric() else ticker[2]
        ticker = ticker.replace(c, '', 1)
        return ticker

    @staticmethod
    def full_czc_ticker(ticker: str) -> str:
        c = 1 if ticker[1].isnumeric() else 2
        ticker = ticker[:c] + '2' + ticker[c:]
        return ticker


class FundTickerFormatter(TickerFormatter):
    exchange_fund_code_cache = {}

    def __init__(self, db_interface: DBInterface = None):
        if db_interface is None:
            db_interface = get_db_interface()
        self.db_interface = db_interface

    @staticmethod
    def exchange_ticker_to_of_ticker(ticker: str) -> str:
        if not (ticker.endswith('.SH') | ticker.endswith('SZ')):
            raise ValueError(f'Invalid exchange ticker: {ticker}')
        return ticker[:-3] + '.OF'

    def of_ticker_to_exchange_ticker(self, ticker: str):
        if not ticker.endswith('.OF'):
            raise ValueError(f'Invalid OTC fund ticker: {ticker}')
        self._init_exchange_code_cache()
        return self.exchange_fund_code_cache.get(ticker, None)

    def _init_exchange_code_cache(self):
        if len(self.exchange_fund_code_cache) == 0:
            ids = self.db_interface.get_all_id('基金列表')
            for ticker in ids:
                if not ticker.endswith('OF'):
                    self.exchange_fund_code_cache[ticker[:6] + '.OF'] = ticker


def split_stock_ticker(ticker: str) -> Optional[Tuple[int, str]]:
    try:
        ticker_num, market = ticker.split('.')
    except (ValueError, AttributeError):
        return None
    if market not in ['SH', 'SZ']:
        return None
    try:
        ticker_num = int(ticker_num)
    except ValueError:
        return None
    return ticker_num, market


def is_main_board_stock(ticker: str) -> bool:
    """判断是否为沪深主板股票

    :param ticker: 股票代码, 如 `000001.SZ`
    """
    return get_stock_board_name(ticker) == '主板'


def get_stock_board_name(ticker: str) -> str:
    """获取股票所在版块(主板, 中小板, 创业板, 科创板), 其他返回 `非股票`

        :param ticker: 股票代码, 如 `000001.SZ`
    """
    ret = split_stock_ticker(ticker)
    if ret is None:
        return '非股票'
    ticker_num, market = ret
    if (0 < ticker_num < 2000 and market == 'SZ') or (600000 <= ticker_num < 606000 and market == 'SH'):
        return '主板'
    elif 2000 < ticker_num < 4000 and market == 'SZ':
        return '中小板'
    elif 300000 < ticker_num < 301000 and market == 'SZ':
        return '创业板'
    elif 688000 < ticker_num < 690000 and market == 'SH':
        return '科创板'
    else:
        return '非股票'
