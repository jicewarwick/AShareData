STOCK_EXCHANGES = ['SSE', 'SZSE']
FUTURE_EXCHANGES = ['CFFEX', 'DCE', 'CZCE', 'SHFE', 'INE']
ALL_EXCHANGES = STOCK_EXCHANGES + FUTURE_EXCHANGES
STOCK_INDEXES = {'上证指数': '000001.SH', '深证成指': '399001.SZ', '中小板指': '399005.SZ', '创业板指': '399006.SZ',
                 '上证50': '000016.SH', '沪深300': '000300.SH', '中证500': '000905.SH'}
BOARD_INDEXES = ['000016.SH', '399300.SH', '000905.SH']

TRADING_DAYS_IN_YEAR = 252

INDUSTRY_DATA_PROVIDER = ['中信', '申万', '中证', 'Wind']
INDUSTRY_DATA_PROVIDER_CODE_DICT = {'中信': 'citic', '申万': 'sw', '中证': 'csi', 'Wind': 'gics'}
INDUSTRY_LEVEL = {'中信': 3, '申万': 3, '中证': 4, 'Wind': 4}
INDUSTRY_START_DATE = {'中信': '20030103', '申万': '20050527', '中证': '20161212', 'Wind': '20050105'}
