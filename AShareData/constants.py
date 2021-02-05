import datetime as dt

TRADING_DAYS_IN_YEAR = 252

# exchanges
STOCK_EXCHANGES = ['SSE', 'SZSE']
FUTURE_EXCHANGES = ['CFFEX', 'DCE', 'CZCE', 'SHFE', 'INE']
ALL_EXCHANGES = STOCK_EXCHANGES + FUTURE_EXCHANGES

# indexes
STOCK_INDEXES = {'上证指数': '000001.SH', '深证成指': '399001.SZ', '中小板指': '399005.SZ', '创业板指': '399006.SZ',
                 '上证50': '000016.SH', '沪深300': '000300.SH', '中证500': '000905.SH'}
BOARD_INDEXES = ['000016.SH', '000300.SH', '000905.SH']
STOCK_INDEX_ETFS = {'中小板': '159902.SZ', '创业板': '159915.SZ', '50ETF': '510050.SH', '300ETF': '510300.SH',
                    '500ETF': '510500.SH'}

# financial statements
FINANCIAL_STATEMENTS_TYPE = ['资产负债表', '利润表', '现金流量表']

# industry constants
INDUSTRY_DATA_PROVIDER = ['中信', '申万', '中证', 'Wind']
INDUSTRY_DATA_PROVIDER_CODE_DICT = {'中信': 'citic', '申万': 'sw', '中证': 'csi', 'Wind': 'gics'}
INDUSTRY_LEVEL = {'中信': 3, '申万': 3, '中证': 4, 'Wind': 4}
INDUSTRY_START_DATE = {'中信': dt.datetime(2003, 1, 2), '申万': dt.datetime(2005, 5, 27), '中证': dt.datetime(2016, 12, 12),
                       'Wind': dt.datetime(2005, 1, 5)}

# market return index
MARKET_RETURN_INDEXES = {'000001.IND': {'name': '市场全收益指数',
                                        'unit_base': '自由流通股本',
                                        'industry_provider': None,
                                        'industry_level': None,
                                        'industry': None,
                                        'ignore_new_stock_period': 365,
                                        'ignore_st': True,
                                        'ignore_pause': True,
                                        'ignore_const_limit': True},
                         '000002.IND': {'name': '券商全收益',
                                        'unit_base': '自由流通股本',
                                        'industry_provider': '中信',
                                        'industry_level': 2.0,
                                        'industry': '证券',
                                        'ignore_new_stock_period': 90,
                                        'ignore_st': True,
                                        'ignore_pause': True,
                                        'ignore_const_limit': False}}
