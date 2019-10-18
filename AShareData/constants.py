TRADING_DAYS_IN_YEAR = 252

# exchanges
STOCK_EXCHANGES = ['SSE', 'SZSE']
FUTURE_EXCHANGES = ['CFFEX', 'DCE', 'CZCE', 'SHFE', 'INE']
ALL_EXCHANGES = STOCK_EXCHANGES + FUTURE_EXCHANGES

# indexes
STOCK_INDEXES = {'上证指数': '000001.SH', '深证成指': '399001.SZ', '中小板指': '399005.SZ', '创业板指': '399006.SZ',
                 '上证50': '000016.SH', '沪深300': '000300.SH', '中证500': '000905.SH'}
BOARD_INDEXES = ['000016.SH', '399300.SH', '000905.SH']

# financial statements
FINANCIAL_STATEMENTS_TYPE = ['资产负债表', '利润表', '现金流量表']
BALANCE_SHEETS = ['合并资产负债表', '母公司资产负债表']
INCOME_STATEMENTS = ['合并利润表', '合并单季度利润表', '母公司单季度利润表']
CASH_FLOW_STATEMENTS = ['合并现金流量表', '合并单季度现金流量表', '母公司单季度现金流量表']
FINANCIAL_STATEMENTS = BALANCE_SHEETS + INCOME_STATEMENTS + CASH_FLOW_STATEMENTS

# industry constants
INDUSTRY_DATA_PROVIDER = ['中信', '申万', '中证', 'Wind']
INDUSTRY_DATA_PROVIDER_CODE_DICT = {'中信': 'citic', '申万': 'sw', '中证': 'csi', 'Wind': 'gics'}
INDUSTRY_LEVEL = {'中信': 3, '申万': 3, '中证': 4, 'Wind': 4}
INDUSTRY_START_DATE = {'中信': '20030103', '申万': '20050527', '中证': '20161212', 'Wind': '20050105'}
