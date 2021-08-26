import datetime as dt
import json
import logging
import re
from io import StringIO
from typing import Sequence, Union

import pandas as pd
import requests
from tqdm import tqdm

from .data_source import DataSource
from .. import date_utils, utils
from ..config import get_db_interface
from ..database_interface import DBInterface
from ..tickers import StockTickers, TickerTranslator


class WebDataCrawler(DataSource):
    """Get data through HTTP connections"""
    _SW_INDUSTRY_URL = 'http://www.swsindex.com/downloadfiles.aspx'
    _HEADER = {
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36',
    }
    _ZZ_INDUSTRY_URL = 'http://www.csindex.com.cn/zh-CN/downloads/industry-price-earnings-ratio-detail'

    def __init__(self, db_schema_loc: str = None, init: bool = False, db_interface: DBInterface = None) -> None:
        if db_interface is None:
            db_interface = get_db_interface()
        super().__init__(db_interface)
        if init:
            logging.getLogger(__name__).debug('检查数据库完整性.')
            self._db_parameters = utils.load_param('db_schema.json', db_schema_loc)
            for table_name, type_info in self._db_parameters.items():
                self.db_interface.create_table(table_name, type_info)

        self._stock_list = StockTickers(db_interface).ticker()

    def get_sw_industry(self) -> None:
        """获取申万一级行业"""
        header = self._HEADER
        header['referer'] = 'http://www.swsindex.com/idx0530.aspx'
        params = {'swindexcode': 'SwClass', 'type': 530, 'columnid': 8892}
        response = requests.post(self._SW_INDUSTRY_URL, headers=header, params=params)
        raw_data = pd.read_html(response.content.decode('gbk'))[0]

        def convert_dt(x: str) -> dt.datetime:
            date, time = x.split(' ')
            date_parts = [int(it) for it in date.split('/')]
            time_parts = [int(it) for it in time.split(':')]
            ret = dt.datetime(*date_parts, *time_parts)
            return ret

        raw_data['DateTime'] = raw_data['起始日期'].map(convert_dt)
        raw_data['ID'] = raw_data['股票代码'].map(stock_code2ts_code)

        raw_data.set_index(['DateTime', 'ID'], inplace=True)
        self.db_interface.update_df(raw_data[['行业名称']], '申万一级行业')

    @date_utils.dtlize_input_dates
    def get_zz_industry(self, date: date_utils.DateType) -> None:
        """获取中证4级行业"""
        referer_template = 'http://www.csindex.com.cn/zh-CN/downloads/industry-price-earnings-ratio?type=zz1&date='
        date_str = date_utils.date_type2str(date, '-')
        header = self._HEADER
        header['referer'] = referer_template + date_str
        storage = []

        with tqdm(self._stock_list) as pbar:
            for it in self._stock_list:
                pbar.set_description(f'正在获取{it}的中证行业数据')
                params = {'date': date_str, 'class': 2, 'search': 1, 'csrc_code': it.split('.')[0]}
                response = requests.get(self._ZZ_INDUSTRY_URL, headers=header, params=params)
                res_table = pd.read_html(response.text)[0]
                storage.append(res_table)
                pbar.update(1)
        data = pd.concat(storage)
        data['股票代码'] = data['股票代码'].map(stock_code2ts_code)
        data['trade_date'] = date
        useful_data = data[['trade_date', '股票代码', '所属中证行业四级名称']]
        useful_data.columns = ['DateTime', 'ID', '行业名称']
        useful_data.set_index(['DateTime', 'ID'], inplace=True)
        self.db_interface.update_df(useful_data, '中证行业')


def stock_code2ts_code(stock_code: Union[int, str]) -> str:
    stock_code = int(stock_code)
    return f'{stock_code:06}.SH' if stock_code >= 600000 else f'{stock_code:06}.SZ'


def ts_code2stock_code(ts_code: str) -> str:
    return ts_code.split()[0]


def get_current_cffex_contracts(products: Union[str, Sequence[str]]):
    """Get IC, IF, IH, IO contracts from CFFEX"""
    today = dt.datetime.today()
    url = f'http://www.cffex.com.cn/sj/jycs/{today.strftime("%Y%m")}/{today.strftime("%d")}/{today.strftime("%Y%m%d")}_1.csv'
    rsp = requests.get(url)
    rsp.encoding = 'gbk'
    data = pd.read_csv(StringIO(rsp.text), skiprows=1)
    tickers = data['合约代码'].tolist()
    if isinstance(products, str):
        products = [products]
    ret = [it for it in tickers if it[:2] in products]
    return ret


class EastMoneyCrawler(DataSource):
    def __init__(self, db_interface: DBInterface = None, db_schema_loc: str = None) -> None:
        if db_interface is None:
            db_interface = get_db_interface()
        super().__init__(db_interface)
        self._db_parameters = utils.load_param('db_schema.json', db_schema_loc)
        self.url_template = 'http://fund.eastmoney.com/Data/funddataIndex_Interface.aspx?dt=9&page={page}&rank=FSRQ&sort=desc&gs=&ftype=&year={year}'
        self.default_start_year = dt.datetime(2015, 1, 1)
        self.table_name = '公募基金拆分'
        self.ticker_translator = TickerTranslator(self.db_interface)

    def update_open_fund_split(self):
        latest_ts = self.db_interface.get_latest_timestamp(self.table_name, self.default_start_year)
        storage = []
        for year in range(latest_ts.year, dt.datetime.now().year + 1):
            total_page, res = self._get_url_content(year, 1)
            storage.append(res)

            for page in range(2, total_page + 1):
                _, res = self._get_url_content(year, page)
                storage.append(res)

        of_data = pd.concat(storage).loc[:, ['DateTime', 'ID', '拆分比例']]
        of_data.ID = [f'{it}.OF' for it in of_data.ID.tolist()]
        of_data = of_data.loc[~(of_data['拆分比例'] == '暂未披露'), :]

        exchange_data = of_data.copy()
        exchange_data.ID = [self.ticker_translator.of_ticker_to_exchange_ticker(it) for it in exchange_data.ID]
        exchange_data = exchange_data.dropna()

        df = pd.concat([of_data, exchange_data]).set_index(['DateTime', 'ID'])
        self.db_interface.update_df(df, self.table_name)

    def _get_url_content(self, year: int, page: int):
        url = self.url_template.format(page=page, year=year)
        rsp = requests.get(url)
        values = re.findall(r'var.*?=\s*(.*?);', rsp.text, re.DOTALL | re.MULTILINE)
        total_page = json.loads(values[0])[0]
        j = json.loads(values[1])
        res = pd.DataFrame(j, columns=['ID', 'name', 'DateTime', 'content', '拆分比例', 'can_buy'])
        return total_page, res
