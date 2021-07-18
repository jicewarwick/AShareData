import datetime as dt
import logging
from io import StringIO
from typing import Sequence, Union

import pandas as pd
import requests
from tqdm import tqdm

from .data_source import DataSource
from .. import date_utils, utils
from ..config import get_db_interface
from ..database_interface import DBInterface
from ..tickers import StockTickers


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
