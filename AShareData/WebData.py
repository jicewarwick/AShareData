import datetime as dt
import json
import logging
from importlib.resources import open_text

import pandas as pd
import requests
from tqdm import tqdm

from AShareData import utils
from AShareData.DataSource import DataSource
from AShareData.DBInterface import DBInterface, get_stocks


class WebDataCrawler(DataSource):
    SW_INDUSTRY_URL = 'http://www.swsindex.com/downloadfiles.aspx'
    HEADER = {
        'Connection': 'keep-alive',
        'User-Agent': "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36",
    }
    ZX_INDUSTRY_URL = 'http://www.csindex.com.cn/zh-CN/downloads/industry-price-earnings-ratio-detail'

    def __init__(self, db_interface: DBInterface, db_schema_loc: str = None, init: bool = False) -> None:
        super().__init__(db_interface)
        if init:
            logging.debug('检查数据库完整性.')
            if db_schema_loc is None:
                f = open_text('AShareData.data', 'db_schema.json')
            else:
                f = open(db_schema_loc, 'r', encoding='utf-8')
            with f:
                self._db_parameters = json.load(f)
            for table_name, type_info in self._db_parameters.items():
                self.db_interface.create_table(table_name, type_info)

        self._stock_list = get_stocks(db_interface)

    def get_sw_industry(self) -> None:
        header = self.HEADER
        header['referer'] = 'http://www.swsindex.com/idx0530.aspx'
        params = {'swindexcode': 'SwClass', 'type': 530, 'columnid': 8892}
        response = requests.post(self.SW_INDUSTRY_URL, headers=header, params=params)
        raw_data = pd.read_html(response.content.decode('gbk'))[0]

        def convert_dt(x: str) -> dt.datetime:
            date, time = x.split(' ')
            date_parts = [int(it) for it in date.split('/')]
            time_parts = [int(it) for it in time.split(':')]
            ret = dt.datetime(*date_parts, *time_parts)
            return ret

        raw_data['DateTime'] = raw_data['起始日期'].map(convert_dt)
        raw_data['ID'] = raw_data['股票代码'].map(utils.stock_code2ts_code)

        raw_data.set_index(['DateTime', 'ID'], inplace=True)
        self.db_interface.update_df(raw_data[['行业名称']], '申万一级行业')

    def get_zx_industry(self, date: utils.DateType) -> None:
        referer_template = 'http://www.csindex.com.cn/zh-CN/downloads/industry-price-earnings-ratio?type=zz1&date='
        date_str = utils.date_type2str(date, '-')
        header = self.HEADER
        header['referer'] = referer_template + date_str
        storage = []

        with tqdm(self._stock_list) as pbar:
            for it in self._stock_list:
                pbar.set_description(f'正在获取{it}的中信行业数据')
                params = {'date': date_str, 'class': 2, 'search': 1, 'csrc_code': it.split('.')[0]}
                response = requests.get(self.ZX_INDUSTRY_URL, headers=header, params=params)
                res_table = pd.read_html(response.text)[0]
                storage.append(res_table)
                pbar.update(1)
        data = pd.concat(storage)
        data['股票代码'] = data['股票代码'].map(utils.stock_code2ts_code)
        data['trade_date'] = utils.date_type2datetime(date)
        useful_data = data[['trade_date', '股票代码', '所属中证行业四级名称']]
        useful_data.columns = ['DateTime', 'ID', '行业名称']
        useful_data.set_index(['DateTime', 'ID'], inplace=True)
        self.db_interface.update_df(useful_data, '中证行业')
