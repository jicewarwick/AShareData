import datetime as dt
import json

import pandas as pd
import requests

from DataFrameMySQLWriter import DataFrameMySQLWriter


class WebDataCrawler(object):
    SW_INDUSTRY_URL = 'http://www.swsindex.com/downloadfiles.aspx'
    HEADER = {
        'Connection': 'keep-alive',
        'User-Agent': "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36",
    }

    def __init__(self, param_json: str, ip: str, port: int, username: str, password: str, db_name='tushare_db') -> None:
        with open(param_json, 'r', encoding='utf-8') as f:
            parameters = json.load(f)
        self._db_parameters = parameters['数据库参数']

        self.mysql_writer = DataFrameMySQLWriter(ip, port, username, password, db_name)
        for table_name, type_info in self._db_parameters.items():
            self.mysql_writer.create_table(table_name, type_info)

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
        raw_data['ID'] = raw_data['股票代码'].map(lambda x: f'{x:06}.SH' if x >= 600000 else f'{x:06d}.SZ')

        raw_data.set_index(['DateTime', 'ID'], inplace=True)
        self.mysql_writer.update_df(raw_data[['行业名称']], '申万一级行业')
