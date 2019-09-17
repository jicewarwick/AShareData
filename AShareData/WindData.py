import datetime as dt
import logging

import pandas as pd
from tqdm import tqdm

from AShareData import constants, utils
from AShareData.DataSource import DataSource
from AShareData.DBInterface import DBInterface, get_stocks
from AShareData.TradingCalendar import TradingCalendar
from AShareData.WindWrapper import WindWrapper


class WindData(DataSource):
    def __init__(self, db_interface: DBInterface):
        super().__init__(db_interface)
        self.calendar = TradingCalendar(db_interface)
        self.stocks = get_stocks(db_interface)
        self.w = WindWrapper()
        self.w.connect()

    def _get_industry_data(self, wind_code, provider, date):
        wind_data = self.w.wsd(wind_code, f'industry_{constants.INDUSTRY_DATA_PROVIDER_CODE_DICT[provider]}',
                               date, date, industryType=constants.INDUSTRY_LEVEL[provider])
        wind_data.reset_index(inplace=True)
        wind_data.columns = ['ID', '行业名称']
        wind_data['DateTime'] = date
        wind_data = wind_data.set_index(['DateTime', 'ID'])
        wind_data['行业名称'] = wind_data['行业名称'].str.replace('III|Ⅲ|IV|Ⅳ$', '')
        return wind_data

    def _find_industry(self, wind_code: str, provider: str,
                       start_date: utils.DateType, start_data: str,
                       end_date: utils.DateType, end_data: str) -> None:
        if start_data != end_data:
            logging.info(f'{wind_code} 的行业由 {start_date} 的 {start_data} 改为 {end_date} 的 {end_data}')
            while True:
                if self.calendar.days_count(start_date, end_date) < 2:
                    entry = pd.DataFrame({'DateTime': end_date, 'ID': wind_code, '行业名称': end_data}, index=[0])
                    entry.set_index(['DateTime', 'ID'], inplace=True)
                    logging.info(f'插入数据: {wind_code} 于 {end_date} 的行业为 {end_data}')
                    self.db_interface.update_df(entry, f'{provider}行业')
                    break
                mid_date = self.calendar.middle(start_date, end_date)
                logging.debug(f'查询{wind_code} 在 {mid_date}的行业')
                mid_data = self._get_industry_data(wind_code, provider, mid_date).iloc[0, 0]
                if mid_data == start_data:
                    start_date = mid_date
                elif mid_data == end_data:
                    end_date = mid_date
                else:
                    self._find_industry(wind_code, provider, start_date, start_data, mid_date, mid_data)
                    self._find_industry(wind_code, provider, mid_date, mid_data, end_date, end_data)
                    break

    def update_industry(self, provider: str):
        table_name = f'{provider}行业'
        query_date = self.calendar.offset(dt.date.today(), -1)
        latest, _ = self.db_interface.get_progress(table_name)
        if latest is None:
            latest = utils.date_type2datetime(constants.INDUSTRY_START_DATE[provider])
            initial_data = self._get_industry_data(self.stocks, provider, latest).dropna()
            self.db_interface.update_df(initial_data, table_name)
        else:
            initial_data = self.db_interface.read_table(table_name, index_col=['DateTime', 'ID'])

        new_data = self._get_industry_data(self.stocks, provider, query_date).dropna()

        diff_stock = utils.compute_diff(new_data, initial_data)
        with tqdm(diff_stock) as pbar:
            for (_, stock), new_industry in diff_stock.iterrows():
                pbar.set_description(f'获取{stock}的{provider}行业')
                try:
                    old_industry = initial_data.loc[(slice(None), stock), '行业名称'].values[-1]
                except KeyError:
                    old_industry = None
                self._find_industry(stock, provider, latest, old_industry, query_date, new_industry['行业名称'])
                pbar.update(1)
