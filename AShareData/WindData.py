import datetime as dt
import logging

import pandas as pd
from tqdm import tqdm

import AShareData.constants
import AShareData.utils as utils
from AShareData.DataFrameMySQLWriter import DataFrameMySQLWriter
from AShareData.WindWrapper import WindWrapper


class WindData(object):
    def __init__(self, engine):
        self.engine = engine
        self.mysql_writer = DataFrameMySQLWriter(engine)
        self.calendar = utils.get_calendar(engine)
        self.stocks = utils.get_stocks(engine)
        self.w = WindWrapper()
        self.w.connect()

    def _get_industry_data(self, wind_code, provider, date):
        wind_data = self.w.wsd(wind_code, f'industry_{AShareData.constants.INDUSTRY_DATA_PROVIDER_CODE_DICT[provider]}',
                               date, date, industryType=AShareData.constants.INDUSTRY_LEVEL[provider])
        wind_data.reset_index(inplace=True)
        wind_data.columns = ['ID', '行业名称']
        wind_data['DateTime'] = date
        wind_data = wind_data.set_index(['DateTime', 'ID'])
        wind_data['行业名称'] = wind_data['行业名称'].str.replace('[III|Ⅲ|IV|Ⅳ]', '')
        return wind_data

    def _find_industry(self, wind_code: str, provider: str,
                       start_date: utils.DateType, start_data: str,
                       end_date: utils.DateType, end_data: str) -> None:
        start_date = utils.date_type2datetime(start_date)
        end_date = utils.date_type2datetime(end_date)
        if start_data != end_data:
            logging.info(f'{wind_code} 的行业由 {start_date} 的 {start_data} 改为 {end_date} 的 {end_data}')
            while True:
                start_index = self.calendar.index(start_date)
                end_index = self.calendar.index(end_date)
                if end_index - start_index < 2:
                    entry = pd.DataFrame({'DateTime': end_date, 'ID': wind_code, '行业名称': end_data}, index=[0])
                    entry.set_index(['DateTime', 'ID'], inplace=True)
                    logging.info(f'插入数据: {wind_code} 于 {end_date} 的行业为 {end_data}')
                    self.mysql_writer.update_df(entry, f'{provider}行业')
                    break
                mid_date = self.calendar[int((start_index + end_index) / 2.0)]
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
        query_date = utils.trading_days_offset(self.calendar, dt.date.today(), -1)
        latest, _ = self.mysql_writer.get_progress(table_name)
        if latest is None:
            latest = utils.date_type2datetime(AShareData.constants.INDUSTRY_START_DATE[provider])
            initial_data = self._get_industry_data(self.stocks, provider, latest).dropna()
            self.mysql_writer.update_df(initial_data, table_name)
        else:
            initial_data = pd.read_sql_table(table_name, self.engine, index_col=['DateTime', 'ID'])
            initial_data = initial_data.unstack().ffill().tail(1).stack()

        new_data = self._get_industry_data(self.stocks, provider, query_date).dropna()

        # find diff
        tmp_data = pd.concat([initial_data, new_data]).unstack().droplevel(None, axis=1)
        tmp_data = tmp_data.where(tmp_data.notnull(), None)
        diff = (tmp_data != tmp_data.shift())
        diff_stock = diff.iloc[-1, :]
        diff_stock = diff_stock.loc[diff_stock].index.tolist()

        data_storage = []
        with tqdm(diff_stock) as pbar:
            for stock in diff_stock:
                pbar.set_description(f'获取{stock}的{provider}行业')
                stock_slice = tmp_data.loc[:, stock]
                data_storage.append(self._find_industry(stock, provider, latest, stock_slice[0],
                                                        query_date, stock_slice[1]))
                pbar.update(1)
