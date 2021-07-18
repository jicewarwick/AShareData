import pandas as pd

import AShareData as asd

if __name__ == '__main__':
    config_loc = './config.json'
    asd.set_global_config(config_loc)

    data_reader = asd.AShareDataReader()
    calendar = asd.SHSZTradingCalendar()
    date = calendar.yesterday()

    industry = data_reader.industry('申万', 2).get_data(dates=date)
    cap = data_reader.stock_market_cap.get_data(dates=date) / 1e8
    sec_name = data_reader.sec_name.get_data(dates=date)

    df = pd.concat([sec_name, industry, cap], axis=1).dropna()
    df.columns = df.columns[:2].tolist() + ['cap']
    df = df.sort_values('cap', ascending=False)
    industry_big_name = df.groupby('申万2级行业').head(3)
    big_cap = df.head(300)
    all_ticker = pd.concat([industry_big_name, big_cap]).drop_duplicates().sort_index().droplevel('DateTime')

    company_info = asd.get_db_interface().read_table('上市公司基本信息', columns=['所在城市', '主要业务及产品', '经营范围'],
                                                     ids=all_ticker.index.get_level_values('ID').tolist())
    ret = pd.concat([all_ticker, company_info], axis=1)
    ret.to_excel('big_names.xlsx', freeze_panes=(0, 3))
