# 自动化Tushare数据获取和MySQL储存
Manual:
- 在 `config.json` 里填写相关信息. 模板文件为 `config_example.json`
- 已完成函数:
    - get_company_info(获取上市公司基本信息)
    - get_daily_hq(获取股票每日行情:开高低收, 量额, 复权因子, 换手率, 市盈, 市净, 市销, 股本, 市值)
    - get_past_names(股票曾用名)
    - get_ipo_info(IPO新股列表)
    - get_all_stocks(股票列表)
    - get_calendar(交易日历)
    - update_index_daily(指数日行情, 列类似于股票日行情)

Dependencies:
- numpy
- pandas
- tushare
- sqlalchemy
- tqdm: 展示下载进度

Optional:
- pymysql: 数据库驱动
