# A股数据获取及本地SQL储存与读取
Manual:
- 在 `config.json` 里填写相关信息. 模板文件为 `config_example.json`
- 已完成数据:
    - 交易日历
    - 股票
        - 股票列表
        - 上市公司基本信息
        - IPO新股列表
        - 日行情
        - 中信, 中证, 申万, Wind行业
        - 股票曾用名 / ST处理情况
        - 财报
        - 指数日行情, 列类似于股票日行情
    - 期货
        - 合约列表
        - 日行情
    - 期权
        - 合约列表
        - 行情
    - 基金
        - ETF基金列表
        - ETF日行情
    - 股票指数
        - 日行情
    - 自合成指标:
        - 股票涨跌停一字板
        - 股票自定义指数合成

Dependencies:
- numpy
- pandas
- tushare
- sqlalchemy
- tqdm: 进度显示
- requests
- sortedcontainers

Optional:
- pymysql: 数据库驱动
- pymysqldb
- WindPy
- alphalens
