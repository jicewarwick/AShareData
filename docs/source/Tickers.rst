Tickers
=======

``Ticker`` class helps to select tickers that you need. They implements

* ``.all_ticker()`` to get all tickers that belongs to that type, alive or dead
* ``.ticker(date)`` to get all ticker available on that date
* ``.list_date()`` returns a dict mapping ticker to the time it is listed
* ``.get_list_date(ticker)`` return ``ticker``'s list date

基类
-----
.. autoclass:: AShareData.Tickers.TickersBase
    :members:

股票, 股基
-------------
.. autoclass:: AShareData.Tickers.StockTickers

.. autoclass:: AShareData.Tickers.StockFundTickers
.. autoclass:: AShareData.Tickers.ExchangeStockETFTickers
.. autoclass:: AShareData.Tickers.EnhancedIndexFund

.. autoclass:: AShareData.Tickers.StockOTCFundTickers
.. autoclass:: AShareData.Tickers.ActiveManagedOTCStockFundTickers

债券, 债基
------------
.. autoclass:: AShareData.Tickers.ConvertibleBondTickers
.. autoclass:: AShareData.Tickers.BondETFTickers

基金
------------
.. autoclass:: AShareData.Tickers.FundTickers
.. autoclass:: AShareData.Tickers.ETFTickers
.. autoclass:: AShareData.Tickers.ExchangeFundTickers
.. autoclass:: AShareData.Tickers.IndexFund

衍生品
--------
.. autoclass:: AShareData.Tickers.FutureTickers
.. autoclass:: AShareData.Tickers.OptionTickers
.. autoclass:: AShareData.Tickers.IndexOptionTickers
.. autoclass:: AShareData.Tickers.ETFOptionTickers

股票筛选器
------------------
.. autoclass:: AShareData.utils.StockSelectionPolicy
    :members:

.. autoclass:: AShareData.Tickers.StockTickerSelector
    :members:

