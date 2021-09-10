DataSources
===========
DataSources writes data into database.

Outside data source inherent from :py:class:`~.DataSource`.
They can be used as a context manager. log ins and log outs should be handled in :py:meth:`~.DataSource.__enter__` and :py:meth:`~.DataSource.__exit__`

Base Class
-----------
.. autoclass:: AShareData.data_source.DataSource
    :members:

Market Data Implementation
----------------------------
Tushare
^^^^^^^^^^
.. autoclass:: AShareData.TushareData
    :members:

Wind
^^^^^^^^^^
.. autoclass:: AShareData.WindData
    :members:

Join Quant
^^^^^^^^^^
.. autoclass:: AShareData.JQData
    :members:

通达讯
^^^^^^^^^^
.. autoclass:: AShareData.TDXData
    :members:

Web HTTP request
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AShareData.WebDataCrawler
    :members:

Internally Computed Data
--------------------------
Base Class
^^^^^^^^^^^
.. autoclass:: AShareData.factor_compositor.FactorCompositor
    :members:

Implementation
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AShareData.factor_compositor.ConstLimitStockFactorCompositor
    :members:

.. autoclass:: AShareData.factor_compositor.FundAdjFactorCompositor
    :members:


Index Compositor
"""""""""""""""""""""
.. autoclass:: AShareData.utils.StockSelectionPolicy
    :members:

.. autoclass:: AShareData.utils.StockIndexCompositionPolicy
    :members:

.. autoclass:: AShareData.factor_compositor.IndexCompositor
    :members:


Factor Portfolio Return
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AShareData.factor_portfolio.FactorPortfolioPolicy
    :members:

.. autoclass:: AShareData.factor_portfolio.FactorPortfolio
    :members:
