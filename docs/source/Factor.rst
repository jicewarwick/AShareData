Factors
===========

Factor class:

* implemented ``.get_data()`` function that retrieve and/or computes required data with (DateTime, ID) multiindex
* common numerical operations(+, -, \*, /), boolean operation(>, >=, ==, <=, <, !=) and transformation(log, pct_change, etc) are supported

基类
-----
.. autoclass:: AShareData.Factor.Factor
    :members:
    :inherited-members:

非财报数据
-----------
.. autoclass:: AShareData.Factor.IndexConstitute
    :members:

.. autoclass:: AShareData.Factor.IndustryFactor
    :members:

行情数据
^^^^^^^^
.. autoclass:: AShareData.Factor.CompactFactor
    :members:

.. autoclass:: AShareData.Factor.OnTheRecordFactor
    :members:

.. autoclass:: AShareData.Factor.ContinuousFactor
    :members:

.. autoclass:: AShareData.Factor.BetaFactor
    :members:


财报数据
------------
.. autoclass:: AShareData.Factor.QuarterlyFactor
    :members:

.. autoclass:: AShareData.Factor.LatestAccountingFactor
    :members:

.. autoclass:: AShareData.Factor.LatestQuarterAccountingFactor
    :members:

.. autoclass:: AShareData.Factor.YearlyReportAccountingFactor
    :members:

.. autoclass:: AShareData.Factor.QOQAccountingFactor
    :members:

.. autoclass:: AShareData.Factor.YOYPeriodAccountingFactor
    :members:

.. autoclass:: AShareData.Factor.YOYQuarterAccountingFactor
    :members:

.. autoclass:: AShareData.Factor.TTMAccountingFactor
    :members:

