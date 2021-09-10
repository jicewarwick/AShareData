Factors
===========

Factor class:

* implemented ``.get_data()`` function that retrieve and/or computes required data with (DateTime, ID) multiindex
* common numerical operations(+, -, \*, /), boolean operation(>, >=, ==, <=, <, !=) and transformation(log, pct_change, etc) are supported

基类
-----
.. autoclass:: AShareData.factor.Factor
    :members:
    :inherited-members:

非财报数据
-----------
.. autoclass:: AShareData.factor.IndexConstitute
    :members:

.. autoclass:: AShareData.factor.IndustryFactor
    :members:

行情数据
^^^^^^^^
.. autoclass:: AShareData.factor.CompactFactor
    :members:

.. autoclass:: AShareData.factor.OnTheRecordFactor
    :members:

.. autoclass:: AShareData.factor.ContinuousFactor
    :members:

.. autoclass:: AShareData.factor.BetaFactor
    :members:


财报数据
------------
.. autoclass:: AShareData.factor.QuarterlyFactor
    :members:

.. autoclass:: AShareData.factor.LatestAccountingFactor
    :members:

.. autoclass:: AShareData.factor.LatestQuarterAccountingFactor
    :members:

.. autoclass:: AShareData.factor.YearlyReportAccountingFactor
    :members:

.. autoclass:: AShareData.factor.QOQAccountingFactor
    :members:

.. autoclass:: AShareData.factor.YOYPeriodAccountingFactor
    :members:

.. autoclass:: AShareData.factor.YOYQuarterAccountingFactor
    :members:

.. autoclass:: AShareData.factor.TTMAccountingFactor
    :members:

