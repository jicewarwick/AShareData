Barra
===========

Barra Partial Replica

Assumptions
---------------------
- Estimation Universe: ``全市场.IND`` 所包含的股票, 即上市超过244个交易日, 当日非停牌,非ST,非一字板,净资产为正的股票
- Market Return: 以自由流通市值加权的 ``Estimation Universe`` 的收益率
- Risk Free Rate: 3个月 ``SHIBOR``


基类
-----
.. autoclass:: AShareData.Factor.Factor
    :members:
    :inherited-members:
