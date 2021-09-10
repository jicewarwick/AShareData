Barra
===========

Assumptions
---------------------
- Estimation Universe: ``全市场.IND`` 所包含的股票, 即上市超过244个交易日, 当日非停牌, 非ST, 非一字板, 净资产为正的股票
- Market Return: 以自由流通市值加权的 ``Estimation Universe`` 的收益率
- Risk Free Rate: 3个月 ``SHIBOR``

Factors
----------------------

Style Factors
^^^^^^^^^^^^^^^^^^

1. Size
""""""""""""""""""""""""""""
.. autoclass:: AShareData.barra.style_factors.Size
    :members:

.. autoclass:: AShareData.barra.descriptors.LNCAP
    :members:

2. Beta
""""""""""""""""""""""""""""
.. autoclass:: AShareData.barra.style_factors.Beta
    :members:

.. autoclass:: AShareData.barra.descriptors.BETA
    :members:

3. Momentum
""""""""""""""""""""""""""""
.. autoclass:: AShareData.barra.style_factors.Momentum
    :members:

.. autoclass:: AShareData.barra.descriptors.RSTR
    :members:


3. Residual Volatility
""""""""""""""""""""""""""""
.. autoclass:: AShareData.barra.style_factors.ResidualVolatility
    :members:

.. autoclass:: AShareData.barra.descriptors.DASTD
    :members:

.. autoclass:: AShareData.barra.descriptors.CMRA
    :members:

.. autoclass:: AShareData.barra.descriptors.HSIGMA
    :members:

4. Non-linear Size
""""""""""""""""""""""""""""
.. autoclass:: AShareData.barra.style_factors.NonLinearSize
    :members:

.. autoclass:: AShareData.barra.descriptors.NLSIZE
    :members:


5. Book-to-Price
""""""""""""""""""""""""""""
.. autoclass:: AShareData.barra.style_factors.BookToPrice
    :members:

.. autoclass:: AShareData.barra.descriptors.BTOP
    :members:

6. Liquidity
""""""""""""""""""""""""""""
.. autoclass:: AShareData.barra.style_factors.Liquidity
    :members:

.. autoclass:: AShareData.barra.descriptors.STOM
    :members:

.. autoclass:: AShareData.barra.descriptors.STOQ
    :members:

.. autoclass:: AShareData.barra.descriptors.STOA
    :members:


6. Earnings Yield
""""""""""""""""""""""""""""
.. autoclass:: AShareData.barra.style_factors.EarningsYield
    :members:

.. autoclass:: AShareData.barra.descriptors.EPFWD
    :members:

.. autoclass:: AShareData.barra.descriptors.CETOP
    :members:

.. autoclass:: AShareData.barra.descriptors.ETOP
    :members:


6. Growth
""""""""""""""""""""""""""""
.. autoclass:: AShareData.barra.style_factors.Growth
    :members:

.. autoclass:: AShareData.barra.descriptors.EGRLF
    :members:

.. autoclass:: AShareData.barra.descriptors.EGRSF
    :members:

.. autoclass:: AShareData.barra.descriptors.SGRO
    :members:

.. autoclass:: AShareData.barra.descriptors.EGRO
    :members:


6. Leverage
""""""""""""""""""""""""""""
.. autoclass:: AShareData.barra.style_factors.Leverage
    :members:

.. autoclass:: AShareData.barra.descriptors.MLEV
    :members:

.. autoclass:: AShareData.barra.descriptors.DTOA
    :members:

.. autoclass:: AShareData.barra.descriptors.BLEV
    :members:


Base Classes
""""""""""""""""""""""""""
.. autoclass:: AShareData.barra.descriptors.BarraComputer
    :members:

.. autoclass:: AShareData.barra.descriptors.BarraDescriptorComputer
    :members:

Helper Classes
"""""""""""""""""""""""""""
.. autoclass:: AShareData.barra.descriptors.BarraCAPMRegression
    :members:

