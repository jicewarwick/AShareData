class StyleFactor(object):
    pass


class Size(StyleFactor):
    """Size

    Defined as 1.0 · LNCAP
    """

    def __init__(self):
        super().__init__()


class Beta(StyleFactor):
    """Beta

    Defined as 1.0 · BETA
    """

    def __init__(self):
        super().__init__()


class Momentum(StyleFactor):
    """Momentum

    Defined as 1.0 · RSTR
    """

    def __init__(self):
        super().__init__()


class ResidualVolatility(StyleFactor):
    """Residual Volatility

    Defined as 0.74 · DASTD + 0.16 · CMRA + 0.10 · HSIGMA
    """

    def __init__(self):
        super().__init__()


class NonLinearSize(StyleFactor):
    """Non-linear Size

    Defined as 1.0 · NLSIZE
    """

    def __init__(self):
        super().__init__()


class BookToPrice(StyleFactor):
    """Book-to-Price

    Defined as 1.0 · BTOP
    """

    def __init__(self):
        super().__init__()


class Liquidity(StyleFactor):
    """Liquidity

    Defined as 0.35 · STOM + 0.35 · STOQ + 0.30 · STOA

    The Liquidity factor is orthogonalized with respect to Size to reduce collinearity.
    """

    def __init__(self):
        super().__init__()


class EarningsYield(StyleFactor):
    """Earnings Yield

    Defined as 0.68 · EPFWD + 0.21 · CETOP + 0.11 · ETOP
    """

    def __init__(self):
        super().__init__()


class Growth(StyleFactor):
    """Growth

    Defined as 0.18 · EGRLF+0.11 · EGRSF + 0.24 · EGRO + 0.47 · SGRO
    """

    def __init__(self):
        super().__init__()


class Leverage(StyleFactor):
    """Leverage

    Defined as 0.38 · MLEV + 0.35 · DTOA + 0.27 · BLEV
    """

    def __init__(self):
        super().__init__()
