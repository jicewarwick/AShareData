import datetime as dt

import pandas as pd

from .common import BarraComputer
from ..database_interface import DBInterface


class BarraStyleFactorComputerBase(BarraComputer):
    TABLE_NAME = 'BarraStyleFactor'

    def __init__(self, db_interface: DBInterface = None):
        """ Base Class for Barra Style Factor Computing

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.name = self.__class__.__name__

    def compute_raw_data(self, date: dt.datetime) -> pd.Series:
        """Compute Style Factor according to its compositing descriptors"""
        raise NotImplementedError()

    def compute(self, date: dt.datetime) -> pd.Series:
        """ Compute Style Factor

        :param date: date to compute
        :return: style factor for the date
        """
        raise NotImplementedError()

    def compute_and_store(self, date: dt.datetime) -> None:
        """ Compute Style Factor and Store in DB

        Combine style descriptors into style factors and then re-standardize.
        The results are stored in `BarraStyleFactor` table

        USE4 Methodology Pg.9:

        The final step of constructing the style factor is to re-standardize the exposures so that
        they have a cap-weighted mean of 0 and an equal-weighted standard deviation of 1.

        :param date: date to compute
        :return: style factor for the date
        """
        data = self.compute(date)
        self.db_interface.update_df(data, self.TABLE_NAME)


class SimpleBarraStyleFactorComputer(BarraStyleFactorComputerBase):
    def __init__(self, db_interface: DBInterface = None):
        """Barra Style Factor that has only one descriptor

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.descriptor_name = None

    def compute_raw_data(self, date: dt.datetime) -> pd.Series:
        data = self.db_interface.read_table(self.TABLE_NAME, self.descriptor_name, dates=date)
        return data

    def compute(self, date: dt.datetime) -> pd.Series:
        """ Compute Style Factor

        Combine style descriptors into style factors.

        :param date: date to compute
        :return: style factor for the date
        """
        data = self.compute_raw_data(date)
        data.name = self.name
        return data


class CompositeBarraStyleFactorComputer(BarraStyleFactorComputerBase):
    def __init__(self, db_interface: DBInterface = None):
        """Barra Style Factor that consists multiple descriptors

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)

    def compute_raw_data(self, date: dt.datetime) -> pd.Series:
        raise NotImplementedError()

    def compute(self, date: dt.datetime) -> pd.Series:
        """ Compute Style Factor

        Combine style descriptors into style factors and then re-standardize.

        USE4 Methodology Pg.9:

        The final step of constructing the style factor is to re-standardize the exposures so that
        they have a cap-weighted mean of 0 and an equal-weighted standard deviation of 1.

        :param date: date to compute
        :return: style factor for the date
        """
        raw = self.compute_raw_data(date)
        data = self.standardize(raw)
        data.name = self.name
        return data


class Size(SimpleBarraStyleFactorComputer):
    """Size

    Defined as 1.0 · LNCAP
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.descriptor_name = 'LNCAP'


class Beta(SimpleBarraStyleFactorComputer):
    """Beta

    Defined as 1.0 · BETA
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.descriptor_name = 'BETA'


class Momentum(SimpleBarraStyleFactorComputer):
    """Momentum

    Defined as 1.0 · RSTR
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.descriptor_name = 'RSTR'


class ResidualVolatility(CompositeBarraStyleFactorComputer):
    """Residual Volatility

    Defined as 0.74 · DASTD + 0.16 · CMRA + 0.10 · HSIGMA

    The Residual Volatility factor is orthogonalized with respect to Beta and Size to reduce collinearity
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)

    def compute_raw_data(self, date: dt.datetime) -> pd.Series:
        raw = self.db_interface.read_table(self.TABLE_NAME, ['DASTD', 'CMRA', 'HSIGMA'], dates=date)
        data = 0.74 * raw['DASTD'] + 0.16 * raw['CMRA'] + 0.10 * raw['HSIGMA']
        return data

    def compute(self, date: dt.datetime) -> pd.Series:
        raw = self.compute_raw_data(date)
        size_beta_data = self.db_interface.read_table(self.TABLE_NAME, ['Size', 'Beta'], dates=date)
        res = self.orthogonalize(self.orthogonalize(raw, size_beta_data['size']), size_beta_data['beta'])
        res = self.standardize(res)
        return res


class NonLinearSize(SimpleBarraStyleFactorComputer):
    """Non-linear Size

    Defined as 1.0 · NLSIZE

    First, the standardized Size exposure (i.e., log of market cap) is cubed.
    The resulting factor is then orthogonalized with respect to the Size factor on a regression-weighted basis.
    Finally, the factor is winsorized and standardized.
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.descriptor_name = 'NLSIZE'

    def compute(self, date: dt.datetime) -> pd.Series:
        raw = self.compute_raw_data(date)
        size_data = self.db_interface.read_table(self.TABLE_NAME, 'Size', dates=date)
        res = self.orthogonalize(raw, size_data)
        res = self.trim_extreme(res)
        res = self.standardize(res)
        return res


class BookToPrice(SimpleBarraStyleFactorComputer):
    """Book-to-Price

    Defined as 1.0 · BTOP
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)
        self.descriptor_name = 'BTOP'


class Liquidity(CompositeBarraStyleFactorComputer):
    """Liquidity

    Defined as 0.35 · STOM + 0.35 · STOQ + 0.30 · STOA

    The Liquidity factor is orthogonalized with respect to Size to reduce collinearity.
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)

    def compute_raw_data(self, date: dt.datetime) -> pd.Series:
        raw = self.db_interface.read_table(self.TABLE_NAME, ['STOM', 'STOQ', 'STOA'], dates=date)
        data = 0.35 * raw['STOM'] + 0.35 * raw['STOQ'] + 0.30 * raw['STOA']
        return data

    def compute(self, date: dt.datetime) -> pd.Series:
        raw = self.compute_raw_data(date)
        size_data = self.db_interface.read_table(self.TABLE_NAME, 'Size', dates=date)
        res = self.orthogonalize(raw, size_data)
        res = self.standardize(res)
        return res


class EarningsYield(CompositeBarraStyleFactorComputer):
    """Earnings Yield

    Defined as 0.68 · EPFWD + 0.21 · CETOP + 0.11 · ETOP
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)

    def compute_raw_data(self, date: dt.datetime) -> pd.Series:
        raw = self.db_interface.read_table(self.TABLE_NAME, ['EPFWD', 'CETOP', 'ETOP'], dates=date).fillna(0)
        data = 0.68 * raw['EPFWD'] + 0.21 * raw['CETOP'] + 0.11 * raw['ETOP']
        return data


class Growth(CompositeBarraStyleFactorComputer):
    """Growth

    Defined as 0.18 · EGRLF + 0.11 · EGRSF + 0.24 · EGRO + 0.47 · SGRO
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)

    def compute_raw_data(self, date: dt.datetime) -> pd.Series:
        raw = self.db_interface.read_table(self.TABLE_NAME, ['EGRLF', 'EGRSF', 'EGRO', 'SGRO'], dates=date).fillna(0)
        data = 0.18 * raw['EGRLF'] + 0.11 * raw['EGRSF'] + 0.24 * raw['EGRO'] + 0.47 * raw['SGRO']
        return data


class Leverage(CompositeBarraStyleFactorComputer):
    """Leverage

    Defined as 0.38 · MLEV + 0.35 · DTOA + 0.27 · BLEV
    """

    def __init__(self, db_interface: DBInterface = None):
        super().__init__(db_interface)

    def compute_raw_data(self, date: dt.datetime) -> pd.Series:
        raw = self.db_interface.read_table(self.TABLE_NAME, ['MLEV', 'DTOA', 'BLEV'], dates=date)
        data = 0.38 * raw['MLEV'] + 0.35 * raw['DTOA'] + 0.27 * raw['BLEV']
        return data
