from ..ashare_data_reader import AShareDataReader
from ..data_source.data_source import DataSource
from ..database_interface import DBInterface


class FactorCompositor(DataSource):
    def __init__(self, db_interface: DBInterface = None):
        """
        Factor Compositor

        This class composite factors from raw market/financial info

        :param db_interface: DBInterface
        """
        super().__init__(db_interface)
        self.data_reader = AShareDataReader(db_interface)

    def update(self):
        """更新数据"""
        raise NotImplementedError()
