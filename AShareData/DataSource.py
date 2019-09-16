from AShareData.DBInterface import DBInterface


class DataSource(object):
    def __init__(self, db_interface: DBInterface):
        """数据源基类"""
        self.db_interface = db_interface
