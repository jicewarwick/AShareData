from AShareData.DBInterface import DBInterface


class DataSource(object):
    """Data Source Base Class"""
    def __init__(self, db_interface: DBInterface):
        self.db_interface = db_interface
