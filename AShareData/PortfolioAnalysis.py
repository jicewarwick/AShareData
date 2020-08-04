from . import AShareDataReader, MySQLInterface, prepare_engine


class ASharePortfolioAnalysis(object):
    def __init__(self, config_loc):
        super().__init__()
        engine = prepare_engine(config_loc)
        mysql_writer = MySQLInterface(engine, init=True)
        self.data_reader = AShareDataReader(mysql_writer)

    def beta_portfolio(self):
        pass

    def size_portfolio(self):
        pass
