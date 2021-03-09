import sys
import datetime as dt

from AShareData import IndexHighlighter, set_global_config
from AShareData.model.FamaFrench3FactorModel import FamaFrench3FactorModel

if __name__ == '__main__':
    set_global_config(sys.argv[1])
    IndexHighlighter().summary()
    model = FamaFrench3FactorModel()
    date = dt.datetime.combine(dt.date.today(), dt.time())
    print(model.get_factor_return(date))
