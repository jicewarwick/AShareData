from .model import FinancialModel
from ..config import get_db_interface
from ..DBInterface import DBInterface


class CapitalAssetPricingModel(FinancialModel):
    def __init__(self, db_interface: DBInterface = None):
        super().__init__('Capital Asset Pricing Model', ['FF3_RM'])
        self.db_interface = db_interface if db_interface else get_db_interface()
