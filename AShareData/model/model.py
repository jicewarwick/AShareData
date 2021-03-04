from typing import List


class FinancialModel(object):
    def __init__(self, model_name: str, factor_names: List[str]):
        self.model_name = model_name
        self.factor_names = factor_names.copy()
