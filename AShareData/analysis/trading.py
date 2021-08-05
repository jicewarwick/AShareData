import pandas as pd

import AShareData as asd


class TradingAnalysis(object):
    def __init__(self, db_interface: asd.database_interface.DBInterface = None):
        self.db_interface = db_interface if db_interface else asd.get_db_interface()
        self.data_reader = asd.AShareDataReader(self.db_interface)

    def trading_volume_summary(self, trading_records: pd.DataFrame) -> pd.DataFrame:
        vol_summary = trading_records.groupby(['ID', 'tradeDirection'], as_index=False).tradeVolume.sum()
        single_direction_vol = vol_summary.groupby('ID').max()
        bi_direction_vol = vol_summary.groupby('ID').sum()

        date = trading_records.DateTime[0].date()
        market_vol_info = self.data_reader.stock_trading_volume.get_data(dates=date)
        market_vol_info.index = market_vol_info.index.droplevel('DateTime')

        single_ratio = (single_direction_vol.tradeVolume / market_vol_info).dropna()
        bi_direction_ratio = (bi_direction_vol.tradeVolume / market_vol_info / 2).dropna()
        ret = pd.concat([single_ratio, bi_direction_ratio], axis=1, sort=False)
        ret.columns = ['单向成交量占比', '双向成交量占比']
        ret = ret.sort_values('单向成交量占比', ascending=False)

        return ret
