import json
from typing import Dict, Optional, Union

import sqlalchemy as sa
from sqlalchemy.engine.url import URL

from .database_interface import DBInterface, MySQLInterface

__config__: Dict = None
__db_interface__: DBInterface = None


def prepare_engine(config: Dict) -> sa.engine.Engine:
    """Create sqlalchemy engine from config dict"""
    url = URL(drivername=config['driver'], host=config['host'], port=config['port'], database=config['database'],
              username=config['username'], password=config['password'],
              query={'charset': 'utf8mb4'})
    return sa.create_engine(url)


def generate_db_interface_from_config(config_loc: Union[str, Dict], init: bool = False) -> Optional[DBInterface]:
    if isinstance(config_loc, str):
        with open(config_loc, 'r', encoding='utf-8') as f:
            global_config = json.load(f)
    else:
        global_config = config_loc
    if 'mysql' in global_config['db_interface']['driver']:
        engine = prepare_engine(global_config['db_interface'])
        return MySQLInterface(engine, init=init)


def set_global_config(config_loc: str):
    global __config__
    with open(config_loc, 'r', encoding='utf-8') as f:
        __config__ = json.load(f)


def get_global_config():
    global __config__
    if __config__ is None:
        raise ValueError('Global configuration not set. Please use "set_global_config" to initialize.')
    return __config__


def get_db_interface():
    global __db_interface__
    if __db_interface__ is None:
        __db_interface__ = generate_db_interface_from_config(get_global_config())
    return __db_interface__


def set_db_interface(db_interface: DBInterface):
    __db_interface__ = db_interface
