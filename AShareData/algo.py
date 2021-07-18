import re
from typing import Dict, Optional, Sequence


def chunk_list(l: Sequence, n: int):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def human_sort(l):
    """ Sort the given list in the way that humans expect.
    """
    l = l.copy()
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)
    return l


def get_less_or_equal_of_a_in_b(a: Sequence, b: Sequence) -> Dict:
    """ return {va: vb} s.t. vb = max(tb) where tb <= va and tb in b, for all va in a

    :param a: sored sequence of comparable T
    :param b: non-empty sorted sequence of comparable T
    :return: {va: vb} s.t. vb = max(b) given vb <= va for all va in a
    """
    if len(b) <= 0:
        raise ValueError(f'b({b}) cannot be empty')
    ret = {}
    i, j = 0, 1
    la, lb = len(a), len(b)
    while i < la and a[i] < b[0]:
        i += 1
    while i < la:
        while j < lb and a[i] >= b[j]:
            j += 1
        ret[a[i]] = b[j - 1]
        i += 1
    return ret


def extract_close_operate_period(fund_name: str) -> Optional[int]:
    if fund_name:
        if '封闭运作' in fund_name:
            fund_name = fund_name.replace('三', '3').replace('二', '2').replace('一', '1').replace('两', '2')
            if '年' in fund_name:
                return int(fund_name[fund_name.index('年') - 1]) * 12
            elif '月' in fund_name:
                loc = fund_name.index('月') - 1
                ret_str = fund_name[loc - 2:loc] if fund_name[loc - 2].isnumeric() else fund_name[loc]
                return int(ret_str)
