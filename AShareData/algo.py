import re
from typing import Dict, Sequence


def chunk_list(l: list, n: int):
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
    assert len(b) > 0, 'b cannot be empty'
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
