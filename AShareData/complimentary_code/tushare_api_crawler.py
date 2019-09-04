import json
from collections import defaultdict

import pandas as pd
import requests
from bs4 import BeautifulSoup

if __name__ == '__main__':
    # persistence loc
    save_loc = 'param_raw.json'

    # web addresses
    domain = 'https://tushare.pro'
    api_homepage = domain + '/document/2'

    r = requests.get(api_homepage)
    bs = BeautifulSoup(r.text, 'lxml')

    # find all links in the api_homepage
    link_storage = []
    for link in bs.body.div.section.find_all('a'):
        l = link.get('href')
        if 'https' not in l:
            l = domain + l
        link_storage.append(l)
    link_storage = sorted(list(set(link_storage)))

    # fetch all links and try to get parameters
    page_info = defaultdict(dict)
    for link in link_storage:
        r = requests.get(link)
        bs = BeautifulSoup(r.text, 'lxml')
        try:
            title = bs.h2.text
            tables = pd.read_html(r.text)
            desc = bs.find_all('strong')
            page_info[title] = defaultdict(dict)
            for i, it in enumerate(tables):
                if {'名称', '描述'} <= set(it.columns):
                    page_info[title][desc[i].text] = {}
                    for _, row in it.iterrows():
                        page_info[title][desc[i].text][row['名称']] = row['描述']
        except:
            pass

    # save to loc
    with open(save_loc, 'w', encoding='utf-8') as f:
        json.dump(page_info, f, ensure_ascii=False, indent=4)
