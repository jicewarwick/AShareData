# -*- coding: UTF-8 -*-

import json

import pandas as pd
import requests

from . import DataS

classdef CNInfoData()

####用于获取token
def gettoken(client_id, client_secret):
    url = 'http://webapi.cninfo.com.cn/api-cloud-platform/oauth2/token'
    post_data = "grant_type=client_credentials&client_id=%s&client_secret=%s" % (client_id, client_secret)
    post_data = {"grant_type": "client_credentials",
                 "client_id": client_id,
                 "client_secret": client_secret
                 }
    req = requests.post(url, data=post_data)
    tokendic = json.loads(req.text)
    return tokendic['access_token']


token = gettoken('MDf8LphLg776MTi7GJljmLEtssOOnS3K',
                 'fbq7X8G3tOsks3k3Jy0v9Z3pjxixEax0')  ##请在平台注册后并填入个人中心-我的凭证中的Access Key，Access Secret
url = 'http://webapi.cninfo.com.cn/api/stock/p_stock2300?&access_token=' + token + '&scode=000002&edate=20180306'

response = requests.get(url)
response.text

result = json.loads(response.text)

data = pd.DataFrame(result['records'])

for i in range(len(result['records'])):
    print(result['records'][i]['PARENTCODE'], result['records'][i]['SORTCODE'], result['records'][i]['SORTNAME'],
          result['records'][i]['F002V'])
