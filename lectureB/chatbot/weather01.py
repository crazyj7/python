
import codecs
import urllib.request
from bs4 import BeautifulSoup
import json

'''
네이버 동네 날씨.
길음1동 현재날씨 추출. 2019.04
'''

def getWeather():
    # url = 'https://weather.naver.com'
    # url='https://weather.naver.com/flash/naverRgnTownFcast.nhn?m=jsonResult&naverRgnCd=09290660'
    url = 'https://weather.naver.com/json/crntWetrDetail.nhn?_callback=window.__jindo2_callback._284&naverRgnCd=09290660'

    response = urllib.request.urlopen(url)
    data = response.read()
    data2 = codecs.utf_8_decode(data)
    # print('respose=', data2)
    data3 = data2[0]

    # [ ...  ] substring
    data4 = data3[ data3.find('['): data3.rfind(']')+1 ]
    # print(data4)

    json_data = json.loads(data4)
    # print(json_data[0])
    # print(json_data[1])
    # print( json_data[1][1] )

    loc1 = json_data[1][1]['mareaNm']
    loc2 = json_data[1][1]['sareaNm']
    degree = json_data[1][1]['tmpr']
    water =  json_data[1][1]['rainAmt']
    w = json_data[1][1]['wetrTxt']

    msg = "지금 {} {} 날씨는 {} 이고 온도는 {}도 강수량은 {} 이야.".format(loc1, loc2,
                                                         w, degree, water)
    return msg

# print(getWeather())


