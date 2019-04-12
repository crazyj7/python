
import codecs
import urllib.request
from bs4 import BeautifulSoup
import json

'''
네이버 동네 날씨.
길음1동 현재날씨 추출. 2019.04
'''

file='weather01.txt'

# url = 'https://weather.naver.com'
# url='https://weather.naver.com/flash/naverRgnTownFcast.nhn?m=jsonResult&naverRgnCd=09290660'

url = 'https://weather.naver.com/json/crntWetrDetail.nhn?_callback=window.__jindo2_callback._284&naverRgnCd=09290660'

response = urllib.request.urlopen(url)
data = response.read()
data2 = codecs.utf_8_decode(data)
print('respose=', data2)
# for item in data2:
#     print('item=', item)
data3 = data2[0]

# [ ...  ] substring
data4 = data3[ data3.find('['): data3.rfind(']')+1 ]
print(data4)

# soup = BeautifulSoup(response, "html.parser")
# print('soup=', soup)

# json_data = json.loads(data2[0])
# print(json_data)

# print(json_data['todayColCnt'])
# townWetrs = json_data['townWetrs']
# print(townWetrs[0])

json_data = json.loads(data4)
print(json_data[0])
print(json_data[1])
print( json_data[1][1] )

print( json_data[1][1]['mareaNm'] )
print( json_data[1][1]['sareaNm'] )
print( '온도:', json_data[1][1]['tmpr'] )
print( '강수량:', json_data[1][1]['rainAmt'] )
print( '날씨:', json_data[1][1]['wetrTxt'] )


