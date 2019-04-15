'''
미세미세 미세 먼저 정보
'''

from bs4 import BeautifulSoup
import urllib.request


url = 'https://www.airkorea.or.kr/web/dustForecast?pMENU_NO=113'

conn = urllib.request.urlopen(url)
html = conn.read()

soup = BeautifulSoup(html, 'lxml')

dllist = soup.select("dl.forecast")

# today
for i in range(2):
    txt_dt =  dllist[i].select_one('dt').text
    txt_status = dllist[i].select_one('.txtbox').text
    print(txt_dt, txt_status)









