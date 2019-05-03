'''
미세미세 미세 먼저 정보
'''

from bs4 import BeautifulSoup
import urllib.request


def getmise():
    url = 'https://www.airkorea.or.kr/web/dustForecast?pMENU_NO=113'

    conn = urllib.request.urlopen(url)
    html = conn.read()

    soup = BeautifulSoup(html, 'lxml')

    dllist = soup.select("dl.forecast")

    # today
    items=[]
    for i in range(2):
        item=dict()
        item['day']=dllist[i].select_one('dt').text
        item['content'] = dllist[i].select_one('.txtbox').text
        items.append(item)  # today, tomorrow
    return items

if __name__ == '__main__':
    items = getmise()
    for item in items:
        print(item['day'], item['content'])




