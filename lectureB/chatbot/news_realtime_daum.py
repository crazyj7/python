'''
다음 실시간 뉴스.

'title': title
'url': url

'''

from bs4 import BeautifulSoup
import urllib.request
import time

def get_newsrealtime_daum():
    # url = 'https://m.media.daum.net/m/media/'
    url = 'https://media.daum.net/'

    conn = urllib.request.urlopen(url)
    # art_eco = conn.read()
    soup = BeautifulSoup(conn, "html.parser")

    # print(soup)
    # print('-'*80)

    # rb = soup.find(class_='box_realtime')
    # print(rb)
    # lis = soup.select("div.box_realtime li")
    # lis = soup.find_all("a", class_="link_news")

    headline = soup.find("div", class_="box_headline")
    lis = headline.find_all("a", class_="link_txt")
    # print(lis)
    realnews = []
    for d in lis:
        item = dict()
        item['title']=d.text.strip()
        item['url']=d["href"]
        realnews.append(item)
    return realnews

if __name__ == '__main__':
    realnews = get_newsrealtime_daum()
    for i in realnews:
        print( i['title'] , i['url'])

'''
실시간 뉴스 제목 
 #kakaoContent > div > div.box_g.box_realtime > ul > li:nth-child(1) > a > div.cont_thumb > strong > span.txt_g
 #kakaoContent > div > div.box_g.box_realtime > ul > li:nth-child(7) > a > div.cont_thumb > strong > span.txt_g
'''


