'''
daum 기사 수집
'''
import urllib.request
import urllib3
from bs4 import BeautifulSoup


url = 'https://m.media.daum.net/m/media/economic'

conn = urllib.request.urlopen(url)
# art_eco = conn.read()
soup = BeautifulSoup(conn, "html.parser")

print(soup)

print('-'*80)

arta = soup.select('.list_timenews a')
print(arta)




