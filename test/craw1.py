
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

# 날씨 가져오기...

url="https://weather.naver.com/"


# html = urlopen(url).read()
# or
req = Request(url)
req.add_header('User-Agent', 'Mozilla/5.0')
html = urlopen(req).read()


# print('html=',html)   # 많이 출력되어 검색을 해야 함.

soup = BeautifulSoup(html, 'html.parser')

# a 태그를 2개만 가지고 온다.
tagas=soup.find_all('a',limit=2)
print('atags (find_all(a))=', tagas)    # [ <a href=...>...</a>,  <a href=...</a> ]
print('a tag first atgas[0]=', tagas[0])
print('href attrib. ([href])=', tagas[0]['href'])
print('between a tag. only string (.string)=', tagas[0].string)
print('between a tag. only string (.get_text())=', tagas[0].get_text())
print('a tag contents (.contents)=', tagas[0].contents)

# 모든 테이블 태그 검색. 리턴은 list
tbl = soup.find_all('table')
# print('tbl[1]=', tbl[1])  # long data...
# table search :     ptag('table')[0]('tr')[0]('td')[0]

# 두 번째 테이블에서 하위에 tr 태그의 첫번째 조회
wtr = tbl[1]('tr')[0]
print ( 'tbl[1](tr)[0]=', wtr )

# tr태그에서 하위에 첫번째 td 조회
print ('wtr(td)[0]=', wtr('td')[0])

# 첫번째 태그만 가져올때는 .으로 태그명으로 계속 접근한다.
print('wtr.td=', wtr.td)
print('wtr.td.img=', wtr.td.img)

# 중복된 태그들의 순서를 지정할때는 (태그명)으로 접근하여(list) 순서 인덱스를 지정한다.
print ('wtr(td)[1]=', wtr('td')[1])

# 태그명과 클래스 속성으로 내부 검색하여 찾을 때는 select를 쓴다. 리턴결과는 list.
print ('get info class td=', wtr.select('td[class=info]'))

# 내부의 텍스트를 list로 가져올때. contents 사용.
print ('get info class td=', wtr.select('td[class=info]')[0].contents[0])


# 찾고 싶은 부분을 소스보기를 통해 ^+Shift+I (inspect 검사) 어느 태그부분인지 확인.
#ex) copy selector:
#     #content > div.m_zone1 > table > tbody > tr.now > td.info
# select로 검색. list.
current_weather = soup.select("#content > div.m_zone1 > table > tbody > tr.now > td.info")
print('current_weather=', current_weather)
print('current_weather[0].string=', current_weather[0].string)  # None
print('contents=', current_weather[0].contents)
print('contents[0]=', current_weather[0].contents[0])



soup.clear()

