from bs4 import BeautifulSoup

html = '''
<html>
<body>
    <table name="t1" id="t1" class="cls1">
        <tr style="st1"> <td>aaaa<b>bbb</b>ccc</td> <td>xxx</td> </tr>
        <tr> <td id='id2'> aaaa </td> </tr>
        <tr> <td class="big"> aaaa </td> </tr>
    </table>
    <table>
        <tr><td>xxx<span>hi<b>there</b></span></td><td>bb</td></tr>
        <tr><td>aaa</td><td>bb<span>hi</span></td></tr>
    </table>
</body>
</html>

'''

soup = BeautifulSoup(html, "html.parser")
print('soup=', soup)        # soup 객체를 그대로 쓸 경우, 파싱된 스트링
print('soup.table=', soup.table)    # 자손 중 table 태그 파트
print('soup.tr=', soup.tr)      # 자손 중 tr 태그 파트 (가장 먼저 나온 태그). (바로 자식이 아니라 자손이어도 가능)
print('soup.tr.td=', soup.tr.td )
print('soup.tr.td.string=', soup.tr.td.string)            # td안에 태그가 있으면 안 됨.
print('soup.tr.td.b.string=', soup.tr.td.b.string)            # b 태그 내의 스트링.
print('soup.tr.td.text=', soup.tr.td.text )            # td 태그 내의 스트링. (내부의 태그는 제거하고 스트링만)
print('soup.tr.td.get_text()', soup.tr.td.get_text())        # td 태그 내의 스트링. 상동
print('soup.tr.b.text=', soup.tr.b.text )            # tr의 자손 b태그 내의 스트링.

# 속성
print( 'soup.tr["style"]=', soup.tr["style"])            # 속성을 가져올 때는 array []을 사용.

# LIST
print( 'soup.table("td")=', soup.table("td") )       #LIST: table내의 자손 태그중 td를 모두 찾는다.
# LIST
print( 'soup.tr("td")=', soup.tr("td") )              #LIST: 하위 태그를 가져올 때 태그명을 바로 쓰면 처음나온 1개이지만,
                                    # ()를 사용하면 여러개를 가져올 수 있음.
print( 'soup.tr("td")[1]=', soup.tr("td")[1] )           # 두 번째 td 태그를 가져옴.

##############################################
print('soup.select_one("#id2")=', soup.select_one("#id2") )
print('soup.select_one("td.big")=', soup.select_one("td.big") )
# LIST
print('soup.select("#id2")=', soup.select("#id2") )
# LIST
print('soup.find_all("td") ')
tds = soup.find_all("td")
for t in tds:
    print(t)

###################################
# string search and next tag part ?
# HTML 내부에서 태그, id, 클래스 등으로 찾기 어려운 경우, 보통은 복수개로 나와서 n번째를 찾아 가면 되는데
# 만약 위치가 변하고 특정 스트링이 항상 앞에 있다면, 스트링을 찾아서 다시 파싱하는 게 좋을 수도 있다.
#
pox = html.find("xxx")
pox = html.find("<span>", pox)   # find start tag
poe = html.find("</span>", pox)+7   # find end tag
parthtml = html[pox:poe]
print('part html=', parthtml)
soup2 = BeautifulSoup(parthtml, "html.parser")
print(soup2)
print(soup2.b.text)


