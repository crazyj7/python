"""네이버 뉴스 기사 웹 크롤러 모듈
분야별 기사를 수집.
"""



from bs4 import BeautifulSoup
import urllib.request
import time



# 출력 파일 명
OUTPUT_FILE_NAME = 'naver_article.txt'
# 긁어 올 URL
URL = 'http://news.naver.com'
sleeptime=5

def get_article(URL):
    print("article URL=", URL)
    source_code_from_URL = urllib.request.urlopen(URL)
    soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
    print('soup=', soup)

    text=''
    for item in soup.find_all('div', id='articleBodyContents'):
        ac=item.text
        # remove datas
        if ac.find("</script>")>0 :
            ac = ac[ac.find("</script>")+len("</script>"):]
        ac = ac.replace("function _flash_removeCallback() {}", "")
        ac = ac.replace("// flash 오류를 우회하기 위한 함수 추가", "")
        ac = ac.strip()
        print('item=', ac)
        text=text+ac
    # print(text)
    return text


# 크롤링 함수
def get_text(URL):
    print("URL=", URL)
    source_code_from_URL = urllib.request.urlopen(URL)
    soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
    # print('soup=', soup)
    text = ''

    # old for item in soup.find_all('div', id='articleBodyContents'):


    # 헤드라인 뉴스
    ids = ["hdline_article_tit"]
    for tid in ids:
        for item in soup.find_all('div', tid):
            # print('item=', item)
            atag = item.find('a')
            print('href=', atag['href'])
            print(atag.text)
            text += "\n"+atag.text +"\n"
            arttext= get_article(URL+atag['href'])
            text += str(arttext)

    # 주제 클래스. 변경될 수 있음.
    sel=['section_politics', 'section_economy',
         'section_society', 'section_life',
         'section_world', 'section_it']
    for tid in sel:
        item = soup.select_one("div#"+tid+" .com_list")
        # print('item=', item)
        atags = item.find_all('a')
        for atag in atags:
            print('href=', atag['href'])
            print(atag.text)
            text += "\nTITLE: "+atag.text+"\n"

            time.sleep(sleeptime)
            arttext= get_article(atag['href'])
            text += str(arttext)

    return text


# 메인 함수
def main():
    open_output_file = open(OUTPUT_FILE_NAME, 'w')
    result_text = get_text(URL)
    open_output_file.write(result_text)
    open_output_file.close()


if __name__ == '__main__':
    t1=time.time()
    main()
    t2 = time.time()
    print('elapsed = ', t2-t1)

