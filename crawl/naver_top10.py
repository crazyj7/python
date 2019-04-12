from selenium import webdriver
from bs4 import BeautifulSoup

'''
네이버  실시간 탑10 뉴스  
'''

url = "https://m.naver.com"

# browser = webdriver.PhantomJS()
# browser.implicitly_wait(3)

options = webdriver.ChromeOptions()
options.add_argument('headless')
# browser = webdriver.Chrome(options=options)
# browser = webdriver.Chrome()
browser = webdriver.Chrome(chrome_options=options)
browser.implicitly_wait(3)
browser.get(url)
browser.save_screenshot("web1.png")

btn = browser.find_element_by_class_name("lm_btn_ok")
if btn!=None:
    btn.click()
browser.save_screenshot("web2.png")

url = 'https://m.naver.com/naverapp/?cmd=onMenu&version=3&menuCode=DATA'
browser.get(url)
browser.save_screenshot("web3.png")

html = browser.page_source
soup = BeautifulSoup(html, 'html.parser')
notices = soup.select('div#RTK_LIST li.cd_item')

for n in notices:
    print(n.select_one('.cd_num').text)
    print(n.select_one('.cd_t').text)
    #<span class="cd_t">방탄 컴백</span>

    print('href=',n.a['href'])


browser.quit()

