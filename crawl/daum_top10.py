from selenium import webdriver
from bs4 import BeautifulSoup

'''
네이버  실시간 탑10 뉴스  
'''

url = "https://m.daum.net"

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

browser.find_element_by_xpath('//*[@id="mAside"]/div[1]/div[1]/strong/a').click()
browser.save_screenshot("web2.png")

html = browser.page_source
soup = BeautifulSoup(html, 'html.parser')
notices = soup.select('div.realtime_layer ol.list_issue #hotissue ')

for n in notices:
    print(n)
    if n.select_one('.num_issue')!=None:
        print(n.select_one('.num_issue').text)
        print(n.select_one('.txt_issue').text)
        print('href=',n.a['href'])

browser.quit()

