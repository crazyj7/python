from selenium import webdriver
from bs4 import BeautifulSoup

'''
다음 실시간 탑10 뉴스  
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
# print(soup)
notices = soup.select('div.realtime_layer div.panel')

for n in notices:
    # print ('aria-hidden-', n['aria-hidden'])
    if n['aria-hidden']=='false':
        lis = n.select('li')
        for l in lis:
            print(l.select_one('.num_issue').text)
            print(l.select_one('.txt_issue').text)
            print('href=',l.a['href'])

browser.quit()

