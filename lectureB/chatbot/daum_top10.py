from selenium import webdriver
from bs4 import BeautifulSoup

'''
다음 실시간 탑10 검색어    
'''

'''
return ; [ { title : ..., url: ... } ... ]
'''
def getTop10Daum():
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

    # mAside > div.head_issue > div.roll_issue.\#searchrank\#rolling > strong > a

    # browser.find_element_by_xpath('//*[@id="mAside"]/div[1]/div[1]/strong/a').click()
    browser.find_element_by_css_selector('div.roll_issue.\#searchrank\#rolling > strong > a').click()
    browser.save_screenshot("web2.png")

    html = browser.page_source
    soup = BeautifulSoup(html, 'html.parser')
    # print(soup)
    notices = soup.select('div.realtime_layer div.panel')

    resultlist = []

    for n in notices:
        # print ('aria-hidden-', n['aria-hidden'])
        if n['aria-hidden']=='false':
            lis = n.select('li')
            for l in lis:
                result = dict()
                result['rank'] = l.select_one('.num_issue').text
                result['title']= l.select_one('.txt_issue').text
                result['url'] = l.a['href']
                # print(l.select_one('.num_issue').text)
                # print(l.select_one('.txt_issue').text)
                # print('href=',l.a['href'])
                resultlist.append(result)
    browser.quit()

    # print(resultlist)
    return resultlist


if __name__ == '__main__':
    items = getTop10Daum()
    for it in items:
        print(it['rank'], it['title'], it['url'])