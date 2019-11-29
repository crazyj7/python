'''
starbucks wifi auto connect
'''

from selenium import webdriver
from bs4 import BeautifulSoup


def getBrowser(target_url):
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")
    options.add_argument("lang=ko_KR")  # 한국어!
    browser = webdriver.Chrome(chrome_options=options)
    contents = browser.get(target_url)
    browser.execute_script("Object.defineProperty(navigator, 'plugins', {get: function() {return[1, 2, 3, 4, 5]}})")
    browser.execute_script("Object.defineProperty(navigator, 'languages', {get: function() {return ['ko-KR', 'ko']}})")
    browser.execute_script(
        """const getParameter = WebGLRenderingContext.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) 
        {if (parameter === 37445) {return 'NVIDIA Corporation'} 
        if (parameter === 37446) {return 'NVIDIA GeForce GTX 980 Ti OpenGL Engine';}
        return getParameter(parameter);
        };""")
    return browser


url = 'https://first.wifi.olleh.com/starbucks/index_kr.html'
# https://first.wifi.olleh.com/starbucks/index_kr.html

browser = getBrowser(url)

# 아이디/비밀번호를 입력해준다.
# driver.find_element_by_name('id').send_keys('naver_id')
# driver.find_element_by_name('pw').send_keys('mypassword1234')
# 로그인 버튼을 눌러주자.
# driver.find_element_by_xpath('//*[@id="frmNIDLogin"]/fieldset/input').click()
browser.save_screenshot('web01.png')

# wait
browser.implicitly_wait(2)

# browser.find_element_by_xpath('//*[@id="agreement_agree"]').click()

# selid = '#agreement_agree'
# #agreement_agree   by css selector ; result ; element not interactable.
# input 태그가 안되고 label 태그가 클릭이 된다.
selid = '#contents > div.con_agree > div.con_box > fieldset > div:nth-child(1) > div.checkbox.agreement > label'
browser.find_element_by_css_selector(selid).click()


# browser.find_element_by_xpath('//*[@id="purpose_agree"]').click()
# selid = '#purpose_agree' : input tag // fail.
selid = '#contents > div.con_agree > div.con_box > fieldset > div:nth-child(2) > div.checkbox.purpose > label'
browser.find_element_by_css_selector(selid).click()

browser.save_screenshot('web02.png')
html = browser.page_source
print(html)
# //*[@id="contents"]/div[3]/div[2]/fieldset/div[3]/div[1]/a/span
# #contents > div.con_agree > div.con_box > fieldset > div.bottom_btn > div.btn > a > span
# selid = '#contents > div.con_agree > div.con_box > fieldset > div.bottom_btn > div.btn > a > span'
# browser.find_elements_by_css_selector(selid).click()

browser.execute_script('javascript:goAct()')

# wait
browser.implicitly_wait(2)
html = browser.page_source
print(html)
browser.save_screenshot('web03.png')



