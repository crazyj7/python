from selenium import webdriver


url = "https://m.naver.com"
# url = "https://www.daum.net"

# browser = webdriver.PhantomJS()
# browser.implicitly_wait(3)

options = webdriver.ChromeOptions()
options.add_argument('headless')
# browser = webdriver.Chrome(options=options)
# browser = webdriver.Chrome()
browser = webdriver.Chrome(chrome_options=options)
browser.implicitly_wait(3)

browser.get(url)

btn = browser.find_element_by_class_name("lm_btn_ok")
btn.click()

browser.save_screenshot("web.png")


browser.quit()

