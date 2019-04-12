from selenium import webdriver

url = "https://m.naver.com"
# url = "https://www.daum.net"

# browser = webdriver.PhantomJS()
# browser.implicitly_wait(3)

options = webdriver.ChromeOptions()
options.add_argument('headless')
browser = webdriver.Chrome(options=options)
browser.implicitly_wait(3)

browser.get(url)
browser.save_screenshot("web.png")

trank = browser.find_element_by_class_name(".cd_t")
print(trank)

browser.quit()

