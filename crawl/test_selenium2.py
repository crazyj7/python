
'''
가짜 브라우저 방어 기능을 우회하는 방안.
브라우저 정보 설정.
'''

from selenium import webdriver

# TEST_URL = 'https://intoli.com/blog/making-chrome-headless-undetectable/chrome-headless-test.html'
TEST_URL = 'https://www.naver.com'

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")
options.add_argument("lang=ko_KR") # 한국어!
driver = webdriver.Chrome('chromedriver', chrome_options=options)

contents = driver.get(TEST_URL)

driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: function() {return[1, 2, 3, 4, 5]}})")
driver.execute_script("Object.defineProperty(navigator, 'languages', {get: function() {return ['ko-KR', 'ko']}})")
driver.execute_script("const getParameter = WebGLRenderingContext.getParameter;WebGLRenderingContext.prototype.getParameter = function(parameter) {if (parameter === 37445) {return 'NVIDIA Corporation'} if (parameter === 37446) {return 'NVIDIA GeForce GTX 980 Ti OpenGL Engine';}return getParameter(parameter);};")

user_agent = driver.find_element_by_css_selector('#user-agent').text
plugins_length = driver.find_element_by_css_selector('#plugins-length').text
languages = driver.find_element_by_css_selector('#languages').text
webgl_vendor = driver.find_element_by_css_selector('#webgl-vendor').text
webgl_renderer = driver.find_element_by_css_selector('#webgl-renderer').text

print('User-Agent: ', user_agent)
print('Plugin length: ', plugins_length)
print('languages: ', languages)
print('WebGL Vendor: ', webgl_vendor)
print('WebGL Renderer: ', webgl_renderer)

print(contents)

driver.quit()