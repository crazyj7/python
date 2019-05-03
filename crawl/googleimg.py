import os
import urllib.request as req
import urllib
from urllib.parse import unquote

from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from bs4 import BeautifulSoup
import ast
import time
import random


# google image search url and query

# url='https://www.google.co.kr/search?newwindow=1&tbm=isch&source=hp&biw=1311&bih=678&ei=0D29XJnlPOzEmAXuz7PQCg&q=Bus&oq=Bus&gs_l=img.3..0l10.3162.3363..3471...0.0..1.90.269.3......2....1..gws-wiz-img.....0.3jhVlU2skVk'
# url = 'https://www.google.co.kr/search?newwindow=1&biw=818&bih=721&tbm=isch&sa=1&ei=1T29XPLdA622mAWKqrVI&q=bus&oq=bus&gs_l=img.3..35i39j0l9.5148168.5148408..5148774...0.0..0.99.289.3......1....1..gws-wiz-img.Rn68F__Pfz0'

url='https://www.google.co.kr/search?newwindow=1&biw=1432&bih=721&tbm=isch&sa=1&ei=8lG9XMPtIJvVmAXCtKL4Bg&q=tiger&oq=tiger&gs_l=img.3..0l10.1775869.1776611..1776781...0.0..0.100.471.4j1......1....1..gws-wiz-img.YU6N8rxVMQg'


savedir = 'images'

def save_images(urls, savedir, offset):
    cnt = len(urls)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for i, url in enumerate(urls):
        print(i+offset, url)
        extp = url.rfind('.')
        ext = "jpg"
        if extp>=0 :
            exttmp = url[extp+1:extp+4]
            if exttmp.upper() in ['JPG', 'PNG', 'GIF', 'BMP']:
                ext = exttmp

        savepath = os.path.join(savedir, '{:06}.{}'.format(i+offset, ext) )
        try:
            req.urlretrieve(url, savepath)
            print('download ok', i+offset, cnt,  url)
        except:
            print('fail ', i+offset, cnt,  url)
        # time.sleep(1)


# scroll down
SCROLL_PAUSE_TIME = 0.5

def scroll_down(browser):
    # Get scroll height
    last_height = browser.execute_script("return document.body.scrollHeight")
    i = 1
    while True:
        # Scroll down to bottom
        print('scroll down')
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME + random.random())

        browser.save_screenshot("web" + str(i) + ".png")
        i += 1
        # Calculate new scroll height and compare with last scroll height
        new_height = browser.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME + random.random())
            new_height = browser.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            else:
                print('try scroll')
        last_height = new_height

def scroll_down2(browser):
    body = browser.find_element_by_css_selector('body')
    for i in range(10):
        new_height = browser.execute_script("return document.body.scrollHeight")
        print('new_height=', new_height)
        clientHeight = browser.execute_script("return document.body.clientHeight")
        print('clientHeight=', clientHeight)
        print('scroll down')
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(SCROLL_PAUSE_TIME + random.random())

        browser.save_screenshot("web" + str(i) + ".png")





# browser = webdriver.PhantomJS()
# browser.implicitly_wait(3)

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('--kiosk')
options.add_argument('--start-maximized')
# browser = webdriver.Chrome(options=options)
# browser = webdriver.Chrome()
browser = webdriver.Chrome(chrome_options=options)
browser.implicitly_wait(3)
# browser.set_window_size(1024, 600)
browser.fullscreen_window()
browser.maximize_window()

browser.get(url)

browser.save_screenshot("web0.png")


scroll_down(browser)
# scroll_down2(browser)

html = browser.page_source
soup = BeautifulSoup(html, 'lxml')
# print(soup)

# my method
# search = soup.select_one('div#search')
# atags = search.select('a')
# for t in atags:
#     imgurl = t['href']
#     if imgurl=='#':
#         continue
#     tmp1 = imgurl.find('imgurl=')
#     tmp2 = imgurl.find('&')
#     imgurl = imgurl[tmp1+len('imgurl='):tmp2]
#     imgurl = unquote(imgurl)
#     print('url=', imgurl)


# internet
oururls=[]
urls = soup.findAll('div', {'class':'rg_meta notranslate'})
for i, url in enumerate(urls):
    theurl = url.text
    theurl = ast.literal_eval(theurl)['ou']
    print(i, theurl)
    oururls.append(theurl)


# save image files....
save_images(oururls, savedir, 0)

browser.close()

