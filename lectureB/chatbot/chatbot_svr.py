'''
챗봇 서버
>pip install python-telegram-bot --upgrade
pip install emoji --upgrade

@botfather
'''



from telegram.ext import Updater, MessageHandler, Filters

from emoji import emojize


from daum_top10 import getTop10Daum
from weather01 import getWeather
from news_realtime_daum import get_newsrealtime_daum
from misemise import getmise



apikey = '848818498:AAGvUjAKHoB4Qnl6AynzKeLpJNT6ed2dLkE'

updater = Updater(token=apikey)
dispatcher = updater.dispatcher
updater.start_polling()
mybot = None
myupdate = None

def weather():
    msg = getWeather()
    return msg

def mise():
    items = getmise()
    msg = ''
    for item in items:
        msg+='{} {}\n'.format(item['day'], item['content'])
    return msg

def top10():
    top10daum = getTop10Daum()
    msg=''
    for iss in top10daum:
        msg += "{}위 {} ".format(iss['rank'], iss['title'])
    return msg
def news():
    items = get_newsrealtime_daum()
    msg=''
    for item in items:
        msg += '-<a href="{}">{}</a>\n'.format(item['url'], item['title'])
    return msg

def send(msg):
    mybot.send_message(chat_id=myupdate.message.chat_id, text=msg)

def send_html(msg):
    mybot.send_message(chat_id=myupdate.message.chat_id, text=msg, parse_mode='HTML')

def send_photo(imgpath):
    mybot.send_photo(chat_id=myupdate.message.chat_id, photo=open(imgpath, 'rb'))

def handler(bot, update):
    global mybot, myupdate

    mybot = bot
    myupdate = update

    text = update.message.text
    chat_id = update.message.chat_id
    print('recved:', text)

    if text=='?' or text=='도움':
        send('이런거 쳐봐. 모해/아잉/사진 실검/날씨/미세먼지/뉴스...')

    elif '모해' in text:
        send('오빠 생각 ㅎㅎ')
    elif '아잉' in text:
        send(emojize('아잉:heart_eyes:', use_aliases=True))

    elif '사진' in text:
        send_photo('image.jpg')

    elif '검색어' in text or \
         '이슈' in text or \
         '실검' in text :
        send('지금 실시간 검색어는...잠시만...')
        send(top10())

    elif '날씨' in text:
        send(weather())

    elif '미세먼지' in text:
        send(mise())

    elif '뉴스' in text:
        send('지금 최신 뉴스는... 잠시만...')
        send_html(news())

    else:
        send('몰라')

echo_handler = MessageHandler(Filters.text, handler)
dispatcher.add_handler(echo_handler)

