
from telegram.ext import Updater, MessageHandler, Filters

from emoji import emojize
from daum_top10 import getTop10Daum
from weather01 import getWeather

apikey = '848818498:AAGvUjAKHoB4Qnl6AynzKeLpJNT6ed2dLkE'

updater = Updater(token=apikey)
dispatcher = updater.dispatcher
updater.start_polling()


def weather():
    msg = getWeather()
    return msg

def top10():
    top10daum = getTop10Daum()
    msg=''
    for iss in top10daum:
        msg += "{}위 {} ".format(iss['order'], iss['title'])
    return msg


def handler(bot, update):
    text = update.message.text
    chat_id = update.message.chat_id

    if '모해' in text:
        bot.send_message(chat_id=chat_id, text='오빠 생각 ㅎㅎ')
    elif '아잉' in text:
        bot.send_message(chat_id=chat_id, text=emojize('아잉:heart_eyes:', use_aliases=True))
    elif '사진' in text:
        bot.send_photo(chat_id=chat_id, photo=open('photo.jpg', 'rb'))
    elif '검색어' in text or '이슈' in text:
        bot.send_message(chat_id=chat_id, text='지금 실시간 이슈는...')
        bot.send_message(chat_id=chat_id, text=top10())
    elif '날씨' in text:
        bot.send_message(chat_id=chat_id, text=weather())
    else:
        bot.send_message(chat_id=chat_id, text='몰라')

echo_handler = MessageHandler(Filters.text, handler)
dispatcher.add_handler(echo_handler)

