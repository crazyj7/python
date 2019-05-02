
from telegram.ext import Updater, MessageHandler, Filters

from emoji import emojize

apikey = '848818498:AAGvUjAKHoB4Qnl6AynzKeLpJNT6ed2dLkE'

updater = Updater(token=apikey)
dispatcher = updater.dispatcher
updater.start_polling()

def handler(bot, update):
    text = update.message.text
    chat_id = update.message.chat_id

    if '모해' in text:
        bot.send_message(chat_id=chat_id, text='오빠 생각 ㅎㅎ')
    elif '아잉' in text:
        bot.send_message(chat_id=chat_id, text=emojize('아잉:heart_eyes:', use_aliases=True))
    elif '사진' in text:
        bot.send_photo(chat_id=chat_id, photo=open('photo.jpg', 'rb'))
    else:
        bot.send_message(chat_id=chat_id, text='몰라')

echo_handler = MessageHandler(Filters.text, handler)
dispatcher.add_handler(echo_handler)

