
'''
미녀검객
beautyswordsman_bot

>pip install python-telegram-bot --upgrade
pip install emoji --upgrade

@botfather

'''

import telegram

apikey = '848818498:AAGvUjAKHoB4Qnl6AynzKeLpJNT6ed2dLkE'

bot = telegram.Bot(token=apikey)

# chat_id = bot.get_updates()[-1].message.chat_id     # 챗봇에 메시지를 보낸 다음에 실행해야 함. 마지막 메시지의 챗ID
# print(chat_id)    # 저장함.

chat_id = '30408279'

bot.send_message(chat_id=chat_id, text='방가방가')







