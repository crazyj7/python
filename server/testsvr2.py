'''
test api server
'''

from flask import Flask, request
from flask_restful import Resource, Api
from flask.views import MethodView

import logging
import json,base64

import os, sys
import datetime
import threading, time

app = Flask(__name__)

# log = logging.basicConfig(filename='testsvr.log', level=logging.INFO)
# 로깅을 전부 끄기
log = logging.getLogger('werkzeug')
log.disabled = True
app.logger.disabled = True

'''
/apitest1
'''
@app.route('/apitest1', methods=['GET'])
def apitest1_get():
    data = request.args
    print('recv:', data)  # dictionary
    abc = data.get('abc')
    if abc :
        result = 'This is GET method!'+str(abc)
    else:
        result = 'Hello World! input abc'
    return result

@app.route('/apitest1', methods=['POST'])
def apitest1_post():
    # get과 동일하게 작동
    return apitest1_get()


'''
/sum
'''
@app.route('/sum', methods=['GET'])
def sum_get():
    data = request.args
    print('recv:', data)  # dictionary
    return 'Hello.'

@app.route('/sum', methods=['POST'])
def sum_post():
    logging.info('sum test')
    # print('request.data=', request.data)  # binary data read all
    data=request.get_json(force=True) # parse json string
    print('request.json=', data)
    a = data['a']
    b = data['b']
    now = datetime.datetime.now()
    print(now)
    timestr = now.strftime('%Y%m%d %H:%M:%S')
    result = {
        'sum':int(a+b),
        'time':timestr
    }
    logging.info('result='+json.dumps(result))
    return result

port = 18899
if __name__=='__main__':
    print('Start Server... port=', port)
    logging.info('start server')
    app.run(host='0.0.0.0', port=port, debug=False)
    # 디버그 모드로 하면 소스 수정시 자동으로 서버 재시작이 된다.

