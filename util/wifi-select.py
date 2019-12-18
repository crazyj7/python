import os, sys
import subprocess

# utf8이라고 나오긴 하는데...
print('stdout encoding=', sys.stdout.encoding)


cmd = 'netsh wlan show profile'

# 실제 실행하면 cmd의 디폴트인 cp949 (euc-kr)로 출력된다. 인코딩 주의.
# result = subprocess.check_output(cmd, encoding='utf8', shell=True)
result = subprocess.check_output(cmd)

# out = result.decode('ascii', 'ignore') # 한글제외하고 받음.
# out = result.decode('UTF-8', 'ignore')
# out = result.decode('cp949', 'ignore')
out = result.decode('euc-kr', 'ignore')
print(out)

# connect
wifi = 'KT_starbucks'
cmd = 'netsh wlan connect ssid="{}" name="{}"'.format(wifi, wifi)
print(cmd)
result = subprocess.check_output(cmd)
out = result.decode('euc-kr', 'ignore')
print(out)



