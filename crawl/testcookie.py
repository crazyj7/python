import urllib
import urllib2

url = "http://www.suninatas.com/member/mem_action.asp"
login_form={"Hid":"아이디","Hpw":"암호"}
login_req=urllib.urlencode(login_form)
request=urllib2.Request(url,login_req)
response = urllib2.urlopen(request)
cookie = response.headers.get('Set-Cookie')

data = response.read()

print (cookie)


url2 = "http://suninatas.com/main/auth_check.asp"
request2 = urllib2.Request(url2)
request2.add_header('cookie',cookie)
response2 = urllib2.urlopen(request2)

data2 = response2.read()

print (data2)


