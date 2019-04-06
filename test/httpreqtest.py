import requests


session = requests.Session()
data = session.get("https://pypi.python.org/simple/").content
print(data)




