'''

pip install cryptoshop

# install botan

git clone https://github.com/randombit/botan.git
cd botan
python configure.py --cc=msvc


'''

from cryptoshop import encryptstring

plain = 'Hello World!'
passphase = '123456'
result = encryptstring(plain, passphase)
print('result=', result)

