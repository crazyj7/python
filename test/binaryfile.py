import sys
import struct

filename='binfile.bin'

print(chr(65), ord('A'))

# binary file write
try:
    fp = open(filename, "wb")
    datas=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    for num in datas:
        # 0x01 0x02 ...
        fp.write(bytearray(chr(num), 'ascii'))
    for num in datas:
        # 00 00 00 01, 00 00 00 02
        fp.write(struct.pack('i', num))
    fp.write(bytearray('text', 'utf-8'))
    fp.write(bytearray('text', 'ascii'))
    fp.close()
except IOError:
    print('file open failed')

with open(filename, "rb") as fp:
    for i in range(15):
        b=fp.read(1)
        print(b, end=' ')
    print()
    for i in range(15):
        b = fp.read(4)
        print(b, end=' ')
        print(struct.unpack('i', b))
    b=fp.read(4)
    print(str(b, 'utf-8'))
    b=fp.read(4)
    print(str(b, 'ascii'))
b=bytearray([1,0,0,0])
print('little endian=', struct.unpack('i', b))
print('bigendian parse=', struct.unpack('>i', b))
