
# import modules
import socket
import struct
import binascii
import os
import pye

# if operating system is windows
if os.name == "nt":
    print('windows')
    s = socket.socket(socket.AF_INET,socket.SOCK_RAW,socket.IPPROTO_IP)
    s.bind(("YOUR_INTERFACE_IP",0))
    s.setsockopt(socket.IPPROTO_IP,socket.IP_HDRINCL,1)
    s.ioctl(socket.SIO_RCVALL,socket.RCVALL_ON)

# if operating system is linux
else:
    print('linux')
    s=socket.socket(socket.PF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0800))

input('ready... press enter.')

# create loop
while True:
    # Capture packets from network
    pkt=s.recvfrom(65565)
    # extract packets with the help of pye.unpack class
    unpack=pye.unpack()

    # ethernet header ; 14
    # print ("\n\n===&gt;&gt; [+] ------------ Ethernet Header----- [+]")
    packetsize = len(pkt[0])
    # for i in unpack.eth_header(pkt[0][0:14]).items():
    #     a,b=i
    #     print ("{} : {} | ".format(a,b))

    # ip header ; 40
    if packetsize>40:
        ipheader = unpack.ip_header(pkt[0][14:34])

        protocol = ipheader['Protocol']
        # http://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml
        # TCP=6, UDP=17, ICMP=1
        if protocol==6 :
            tcpheader = unpack.tcp_header(pkt[0][34:54])

            srcip = ipheader['Source Address']
            srcport = tcpheader['Source Port']
            dstip = ipheader['Destination Address']
            dstport = tcpheader['Destination Port']

            if dstport==2222 or dstport==5910:
                continue

            # fileter :  dstport=80 or dstport=443
            # if dstport!=80 and dstport!=443:
            #     continue

            ## cannot capture outgoing dstport 80/443
            if dstport==80 or dstport==443 or \
                srcport==80 or srcport==443 :
                pass
            else:
                continue

            offset = tcpheader['Offset & Reserved'] # word
            offset = offset >> 4  # get only high 4 bits .
            offset = offset * 4   # word to bytes .
            offset += 34 # add TCP Header Start Position.

            datasize = packetsize - offset

            if datasize==0:
                continue

            print("packetsize=", packetsize, "datasize=", datasize)
            print ('{}:{} -> {}:{}'.format(srcip, srcport, dstip, dstport))

            # prevent too much print!!!!!!
            # view only front part....
            maxprintsize = 100
            if datasize > maxprintsize:
                data = pkt[0][offset:offset+maxprintsize]
            else:
                data = pkt[0][offset:]

            # hexa print
            print(binascii.b2a_hex(data))
            try :
                print(data.decode('utf8'))
            except:
                print('decode failed.')

    #
    # print ("\n===&gt;&gt; [+] ------------ Tcp Header ----------- [+]")
    # for  i in unpack.tcp_header(pkt[0][34:54]).items():
    #     a,b=i
    #     print ("{} : {} | ".format(a,b))

