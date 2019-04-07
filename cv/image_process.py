__author__ = 'crazyj'


import numpy as np
import cv2
import sys, getopt
import json

# usage()
#

class ImageProcess:

    @staticmethod
    def usage():
        print('program] imagefile cmd options...')
        print(' cmd=info')
        print(' cmd=resize options= -x 800 -y 600 -o output.png')

    def __init__(self, filename):
        self.filename =filename
        self.img=None
        self.cx=0
        self.cy=0
        self.color=0    # byte unit. color*8 = color bits
        self.getInfo()

    def printInfo(self):
        attrs = vars(self)
        print( ', '.join("%s:%s"%item for item in attrs.items()))

    def getInfo(self):
        self.img = cv2.imread(self.filename, cv2.IMREAD_COLOR)
        self.cy, self.cx, self.color = self.img.shape[:3]


if __name__=='__main__':

    if True:    # test
        argv = ['image_process.py', 'image/alphabet.jpg', 'info']

    else:
        argv = sys.argv


    if len(argv)<3:
        ImageProcess.usage()
        sys.exit()

    filename = argv[1]
    cmd = argv[2]

    ip = ImageProcess(filename)
    ip.getInfo()
    if cmd=='info':
        # ip.printInfo()
        result = { 'cx':ip.cx, 'cy':ip.cy, 'color':ip.color, 'filename':ip.filename }
        print( json.dumps(result, indent=4) )
        sys.exit()

    if cmd=='resize':
        try:
            opts, args = getopt.getotp(argv[3:], "x:y:o:")
        except getopt.GetoptError:
            ImageProcess.usage()
            sys.exit()

        for opt, arg in opts:
            if opt=='-x':
                resize_x = arg
            elif opt=='-y':
                resize_y = arg
            elif opt=='-o':
                outputfilename = arg

        print('x=', resize_x, 'y=', resize_y, 'output=', outputfilename)



