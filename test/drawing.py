import tkinter
from PIL import Image, ImageDraw
import numpy as np
from time import time

import matplotlib.pyplot as plt
import scipy.interpolate as ip
from scipy.interpolate import splrep, splev, splprep
#import threading

class PaintBrush:

    @staticmethod
    def nparrs_to_str(nparrs):
        retstr=""
        for nparr in nparrs:
            retstr+="["
            for it in nparr:
                retstr+="("
                retstr+= "{},{},{}".format(it[0], it[1], it[2])
                retstr+=")"
            retstr+="]"
        return retstr

    @staticmethod
    def str_to_nparrs(strp):
        strp = strp.replace("[", "")
        segs = strp.split("]")
        lsDraw=[]
        for se in segs:
            se = se.replace("(", "")
            spts = se.split(")")
            lsPts=[]
            for pts in spts:
                it = pts.split(",")
                if len(it)>=3:
                    lsPts.append([it[0], it[1], it[2]])
            arPts = np.asarray(lsPts)
            if len(arPts)>0:
                lsDraw.append(arPts)
        return lsDraw

    def __init__(self, title, width=400, height=200):
        self.line_col = 0   # line color
        self.line_width = 5

        self.root = None
        self.frame = None

        # 화면 크기
        self.width = 400
        self.height = 200

        self.canvas = None
        self.cancel = False

        self.imagedraw = None
        self.start_flag = False  # timer start.
        self.start_time = 0

        # members
        self.image1 = None  # org image (400x200)

        # only content area lefttop(rx0,ry0), rightbottom(rx1,ry1)
        self.rx0 = 9999999
        self.rx1 = 0
        self.ry0 = 9999999
        self.ry1 = 0
        self.contentcx = 0  # content area (line width include)
        self.contextcy = 0  # content area (line width include)

        self.pb_sample_signs = []  # sample pts list (multi line)
        self.pb_sample_pts = []  # sample points list (one line)

        self.root = tkinter.Tk()
        self.root.title(title)

        self.width, self.height = width, height

        self.frame = tkinter.Frame(self.root, width=self.width, height=self.height)
        #frame.bind('<Key>', key)
        self.frame.pack()
        self.canvas = tkinter.Canvas(self.frame, width=self.width, height=self.height)


    def callback(self, event):
        # print ('Click-', event.x, event.y)
        pass

    def draw(self, event):
        # global x0, y0
        # global imagedraw
        # global rx0, rx1, ry0, ry1
        # global pb_sample_signs, pb_sample_pts

        self.rx0 = min([self.rx0, event.x])
        self.rx1 = max([self.rx1, event.x])
        self.ry0 = min([self.ry0, event.y])
        self.ry1 = max([self.ry1, event.y])
        self.canvas.create_line(self.x0, self.y0, event.x, event.y, width=self.line_width)
        self.imagedraw.line((self.x0, self.y0, event.x, event.y), fill=self.line_col, width=self.line_width)
        self.x0,self.y0 = event.x, event.y

        dt = int( (time()-self.start_time)*1000 )
        self.pb_sample_pts.append([self.x0,self.y0, dt])

    def down(self, event):
        # global x0, y0
        # global rx0, rx1, ry0, ry1
        # global pb_sample_signs, pb_sample_pts
        if self.start_flag==False:
            self.start_flag=True
            self.start_time=time()

        self.x0,self.y0 = event.x, event.y
        self.rx0 = min([self.rx0, event.x])
        self.rx1 = max([self.rx1, event.x])
        self.ry0 = min([self.ry0, event.y])
        self.ry1 = max([self.ry1, event.y])
        self.pb_sample_pts=[]

        dt = int( (time()-self.start_time)*1000 )
        self.pb_sample_pts.append([self.x0,self.y0, dt])

    def up(self, event):
        # global x0, y0
        # global pb_sample_signs, pb_sample_pts
        if ( self.x0,self.y0 ) == (event.x, event.y):
            self.canvas.create_line(self.x0, self.y0, self.x0 + 1, self.y0 + 1, width=self.line_width)
            self.imagedraw.line((self.x0, self.y0, self.x0+1, self.y0+1), fill=self.line_col, width=self.line_width)
            dt = int((time() - self.start_time) * 1000)
            self.pb_sample_pts.append([self.x0+1,self.y0+1, dt])

            self.canvas.create_line(self.x0+1, self.y0+1, self.x0 + 2, self.y0 + 2, width=self.line_width)
            self.imagedraw.line((self.x0+1, self.y0+1, self.x0+2, self.y0+2), fill=self.line_col, width=self.line_width)
            self.pb_sample_pts.append([self.x0+2,self.y0+2, dt+10])

            self.pb_sample_signs.append(self.pb_sample_pts)
            self.pb_sample_pts = []
        else:
            self.pb_sample_signs.append(self.pb_sample_pts)

    def drawimage1(self, sign):
        # image create from sample points
        # print('sign=', sign)
        plt.figure()
        for i, seg in enumerate(sign):
            print(seg)
            xy = np.asarray(seg)
            plt.plot(xy[:, 0], xy[:, 1], 'k')
        # y reverse
        plt.gca().invert_yaxis()
        plt.show()


    def key(self, event):
        # print('key pressed=', repr(event.char))
        # global image1
        # global pb_sample_signs, pb_sample_pts
        # global canvas

        if event.char=='\r':
            # print('enter!')
            # self.image1.save('paintbrush.png')
            # image content size
            self.contentcx = self.rx1-self.rx0+self.line_width*2
            self.contentcy = self.ry1-self.ry0+self.line_width*2

            # clear
            self.canvas.delete('all')
            # self.pb_sample_signs=[]
            # image view!!!!
            # img = Image.open('paintbrush.png')
            # img.show()  # default os program!
            # imgnp = np.asarray(img)
            # plt.imshow(imgnp)   # plot draw! (more fast)

            drawstr = PaintBrush.nparrs_to_str(self.pb_sample_signs)
            print(drawstr)

            self.root.destroy()

        if event.char=='\x1b':
            # print('ESC!')
            self.cancel=True
            self.root.destroy()
            # quit()


    def InputDraw(self):
        self.image1 = Image.new('L', (self.width, self.height), 255)
        self.imagedraw = ImageDraw.Draw(self.image1)

        self.canvas.configure(background='white')
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<Button-1>', self.down)
        self.canvas.bind('<ButtonRelease-1>', self.up)
        self.canvas.bind('<Key>', self.key)
        self.canvas.focus_set()
        self.canvas.pack()
        self.root.mainloop()
        # print('end of loop')
        if self.cancel==True:
            return None
        return self.image1

    # image invert!
    # img = np.asarray(image1)
    # print(img)
    # print(255-img)
    def CropContent(self):
        image2 = self.image1.crop((self.rx0 - self.line_width, self.ry0 - self.line_width,
                                   self.rx1 + self.line_width, self.ry1 + self.line_width))
        return image2

    def Resize(self, outx, outy, bonlycontent, rmargin):
        # if bonlycontent==true, then crop content + margin => out size.
        #    bonlycontent==false, then just resize.
        # output image size
        # outx = 100
        # outy = 50
        # rmargin = 5
        # mx = (rx*dx)/(fx-2rx)     ; rx=real margin, dx=content size, fx=figure size(to resize)

        if bonlycontent==True:
            image2 = self.CropContent()
            mx = (rmargin * self.contentcx) / (outx - 2 * rmargin)
            my = (rmargin * self.contentcy) / (outy - 2 * rmargin)
            image3 = Image.new('L', (int(self.contentcx + mx * 2), int(self.contentcy + my * 2)), 255)
            image3.paste(image2, (int(mx), int(my)))
        else:
            image3 = self.image1

        image3 = image3.resize([outx, outy])
        return image3

        # print('pb_sample_signs=', pb_sample_signs)
        # t1=threading.Thread(target=drawimage1, args=(pb_sample_signs,))
        # t1.start()
        # bk_pb_sample_signs = self.pb_sample_signs.copy()

    def Spline(self):
        minx, miny = 999999, 999999
        maxx, maxy = 0, 0
        splinesign=[]
        for seg in self.pb_sample_signs:
            xy = np.asarray(seg)
            if len(xy) > 4:
                # spline
                tck, u = splprep(xy.transpose(), s=0)
                unew = np.arange(0, 1.01, 0.01)
                out = splev(unew, tck)
                splinesign.append(out)
                # plt.plot(out[0], out[1], 'k')
            else:
                # just line
                # plt.plot(xy[:, 0], xy[:, 1], 'k')
                splinesign.append([xy[:,0], xy[:,1]])
            xymin = xy.min(axis=0)
            xymax = xy.max(axis=0)
            minx = min(xymin[0], minx)
            miny = min(xymin[1], miny)
            maxx = max(xymax[0], maxx)
            maxy = max(xymax[1], maxy)
        linewidth = 3
        minx -= linewidth
        miny -= linewidth
        maxx += linewidth
        maxy += linewidth
        return splinesign, (minx, miny, maxx, maxy)

def fig2data(fig):
    # canvas = plt.get_current_fig_manager().canvas
    # canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)

    buf.shape = (w, h, 3)
    buf = np.roll(buf, 3, axis=2)
    return buf

def drawtoimage(splinesign, a0,a1,a2,a3, figcx, figcy):
    # draw plot image spline
    fig=plt.figure(figsize=[figcx/100.0,figcy/100.0])
    #plt.axes(frameon=False)
    ax = plt.axes()

    # tight layout
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout(0)

    # draw lines
    for seg in splinesign:
        plt.plot(seg[0], seg[1], 'k')

    # no margin
    plt.xlim([a0, a2])
    plt.ylim([a1, a3])

    # y reverse
    plt.gca().invert_yaxis()

    # convert plot to image
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    return pil_image


stp = "[(105,99,0)(105,98,27)(105,97,58)(109,97,66)(113,97,74)(119,97,82)(125,97,90)(131,97,98)(135,97,106)(141,97,114)(147,97,122)(150,97,130)(154,97,138)(157,97,146)(161,97,154)(164,97,162)(169,97,169)(175,97,177)(181,97,185)(186,97,194)(190,97,202)(193,97,210)(195,97,218)(196,97,225)(197,97,234)(199,97,242)(202,97,249)(204,97,258)(205,97,266)(207,97,274)(210,97,282)(216,97,289)(218,97,298)(226,97,305)(232,97,314)(236,97,321)(242,97,330)(245,97,338)(251,97,346)(253,97,354)(255,97,362)(258,97,370)(260,97,377)(263,97,386)(265,97,393)(266,97,402)(267,97,409)(269,97,417)(270,97,434)(272,97,441)(273,97,458)(274,98,578)(275,99,578)(276,100,588)]"
lsDraw = PaintBrush.str_to_nparrs(stp)
print(lsDraw)

plt.figure()
plt.xlim([0, 400])
plt.ylim([0, 200])
print(len(lsDraw))
for seg in lsDraw:
    print(seg[:,0])
    print(seg[:,1])
    plt.plot(seg[:,0], seg[:,1])
plt.show()

if __name__=='__main__':
    print('**paintbrush test..')
    pb=PaintBrush('Draw and enter or esc')
    image1= pb.InputDraw()
    image1.save('drawingimage.png')   # org size. (400, 200)
    npimage = np.asarray(image1)    # translate to numpy array!
    print('npimage=', npimage.shape)

    img1 = pb.CropContent()
    img1.save('drawingimage_crop.png')    # content crop(include line width). (cx, cy)

    img2 = pb.Resize(100, 50, True, 0)
    img2.save('drawingimage_resize_nomargin.png')  # crop and resize. (100, 50)

    img2 = pb.Resize(100, 50, True, 5)
    img2.save('drawingimage_resize.png')  # crop and resize include fixed margin. (100, 50)

    # spline curve
    splinesign, (a0, a1, a2, a3) = pb.Spline()
    # print('splinesign=', splinesign)
    # print(a0,a1,a2,a3)
    pil_image = drawtoimage(splinesign, a0, a1, a2, a3, 400, 200)
    pil_image.show()
    pil_image.save('drawingimage_plotimage_spline.png')

    pil_image = drawtoimage(splinesign, a0, a1, a2, a3, 100, 50)
    pil_image.save('drawingimage_plotimage_spline_resize.png')

