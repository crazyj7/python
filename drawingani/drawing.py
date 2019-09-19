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
            arPts = np.asarray(lsPts, dtype=int)
            if len(arPts)>0:
                lsDraw.append(arPts)
        return lsDraw

    # 화면 초기화
    def clear(self):
        self.line_col = 0   # line color
        self.line_width = 3 # line thickness
        self.cancel = False
        self.imagedraw = None
        self.start_flag = False  # timer start.
        self.start_time = 0
        self.skip_term = 20 # 모든 점을 기록하면 좋겠지만, 재생시 어려움. sampling interval을 다음 ms로 함. 0이면 전부 추출함.
        self.time_old = 0 # 이전 추출된 포인트 시간
        # members
        self.image1 = None  # org image (400x200)
        # only content area lefttop(rx0,ry0), rightbottom(rx1,ry1)
        self.rx0 = 9999999
        self.rx1 = 0
        self.ry0 = 9999999
        self.ry1 = 0
        self.contentcx = 0  # content area (line width include)
        self.contextcy = 0  # content area (line width include)
        self.pb_sample_lines = []  # segment list (multi line)
        self.pb_sample_seg = []  # one segment (one line (x,y,t) )
        self.canvas.delete('all')

        # 윈도우를 이미지 파일로 뜨기 위함.
        self.image1 = Image.new('L', (self.width, self.height), 255)
        self.imagedraw = ImageDraw.Draw(self.image1)

        # 샘플링 좌표를 텍스트 형태로 저장
        # ex) [(x,y,t)...][(x,y,t)...] ...    (모니터 화면 좌표계 사용)
        # (x,y,t) : x, y 좌표. t=시간 (millisecond) 0부터 시작
        # [...] [...] [...] 세그먼트 단위로 대괄호
        self.drawstr = '[]'


    def __init__(self, title, width=400, height=200):
        self.title = title
        self.root = None
        self.frame = None
        self.canvas = None
        self.root = tkinter.Tk()
        self.root.title(title)
        # 화면 크기
        self.width, self.height = width, height
        self.frame = tkinter.Frame(self.root, width=self.width, height=self.height)
        #frame.bind('<Key>', key)
        self.frame.pack()
        self.canvas = tkinter.Canvas(self.frame, width=self.width, height=self.height)
        self.clear()


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

        now = time() * 1000
        if (now-self.time_old)>self.skip_term:
            dt = int( time()*1000 - self.start_time )
            self.pb_sample_seg.append([self.x0, self.y0, dt])
            self.time_old = now

    def down(self, event):
        # global x0, y0
        # global rx0, rx1, ry0, ry1
        # global pb_sample_signs, pb_sample_pts
        if self.start_flag==False:
            self.start_flag=True
            self.start_time=time()*1000
            self.time_old=time()*1000

        self.x0,self.y0 = event.x, event.y
        self.rx0 = min([self.rx0, event.x])
        self.rx1 = max([self.rx1, event.x])
        self.ry0 = min([self.ry0, event.y])
        self.ry1 = max([self.ry1, event.y])
        self.pb_sample_seg=[]

        dt = int( time()*1000-self.start_time )
        self.pb_sample_seg.append([self.x0, self.y0, dt])

    def up(self, event):
        # global x0, y0
        # global pb_sample_signs, pb_sample_pts
        if ( self.x0,self.y0 ) == (event.x, event.y):
            self.canvas.create_line(self.x0, self.y0, self.x0 + 1, self.y0 + 1, width=self.line_width)
            self.imagedraw.line((self.x0, self.y0, self.x0+1, self.y0+1), fill=self.line_col, width=self.line_width)
            dt = int(time()*1000 - self.start_time)
            self.pb_sample_seg.append([self.x0 + 1, self.y0 + 1, dt])

            self.canvas.create_line(self.x0+1, self.y0+1, self.x0 + 2, self.y0 + 2, width=self.line_width)
            self.imagedraw.line((self.x0+1, self.y0+1, self.x0+2, self.y0+2), fill=self.line_col, width=self.line_width)
            self.pb_sample_seg.append([self.x0 + 2, self.y0 + 2, dt + 10])

            self.pb_sample_lines.append(self.pb_sample_seg)
            self.pb_sample_seg = []
        else:
            self.pb_sample_lines.append(self.pb_sample_seg)

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

        if event.char=='\b':
            print('backspace')
            self.clear()

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

            drawstr = PaintBrush.nparrs_to_str(self.pb_sample_lines)
            self.drawstr = drawstr
            # print(drawstr)

            self.root.destroy()

        if event.char=='\x1b':
            # print('ESC!')
            self.cancel=True
            self.root.destroy()
            # quit()


    def InputDraw(self):
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
        # print('return:', self.drawstr)
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
        for seg in self.pb_sample_lines:
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

if __name__=='__main__':
    print('**paintbrush test..')
    print('drawing with mouse cursor.')
    print('draw end : ENTER key')
    print('cancel : ESC key')
    print('clear : BackSpace key')

    if False:
        # stp = "[(105,99,0)(105,98,27)(105,97,58)(109,97,66)(113,97,74)(119,97,82)(125,97,90)(131,97,98)(135,97,106)(141,97,114)(147,97,122)(150,97,130)(154,97,138)(157,97,146)(161,97,154)(164,97,162)(169,97,169)(175,97,177)(181,97,185)(186,97,194)(190,97,202)(193,97,210)(195,97,218)(196,97,225)(197,97,234)(199,97,242)(202,97,249)(204,97,258)(205,97,266)(207,97,274)(210,97,282)(216,97,289)(218,97,298)(226,97,305)(232,97,314)(236,97,321)(242,97,330)(245,97,338)(251,97,346)(253,97,354)(255,97,362)(258,97,370)(260,97,377)(263,97,386)(265,97,393)(266,97,402)(267,97,409)(269,97,417)(270,97,434)(272,97,441)(273,97,458)(274,98,578)(275,99,578)(276,100,588)]"
        stp = "[(192,143,0)(191,144,40)(189,144,56)(186,144,56)(185,144,62)(181,144,69)(178,144,77)(174,144,86)(171,144,93)(165,143,101)(164,142,109)(160,141,117)(158,139,125)(155,137,133)(152,135,141)(149,132,149)(145,129,157)(144,128,173)(142,127,189)(141,126,198)(141,125,205)(140,124,213)(139,123,221)(138,121,229)(137,120,246)(137,119,270)(137,118,277)(135,116,293)(135,113,309)(135,112,318)(135,111,326)(135,110,334)(135,106,342)(135,105,350)(135,101,358)(135,100,366)(135,98,373)(135,95,382)(135,92,390)(135,91,397)(136,87,406)(139,84,414)(141,81,421)(143,79,429)(146,76,438)(149,73,445)(150,72,453)(151,69,462)(155,66,469)(157,65,478)(159,63,485)(162,60,493)(163,59,501)(165,59,510)(167,57,517)(171,56,525)(173,55,533)(178,53,541)(180,52,549)(185,50,558)(187,50,565)(192,50,574)(194,49,582)(195,49,589)(199,49,598)(202,49,606)(204,49,614)(207,49,622)(210,49,629)(214,49,638)(218,49,646)(219,49,654)(220,49,662)(222,49,669)(227,49,677)(228,49,685)(232,49,694)(235,52,702)(239,54,709)(245,54,717)(248,55,726)(249,56,733)(252,59,741)(254,60,749)(255,60,757)(257,61,765)(258,62,773)(260,65,781)(261,66,789)(263,69,805)(263,70,821)(263,71,830)(265,74,845)(265,76,861)(265,77,869)(265,78,877)(265,82,885)(265,83,893)(265,86,901)(265,88,909)(265,89,917)(265,91,925)(265,92,933)(265,94,941)(265,95,957)(265,96,965)(265,98,973)(265,100,981)(265,103,989)(265,104,997)(264,106,1005)(263,107,1013)(261,110,1021)(259,113,1029)(257,114,1045)(256,116,1053)(255,117,1061)(252,120,1069)(250,123,1085)(248,124,1093)(247,125,1101)(246,126,1109)(245,127,1117)(244,129,1125)(243,130,1134)(242,131,1141)(241,132,1150)(239,133,1158)(236,133,1166)(234,134,1173)(233,134,1182)(229,136,1190)(227,138,1198)(224,138,1214)(221,140,1222)(220,140,1237)(218,140,1245)(217,140,1253)(215,140,1269)(214,140,1277)(212,140,1286)(211,140,1318)(210,141,1325)(208,141,1358)(209,142,1358)(210,143,1368)][(218,69,2261)(216,70,2318)(213,73,2326)(207,80,2333)(202,83,2341)(198,88,2350)(193,89,2357)(187,94,2365)(185,95,2373)(180,98,2382)(177,101,2389)(176,102,2397)(175,102,2405)(174,103,2413)(172,104,2429)(170,104,2446)(167,107,2453)(166,108,2469)(165,109,2493)(164,109,2557)(165,110,2605)(166,111,2615)][(178,71,2991)(179,70,3029)(183,73,3037)(186,76,3046)(190,80,3054)(195,85,3062)(198,86,3070)(202,91,3078)(207,94,3085)(208,95,3094)(210,98,3102)(212,99,3109)(213,100,3118)(214,101,3126)(215,101,3135)(217,104,3143)(219,105,3166)(220,106,3174)(221,106,3182)(222,107,3189)(223,108,3206)(224,109,3382)(225,110,3392)]"
        lsDraw = PaintBrush.str_to_nparrs(stp)
        print(lsDraw)

        plt.figure()
        print(len(lsDraw))
        # plt.plot([0,400,400,0], [0,200,0,200])
        for seg in lsDraw:
            # print(seg[:,0])
            # print(seg[:,1])
            plt.plot(seg[:, 0], seg[:, 1])

        plt.grid()
        plt.xlim([0, 400])
        plt.ylim([0, 200])
        plt.axvline(0)
        plt.axhline(0)
        plt.show()

    pb=PaintBrush('Draw and enter or backspace or esc')
    image1= pb.InputDraw()
    image1.save('drawingimage.png')   # org size. (400, 200)

    # y좌표축은 컴퓨터 좌표계 기준임. (top이 low이고, bottom이 high value)
    # 따라서 matplotlib으로 plot할 때 y-axis를 reverse해줘야 한다.
    print(pb.drawstr)
    with open('drawingimage.txt', 'wt') as f:
        f.write(pb.drawstr)

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
    # pil_image.show()
    pil_image.save('drawingimage_plotimage_spline.png')

    pil_image = drawtoimage(splinesign, a0, a1, a2, a3, 100, 50)
    pil_image.save('drawingimage_plotimage_spline_resize.png')

