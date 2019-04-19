import pyautogui as pag
import random
import time
import subprocess
import codecs

combat_button = {
    'lt':{'x':771, 'y':453},
    'rb':{'x':912, 'y':542}
}

def run_and_result(cmd):
    p = subprocess.Popen('cmd.exe /c "'+cmd+'"', stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, universal_newlines=True)
    datas = p.stdout.read()
    print(datas)

run_and_result("dir")

def myClick():
    print('click')
    pag.mouseDown()
    time.sleep(random.uniform(0.15, 0.29))
    pag.mouseUp()


def myDrag(fro, to ):
    drag_from = {
        'x': random.uniform(fro['x'])
    }
    pag.moveTo()

def myButtonClick(btn):
    # 버튼 클릭하기.
    # 마우스를 천천이 이동하고,
    # 버튼 영역에 오면 클릭한다.
    while True:
        print('move')
        duration = random.uniform(0.5, 1.5)
        pag.moveTo(
            x=random.uniform(btn['lt']['x'], btn['rb']['x']),
            y=random.uniform(btn['lt']['y'], btn['rb']['y']),
            duration=duration
        )
        # 버튼 영역인지 확인.
        x,y = pag.position()
        if (btn['lt']['x'] < x < btn['rb']['x'])==True and  \
                (btn['lt']['y'] < y < btn['rb']['y'])==True :
            myClick()
            break



myButtonClick(combat_button)



# 마우스 위치 보기
# while True:
#     x,y = pag.position()
#     print(x,y)


