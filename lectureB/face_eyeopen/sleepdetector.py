
import time
import numpy as np
from pygame import mixer



class SleepDetector:
    score_list = []
    time_list = []
    duration = 3.0
    readytime = 5.0
    starttime = 0.0
    # thr = 0.4

    playing = False
    lastplay = 0
    playdur = 5.0

    def __init__(self):
        mixer.init()
        mixer.music.load('YaoC3.mp3')
        SleepDetector.starttime = time.time()

    def play(self):
        if SleepDetector.lastplay + SleepDetector.playdur > time.time() :
            return
        print('!!!PLAY!!!')
        SleepDetector.lastplay = time.time()
        mixer.music.play(0, 0)      # async play

    def push(self, score):
        SleepDetector.time_list.append(time.time())
        if SleepDetector.starttime+SleepDetector.readytime < time.time() :
            SleepDetector.score_list.append(score)
        else:
            SleepDetector.score_list.append(1.0)

    def update(self):
        # remove old node
        npt = np.asarray(SleepDetector.time_list, dtype=np.float)
        now = time.time()
        idx = np.argwhere(npt < now-SleepDetector.duration ).flatten()
        idx=np.sort(idx)
        idx=idx[::-1]
        # print(idx)
        if len(idx)>0 :
            for i in idx:
                del SleepDetector.score_list[i]
                del SleepDetector.time_list[i]

    def count(self):
        return len(SleepDetector.score_list)

    def score(self):
        self.update()
        tot = 0
        for v in SleepDetector.score_list:
            tot += v
        return tot / len(SleepDetector.score_list)

    def print(self):
        for i in range(len(SleepDetector.score_list)):
            print(SleepDetector.time_list[i], SleepDetector.score_list[i])

if __name__=='__main__':
    sd = SleepDetector()
    sd.push(0.5)
    sd.push(0.2)
    time.sleep(0.5)
    sd.push(0.1)

    sd.print()
    print( sd.score() )

    sd.play()

    while True:
        time.sleep(1)
        sd.update()
        sd.push(0.5)
        print( sd.score() )
        # sd.play()




