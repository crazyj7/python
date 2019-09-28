
import pygame
from pygame import mixer
import time

# MP3 파일에 따라 제약이 있을 수 있다.
pygame.init()
mixer.init()
print('vol=', mixer.music.get_volume())  # 0~1
mixer.music.set_volume(0.8)
print('vol=', mixer.music.get_volume())  # 0~1

if False:
    # mp3 play
    mixer.music.load('06.mp3')
    mixer.music.play()
    for i in range(20):
        print(i, 'busy=', mixer.music.get_busy(), 'endevent=', mixer.music.get_endevent(),
              'pos=', mixer.music.get_pos())
        time.sleep(1)
    mixer.music.stop()

# 윈도우에서는 작동 함.
if False:
    import playsound
    playsound.playsound('06.mp3', True)

# convert...
from pydub import AudioSegment
if True:
    wa = AudioSegment.from_mp3('06.mp3')
    wa.export('06.wav', format='wav')

    wa2 = AudioSegment.from_wav('06.wav')
    wa2.export('06_.mp3', format='mp3')

# play
mixer.music.load('06.wav')
mixer.music.play()
for i in range(20):
    print(i, 'busy=', mixer.music.get_busy(), 'endevent=', mixer.music.get_endevent(),
          'pos=', mixer.music.get_pos())
    time.sleep(1)
mixer.music.stop()


