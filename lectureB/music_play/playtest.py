'''
비동기 방식 mp3 재생
pygame.mixer 사용

P ; pause / unpause.
다 재생되면 프로그램 종료.
'''

import time
import pygame
from pygame import mixer
import os, sys
from pygame.locals import *

# path1 = '25 seventh.mp3'
path1 = '06.mp3'

pygame.init() # 믹서만 쓸때는 이건 필요없음.
pygame.display.set_caption('music test')
screen = pygame.display.set_mode((400,300))

mixer.init()
mixer.music.load(path1)

print('call play')
mixer.music.play()
print('return play')

clock = pygame.time.Clock()
paused = False
while True:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == KEYUP and event.key== ord('p'):
            if paused==False:
                print('pause')
                mixer.music.pause()
                paused = True
            else:
                print('unpause')
                mixer.music.unpause()
                paused = False

    # key_event = pygame.key.get_pressed()
    # if key_event[pygame.K_p]:   # 'P' paused

    # print('.', end=' ')
    # sys.stdout.flush()

    # time.sleep(1)
    if not mixer.music.get_busy():
        break
    screen.fill((0,0,0))
    pygame.display.update()
print('play end.')


