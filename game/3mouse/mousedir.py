'''
mouse position angle. move
'''

import pygame
from pygame import Surface
import os, sys
import math

bExit = False

RED=(255,0,0)
GREEN=(0,255,0)
BLUE=(0,0,255)
BLACK=(0,0,0)

pygame.init()
pad = pygame.display.set_mode( (640,480))
pygame.display.set_caption('test')

user = Surface((100,100))
pygame.draw.polygon(user, RED, [(0,0), (100,50), (0,100)], 3)
pygame.draw.line(user, GREEN, (100,50), (0, 50), 2)
user.set_colorkey(BLACK)
pos = user.get_rect()
pos.centerx = 100
pos.centery = 100
print('pos (rect)= ', pos, ' current angle=0')

def rot_center2(image, angle):
    """rotate an image while keeping its center and size. 크기 유지. """
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image

def rot_center(image, rect, angle):
    """rotate an image while keeping its center. 크기 잘림."""
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = rot_image.get_rect(center=rect.center)
    return rot_image, rot_rect


clock = pygame.time.Clock()
speed = 10
while not bExit:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            bExit=True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                bExit=True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pass
        elif event.type == pygame.MOUSEBUTTONUP:
            pass

    key = pygame.key.get_pressed()
    if key[pygame.K_a]:
        pos.centerx -= speed
    if key[pygame.K_d]:
        pos.centerx += speed
    if key[pygame.K_w]:
        pos.centery -= speed
    if key[pygame.K_s]:
        pos.centery += speed


    pad.fill(BLACK)

    mousepos = pygame.mouse.get_pos()
    # print('mousepos=', mousepos)
    angle = math.atan2(pos.centery - mousepos[1], mousepos[0] - pos.centerx)
    print('angle=', angle)
    # 각도는 라디안 0~pi, -pi, 0
    # user는 x축 방향 0도 기준으로 있음. user를 angle만큼 CCW로 회전.

    # degree로 변환 필요.
    # img = pygame.transform.rotate(user, angle*180/math.pi)
    img, rect = rot_center(user, user.get_rect(), angle*180/math.pi)
    # img = rot_center2(user, angle*180/math.pi)
    # pad.blit(img, (pos.x, pos.y) )
    rect.centerx += pos.x
    rect.centery += pos.y
    pad.blit(img, (rect.x, rect.y))

    mousedown = pygame.mouse.get_pressed()

    # 마우스 다운 상태면 선을 그림.
    if mousedown[0]:
        pygame.draw.line(pad, BLUE, mousepos, rect.center)

    # pad.blit(user, (pos.x, pos.y) )
    pygame.display.flip()
    # pygame.display.upate()
    clock.tick(60)


pygame.quit()

