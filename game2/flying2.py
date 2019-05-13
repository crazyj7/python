'''
using sprite

'''

import pygame
import random
from pygame.sprite import Sprite
from pygame import Surface

from time import sleep

white = (255,255,255)
pad_width=1024
pad_height=512
background_width=1024
gamepad=None

# image
aircraft=None
background1=None
background2=None
bat=None
fires=None
bullet=None
boom=None

clock=None

# size
bat_height=None
bat_width=None
fire_height=0
fire_width=0

#sound
sound_shot=None
sound_explosion=None

FPS=60


class Plane(Sprite):
    def __init__(self):
        Sprite.__init__(self)

        self.sprite_image = 'plane_sp.png'
        self.sprite_width = 89
        self.sprite_height = 55
        self.sprite_sheet = pygame.image.load(self.sprite_image).convert_alpha()
        self.sprite_columns = 3
        self.current_frame = 0
        self.image = Surface((self.sprite_width, self.sprite_height))   # 이미지객체 (서피스)
        rect = (self.sprite_width*self.current_frame, 0, self.sprite_width, self.sprite_height)
        self.image.blit(self.sprite_sheet, (0,0), rect)
        # self.image.set_colorkey((255,0,255))
        self.image.set_colorkey((0,0,0))
        self.rect = self.image.get_rect()       # 위치. (전체기준)
        self.speed = 5

    def update(self):
        rect = (self.sprite_width*self.current_frame, 0, self.sprite_width, self.sprite_height)
        self.image.blit(self.sprite_sheet, (0,0), rect)

    def move_up(self):
        self.rect.y -= self.speed
        self.current_frame = 1
        if self.rect.y < 0 :
            self.rect.y += self.speed

    def move_down(self):
        self.rect.y += self.speed
        self.current_frame = 2
        if self.rect.y+self.sprite_height > pad_height :
            self.rect.y -= self.speed

    def move_middle(self):
        self.current_frame = 0



class Enemy1(Sprite):
    def __init__(self):
        Sprite.__init__(self)

        self.speed = 3    # 3프레임마다 움직임. 적을수록 빠름.
        self.sprite_image = 'fireball_sp.png'
        self.sprite_width = 140
        self.sprite_height = 61
        self.sprite_sheet = pygame.image.load(self.sprite_image).convert_alpha()
        self.sprite_columns = 2
        self.current_frame = 0
        self.image = Surface((self.sprite_width, self.sprite_height))
        rect = (self.sprite_width*self.current_frame, 0, self.sprite_width, self.sprite_height)
        self.image.blit(self.sprite_sheet, (0,0), rect)
        self.image.set_colorkey((0,0,0))
        self.rect = self.image.get_rect()
        self.delay = self.speed / FPS
        self.lastupdate = pygame.time.get_ticks()
        self.dx = 10
        self.rect.y = random.randrange(0, pad_height-self.sprite_height)
        self.rect.x = pad_width

    def update(self):
        t = pygame.time.get_ticks()
        if t-self.lastupdate > self.delay:
            self.lastupdate = t
            self.current_frame += 1
            self.rect.x -= self.dx

            if self.current_frame==self.sprite_columns:
                self.current_frame = 0
            if self.rect.x < -self.sprite_width :
                self.kill()
                return

        rect = (self.sprite_width*self.current_frame, 0, self.sprite_width, self.sprite_height)
        self.image.blit(self.sprite_sheet, (0,0), rect)

        gamepad.blit(self.image, (self.rect.x, self.rect.y))


def drawObject(obj, x, y):
    gamepad.blit(obj, (x,y))


def gameover():
    global gamepad
    text = 'Game Over!'
    largeText = pygame.font.Font('freesansbold.ttf', 115)
    textSurface = largeText.render(text, True, (255,0,0))
    textRect = textSurface.get_rect()
    textRect.center = ((pad_width/2), (pad_height/2))
    gamepad.blit(textSurface, textRect)
    pygame.display.update()
    sleep(2)
    runGame()



def runGame():
    # global gamepad, clock, aircraft, background1, background2
    # global bat, fires, bullet

    bullet_xy=[]
    fBatDied = False
    boom_count=0

    x = pad_width * 0.05
    y = pad_height * 0.4
    y_change = 0

    crashed= False
    background1_x=0
    background2_x = background_width

    bat_x=pad_width
    bat_y=random.randrange(0, pad_height)

    # fire_x = pad_width
    # fire_y=random.randrange(0, pad_height)
    # random.shuffle(fires)
    # fire = fires[0]


    plane = Plane()
    plane.rect.x = 50
    plane.rect.y = pad_height/2 - plane.sprite_height/2
    planeGroup = pygame.sprite.Group()
    planeGroup.add(plane)


    enemyGroup = pygame.sprite.Group()

    while not crashed:

        bshot=False
        # get key event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    crashed = True
                elif event.key == pygame.K_SPACE:
                    pass
                elif event.key == pygame.K_LCTRL:
                    bshot=True

            if event.type == pygame.KEYUP:
                plane.move_middle()

        # get key status (continous)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            plane.move_up()
        elif keys[pygame.K_DOWN]:
            plane.move_down()

        # if bshot:
        #     bshot=False
        #     bullet_x = x + aircraft_width
        #     bullet_y = y + aircraft_height/2
        #     bullet_xy.append([bullet_x, bullet_y])
        #     pygame.mixer.Sound.play(sound_shot)


        # background first
        gamepad.fill(white)

        background1_x -=2
        background2_x -=2
        if background1_x == -background_width:
            background1_x = background_width
        if background2_x == -background_width:
            background2_x = background_width
        drawObject(background1, background1_x, 0)
        drawObject(background2, background2_x, 0)


        # bat and fire
        bat_x -= 7
        if bat_x<=0:
            bat_x = pad_width
            bat_y = random.randrange(0, pad_height-bat.get_height())

        # if fire==None:
        #     fire_x -=30
        # else:
        #     fire_x -= 15
        # if fire_x<=0:
        #     random.shuffle(fires)
        #     fire=fires[0]
        #     fire_x = pad_width
        #     if fire!=None:
        #         fire_y = random.randrange(0, pad_height-fire.get_height())

        drawObject(bat, bat_x, bat_y)
        # if fire!=None:
        #     drawObject(fire, fire_x, fire_y)


        # player
        plane.update()

        # enemy spawn.  적 생성....
        if len(enemyGroup.sprites())==0 and random.randint(0,10)==0:
            enemy = Enemy1()
            enemyGroup.add(enemy)
        elif random.randint(0,30)==0:
            enemy = Enemy1()
            enemyGroup.add(enemy)


        enemyGroup.update()     # 업데이트 및 그리기

        # 플레이어 그리기
        drawObject(plane.image, plane.rect.x, plane.rect.y)


        # 충돌 감지
        collided = pygame.sprite.groupcollide(planeGroup, enemyGroup, False, True)
        if collided:
            # game over
            # for c1, c2 in collided.items():
            #     drawObject(boom, c2[0].rect.left, c2[0].rect.top)
            #     break
            drawObject(boom, plane.rect.x, plane.rect.y)
            pygame.mixer.Sound.play(sound_explosion)
            return gameover()

        # air_pos=[[x,y], [x+aircraft_width, y], [x, y+aircraft_height], [x+aircraft_width, y+aircraft_height]]
        # if fire!=None:
        #     obj_pos = [[fire_x, fire_y], [fire_x+fire_width, fire_y],
        #             [fire_x, fire_y+fire_height], [fire_x+fire_width, fire_y+fire_height]]
            # if checkCollision(air_pos, obj_pos, 10):

        obj_pos = [[bat_x, bat_y],[bat_x + bat_width, bat_y],
                   [bat_x, bat_y + bat_height], [bat_x + bat_width, bat_y + bat_height]]
        finc = False
        # if checkCollision(air_pos, obj_pos, 10):
        #     game over
            # drawObject(boom, x, y)
            # pygame.mixer.Sound.play(sound_explosion)
            # return gameover()

        # bullet
        brm=[]
        for i, bxy in enumerate(bullet_xy):
            bullet_xy[i][0] += 15 # bullet x pos
            if bullet_xy[i][0]>=pad_width:
                brm.append(i)
        brm.reverse()
        for idx in brm:
            bullet_xy.remove(bullet_xy[idx])
        for i, bxy in enumerate(bullet_xy):
            drawObject(bullet, bxy[0], bxy[1])
            if bxy[0]>=bat_x and bxy[0]<=bat_x+bat_width and    \
                bxy[1]>=bat_y and bxy[1]<=bat_y+bat_height:
                fBatDied=True
                bullet_xy.remove(bxy)

        if fBatDied:
            drawObject(boom, bat_x, bat_y)
            boom_count+=1
            if boom_count>5:
                boom_count=0
                fBatDied=False
                bat_x = pad_width
                bat_y = random.randrange(0, pad_height-bat.get_height())


        # update
        pygame.display.update()
        clock.tick(FPS)
    pygame.quit()
    quit()


def initGame():
    global gamepad, aircraft, clock, background1, background2
    global bat, fires, bullet
    global bat_width, bat_height, boom
    global sound_shot, sound_explosion

    fires=[]

    pygame.init()

    sound_shot = pygame.mixer.Sound('shot.wav')
    sound_explosion = pygame.mixer.Sound('explosion.wav')

    gamepad = pygame.display.set_mode((pad_width, pad_height))
    pygame.display.set_caption('PyFlying')

    # aircraft = pygame.image.load('plane.png').convert_alpha()
    # aircraft_width = aircraft.get_width()
    # aircraft_height = aircraft.get_height()
    # print('aircraft size=', aircraft_width, aircraft_height)

    background1 = pygame.image.load('back2.png').convert_alpha()
    background2 = background1.copy()

    bat = pygame.image.load('bat.png').convert_alpha()
    bat_height = bat.get_height()
    bat_width = bat.get_width()

    # fire1 = pygame.image.load('fireball.png').convert_alpha()
    # fire_width = fire1.get_width()
    # fire_height = fire1.get_height()
    # fires.append(fire1)
    # fires.append(pygame.image.load('fireball2.png'))

    for i in range(5):
        fires.append(None)

    bullet = pygame.image.load('bullet.png').convert_alpha()

    boom = pygame.image.load('boom.png').convert_alpha()

    # background music
    # pygame.mixer.music.load('mybgm.wav')
    # pygame.mixer.music.play(-1)


    clock = pygame.time.Clock()
    runGame()


if __name__=='__main__':
    initGame()



