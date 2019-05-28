'''
using sprite

'''

import pygame
import wave
import random
from pygame.sprite import Sprite
from pygame import Surface

from time import sleep

white = (255, 255, 255)




class Plane(Sprite):
    def __init__(self):
        Sprite.__init__(self)

        self.status = MainClass.STATUS_ALIVE
        self.sprite_image = 'res/plane_sp.png'
        self.power = 1

        self.sprite_bomb = pygame.image.load('res/boom.png').convert_alpha()
        self.sprite_width = 89
        self.sprite_height = 55
        self.sprite_sheet = pygame.image.load(self.sprite_image).convert_alpha()
        self.sprite_columns = 3
        self.current_frame = 0
        self.image = Surface((self.sprite_width, self.sprite_height))  # 이미지객체 (서피스)
        rect = (self.sprite_width * self.current_frame, 0, self.sprite_width, self.sprite_height)
        self.image.blit(self.sprite_sheet, (0, 0), rect)
        # self.image.set_colorkey((255,0,255))
        self.image.set_colorkey((0, 0, 0))
        self.rect = self.image.get_rect()  # 위치. (전체기준)
        self.speed = 5
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        rect = (self.sprite_width * self.current_frame, 0, self.sprite_width, self.sprite_height)
        self.image.fill((0, 0, 0))
        self.image.blit(self.sprite_sheet, (0, 0), rect)
        if self.status == MainClass.STATUS_DEAD:
            self.image.blit(self.sprite_bomb, (0, 0))
        self.image.set_colorkey((0, 0, 0))

    def move_up(self):
        self.rect.y -= self.speed
        self.current_frame = 1
        if self.rect.y < 0:
            self.rect.y += self.speed

    def move_down(self):
        self.rect.y += self.speed
        self.current_frame = 2
        if self.rect.y + self.sprite_height > MainClass.pad_height:
            self.rect.y -= self.speed

    def move_backward(self):
        self.rect.x -= self.speed
        self.current_frame = 1
        if self.rect.x < 0:
            self.rect.x += self.speed

    def move_forward(self):
        self.rect.x += self.speed
        self.current_frame = 2
        if self.rect.x + self.sprite_width > MainClass.pad_width:
            self.rect.x -= self.speed

    def move_middle(self):
        self.current_frame = 0

    def set_status(self, status):
        self.status = status

    def set_power(self, power):
        self.power = power


class BackGround(Sprite):
    def __init__(self):
        Sprite.__init__(self)
        self.sprite_image = 'res/back2_sp.png'  # double size width (2048)
        self.sprite_width = 1024
        self.sprite_height = 512
        self.current_x = 0
        self.sprite_sheet = pygame.image.load(self.sprite_image).convert()
        self.image = Surface((self.sprite_width, self.sprite_height))
        rect = (self.current_x, 0, self.sprite_width, self.sprite_height)
        self.image.blit(self.sprite_sheet, (0, 0), rect)
        self.rect = self.image.get_rect()
        self.dx = 2
        self.rect.y = 0
        self.rect.x = 0

    def update(self):
        self.current_x += self.dx
        if self.current_x > self.sprite_width:
            self.current_x -= self.sprite_width
        rect = (self.current_x, 0, self.sprite_width, self.sprite_height)
        self.image.blit(self.sprite_sheet, (0, 0), rect)


class Enemy1(Sprite):
    def __init__(self):
        Sprite.__init__(self)

        self.speed = 1  # 30프레임마다 움직임. 적을수록 빠름.
        self.sprite_image = 'res/fireball_sp.png'
        self.sprite_width = 140
        self.sprite_height = 61
        self.sprite_sheet = pygame.image.load(self.sprite_image).convert_alpha()
        self.sprite_columns = 2
        self.current_frame = 0
        self.image = Surface((self.sprite_width, self.sprite_height))
        rect = (self.sprite_width * self.current_frame, 0, self.sprite_width, self.sprite_height)
        self.image.blit(self.sprite_sheet, (0, 0), rect)
        self.image.set_colorkey((0, 0, 0))  # 투명컬러 지정.
        self.rect = self.image.get_rect()

        self.delay = 1000 * self.speed * 1 / (MainClass.FPS)  # speed frame에 해당되는 시간.
        self.lastupdate = pygame.time.get_ticks()
        self.lastupdateframe = pygame.time.get_ticks()
        self.delayframe = 500  # 0.5초마다 변경.
        self.dx = 10  # 이동 거리.
        self.rect.y = random.randrange(0, MainClass.pad_height - self.sprite_height)
        self.rect.x = MainClass.pad_width
        self.mask = pygame.mask.from_surface(self.image)  # 투명이외 부분만 충돌 마스킹으로 설정.

    def update(self):
        t = pygame.time.get_ticks()
        if t - self.lastupdate > self.delay:
            self.lastupdate = t
            self.rect.x -= self.dx
            if self.rect.x < -self.sprite_width:
                self.kill()
                return
        if t - self.lastupdateframe > self.delayframe:
            self.lastupdateframe = t
            self.current_frame += 1
            if self.current_frame == self.sprite_columns:
                self.current_frame = 0
            # self.current_frame = 1

        rect = (self.sprite_width * self.current_frame, 0, self.sprite_width, self.sprite_height)
        self.image.fill((0, 0, 0))  # 이전 프레임을 자동으로 지워주지 않는다. 이렇게 채워줘야함.
        self.image.blit(self.sprite_sheet, (0, 0), rect)
        self.image.set_colorkey((0, 0, 0))


class Enemy2(Sprite):
    def __init__(self):
        Sprite.__init__(self)

        self.score = 10
        self.speed = 3  # 프레임마다 움직임. 적을수록 빠름.
        self.sprite_image = 'res/ship1_r.png'
        self.sprite_bomb = pygame.image.load('res/boom.png').convert_alpha()
        self.sprite_width = 100
        self.sprite_height = 100
        self.sprite_sheet = pygame.image.load(self.sprite_image).convert_alpha()
        self.image = Surface((self.sprite_width, self.sprite_height))
        rect = (0, 0, self.sprite_width, self.sprite_height)
        self.image.blit(self.sprite_sheet, (0, 0), rect)
        self.image.set_colorkey((0, 0, 0))  # 투명컬러 지정.
        self.rect = self.image.get_rect()

        self.dx = 10  # 이동 거리.
        self.rect.y = random.randrange(0, MainClass.pad_height - self.sprite_height)
        self.rect.x = MainClass.pad_width
        self.mask = pygame.mask.from_surface(self.image)  # 투명이외 부분만 충돌 마스킹으로 설정.
        self.status = MainClass.STATUS_ALIVE
        self.delayexplode = 50

    def update(self):
        if self.status == MainClass.STATUS_EXPLODE:
            rect = (0, 0, self.sprite_width, self.sprite_height)
            self.image.blit( self.sprite_sheet, (0,0), rect)
            self.image.blit( self.sprite_bomb, (30,10), rect)
            self.image.set_colorkey((0,0,0))
            self.delayexplode -= 1
            if self.delayexplode<=0 :
                self.status = MainClass.STATUS_DEAD
        elif self.status == MainClass.STATUS_DEAD:
            self.kill()
            return
        else:
            self.rect.x -= self.dx
            if self.rect.x < -self.sprite_width:
                self.kill()
                return

    def set_status(self, status):
        self.status = status


class Bullet(Sprite):
    def __init__(self):
        Sprite.__init__(self)

        self.speed = 1  # 1프레임마다 움직임. 적을수록 빠름.
        self.sprite_image = 'res/bullet.png'
        self.sprite_width = 27
        self.sprite_height = 5
        self.sprite_sheet = pygame.image.load(self.sprite_image).convert_alpha()
        self.image = Surface((self.sprite_width, self.sprite_height))
        rect = (0, 0, self.sprite_width, self.sprite_height)
        self.image.blit(self.sprite_sheet, (0, 0), rect)
        self.image.set_colorkey((0, 0, 0))
        self.rect = self.image.get_rect()

        self.delay = 1000 * self.speed * 1 / (MainClass.FPS)  # speed frame에 해당되는 시간.
        self.dx = 10  # 이동 거리.
        self.dy = 0
        self.rect.y = MainClass.plane.rect.y + int(MainClass.plane.sprite_height / 2)
        self.rect.x = MainClass.plane.rect.x + int(MainClass.plane.sprite_width - 20)
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        if self.rect.x > MainClass.pad_width:
            self.kill()
            return
        self.rect.x += self.dx
        self.rect.y += self.dy

    def set_type(self, mode):
        if mode==0 :
            self.dx = 10
            self.dy = 0
        elif mode==1:
            self.dx = 10
            self.dy = -5
        elif mode==2:
            self.dx = 10
            self.dy = 5


class MainClass:
    score = 0

    pad_width = 1024
    pad_height = 512
    gamepad = None

    # char
    plane = None
    planeGroup = None
    enemyGroup = None
    bulletGroup = None
    back = None
    backGroup = None
    enemyGroup2 = None
    explodeGroup = None

    # sound
    sound_shot = None
    sound_explosion = None

    FPS = 60
    STATUS_DEAD = 0
    STATUS_ALIVE = 1
    STATUS_EXPLODE = 2

    STATUS_READY = 0
    STATUS_PLAY = 1
    STATUS_END = 2
    gamestatus = STATUS_READY
    crashed = False

    lastBullet = 0
    bulletDelay = 0     # 재장전까지 딜레이

    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 1, 512)
        pygame.init()
        self.initFirst()
        self.initVars()

    def initVars(self):
        MainClass.score = 0
        MainClass.pad_width = 1024
        MainClass.pad_height = 512
        MainClass.gamestatus = MainClass.STATUS_READY
        MainClass.crashed = False

        if MainClass.planeGroup!= None :
            MainClass.planeGroup.empty()
        MainClass.plane = Plane()
        MainClass.plane.rect.x = 50
        MainClass.plane.rect.y = MainClass.pad_height / 2 - MainClass.plane.sprite_height / 2
        MainClass.planeGroup = pygame.sprite.Group()
        MainClass.planeGroup.add(MainClass.plane)

        MainClass.back = BackGround()
        MainClass.backGroup = pygame.sprite.Group()
        MainClass.backGroup.add(MainClass.back)

        MainClass.bulletGroup = pygame.sprite.Group()
        MainClass.bulletDelay = 1000  # ms
        MainClass.lastBullet = pygame.time.get_ticks()

        if MainClass.enemyGroup!= None :
            MainClass.enemyGroup.empty()
        MainClass.enemyGroup = pygame.sprite.Group()

        if MainClass.enemyGroup2!= None :
            MainClass.enemyGroup2.empty()
        MainClass.enemyGroup2 = pygame.sprite.Group()

        if MainClass.explodeGroup!=None:
            MainClass.explodeGroup.empty()
        MainClass.explodeGroup = pygame.sprite.Group()

    def drawObject(self, obj, x, y):
        MainClass.gamepad.blit(obj, (x, y))

    def drawScore(self):
        text = str(MainClass.score)
        largeText = pygame.font.Font('freesansbold.ttf', 30)
        textSurface = largeText.render(text, True, (0, 0, 255))
        textRect = textSurface.get_rect()
        textRect.x = MainClass.pad_width - 100
        textRect.y = 10
        MainClass.gamepad.blit(textSurface, textRect)
        # pygame.display.update()

    def gameover(self):
        text = 'Game Over!'
        largeText = pygame.font.Font('freesansbold.ttf', 115)
        textSurface = largeText.render(text, True, (255, 0, 0))
        textRect = textSurface.get_rect()
        textRect.center = ((MainClass.pad_width / 2), (MainClass.pad_height / 2))
        MainClass.gamepad.blit(textSurface, textRect)
        pygame.display.update()

    def gameready(self):
        size = 50
        largeText = pygame.font.Font('freesansbold.ttf', size)
        col = (200, 40, 40)
        msgs = [ 'Start: SpaceBar', 'Fire: Left Ctrl', 'ESC: Quit' ]
        for i, text in enumerate(msgs):
            textSurface = largeText.render(text, True, col)
            textRect = textSurface.get_rect()
            textRect.center = ((MainClass.pad_width / 2), (MainClass.pad_height / 2)+i*size)
            MainClass.gamepad.blit(textSurface, textRect)

    # 최초에 한 번 로딩
    def initFirst(self):

        MainClass.sound_shot = pygame.mixer.Sound('res/laser02.wav')
        MainClass.sound_explosion = pygame.mixer.Sound('res/bomb01.wav')

        MainClass.gamepad = pygame.display.set_mode((MainClass.pad_width, MainClass.pad_height))
        pygame.display.set_caption('PyFlying')

        # bullet = pygame.image.load('bullet.png').convert_alpha()
        # boom = pygame.image.load('boom.png').convert_alpha()

        # background music
        pygame.mixer.music.load('res/bgm01.mp3')
        pygame.mixer.music.play(-1)

        MainClass.clock = pygame.time.Clock()

    def runGame(self):
        self.initVars()

        while not MainClass.crashed:
            t = pygame.time.get_ticks()
            # get key event
            bshot = False

            if MainClass.gamestatus == MainClass.STATUS_PLAY:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        crashed = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            crashed = True
                        elif event.key == pygame.K_SPACE:
                            pass
                        elif event.key == pygame.K_LCTRL and (t - MainClass.lastBullet) > MainClass.bulletDelay:
                            bshot = True
                            print("shot!")
                    if event.type == pygame.KEYUP:
                        MainClass.plane.move_middle()

                # get key status (continous)
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    MainClass.plane.move_up()
                if keys[pygame.K_DOWN]:
                    MainClass.plane.move_down()
                if keys[pygame.K_RIGHT]:
                    MainClass.plane.move_forward()
                if keys[pygame.K_LEFT]:
                    MainClass.plane.move_backward()

                # background first
                MainClass.gamepad.fill(white)
                MainClass.backGroup.update()

                # enemy spawn.  적 생성....
                if len(MainClass.enemyGroup.sprites()) == 0 and random.randint(0, 10) == 0:
                    enemy = Enemy1()
                    MainClass.enemyGroup.add(enemy)
                elif random.randint(0, 30) == 0:
                    enemy = Enemy1()
                    MainClass.enemyGroup.add(enemy)

                if len(MainClass.enemyGroup2.sprites()) == 0 and random.randint(0, 10) == 0:
                    enemy = Enemy2()
                    MainClass.enemyGroup2.add(enemy)
                elif random.randint(0, 30) == 0:
                    enemy = Enemy2()
                    MainClass.enemyGroup2.add(enemy)

                # bullet spawn... 총알 생성...
                if bshot:
                    bshot = False
                    bullet = Bullet()
                    MainClass.bulletGroup.add(bullet)
                    if MainClass.plane.power>=2 :
                        bullet2 = Bullet()
                        bullet2.set_type(1)
                        MainClass.bulletGroup.add(bullet2)
                    if MainClass.plane.power>=3 :
                        bullet3 = Bullet()
                        bullet3.set_type(2)
                        MainClass.bulletGroup.add(bullet3)
                    MainClass.sound_shot.play()

                # player  update
                MainClass.plane.update()
                MainClass.bulletGroup.update()
                MainClass.enemyGroup.update()  # 업데이트
                MainClass.enemyGroup2.update()
                MainClass.explodeGroup.update()
                # for e in enemyGroup.sprites():
                #     e.draw(gamepad)

                # 충돌 감지
                collided = pygame.sprite.groupcollide(MainClass.planeGroup, MainClass.enemyGroup, False, False, pygame.sprite.collide_mask)
                if collided:
                    MainClass.sound_explosion.play()
                    MainClass.plane.set_status(MainClass.STATUS_DEAD)
                    MainClass.plane.update()

                collided = pygame.sprite.groupcollide(MainClass.planeGroup, MainClass.enemyGroup2, False, False, pygame.sprite.collide_mask)
                if collided:
                    MainClass.sound_explosion.play()
                    MainClass.plane.set_status(MainClass.STATUS_DEAD)
                    MainClass.plane.update()
                    for c1, c2 in collided.items():
                        c2[0].set_status(MainClass.STATUS_EXPLODE)
                        c2[0].update()

                collided = pygame.sprite.groupcollide(MainClass.bulletGroup, MainClass.enemyGroup2, True, False, pygame.sprite.collide_mask)
                if collided:
                    for c1, c2 in collided.items():
                        for e in c2:
                            MainClass.sound_explosion.play()
                            e.set_status(MainClass.STATUS_EXPLODE)
                            MainClass.score += e.score
                            e.update()
                            MainClass.enemyGroup2.remove(e)
                            MainClass.explodeGroup.add(e)

                ## 그리기...
                MainClass.backGroup.draw(MainClass.gamepad)     # 이 위에서 그리면 배경으로 덮어져서 보이지 않는다.

                MainClass.enemyGroup.draw(MainClass.gamepad)
                MainClass.enemyGroup2.draw(MainClass.gamepad)
                MainClass.planeGroup.draw(MainClass.gamepad)
                MainClass.bulletGroup.draw(MainClass.gamepad)
                MainClass.explodeGroup.draw(MainClass.gamepad)
                self.drawScore()

                # obj_pos = [[bat_x, bat_y], [bat_x + bat_width, bat_y],
                #            [bat_x, bat_y + bat_height], [bat_x + bat_width, bat_y + bat_height]]
                # update
                # pygame.display.update()
                pygame.display.flip()
                MainClass.clock.tick(MainClass.FPS)

                if MainClass.plane.status == MainClass.STATUS_DEAD:
                    self.gameover()
                    pygame.mixer.music.stop()
                    sleep(2)
                    MainClass.plane.status = MainClass.STATUS_READY
                    pygame.mixer.music.load('res/bgm01.mp3')
                    pygame.mixer.music.play(-1)
                    self.initVars()

            elif MainClass.gamestatus == MainClass.STATUS_READY:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        crashed = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            MainClass.crashed = True
                        elif event.key == pygame.K_SPACE:
                            MainClass.gamestatus = MainClass.STATUS_PLAY
                # background first
                MainClass.gamepad.fill(white)
                MainClass.backGroup.draw(MainClass.gamepad)
                self.gameready()
                pygame.display.flip()
                MainClass.clock.tick(MainClass.FPS)
            else:
                print('unknown', MainClass.gamestatus)

        pygame.mixer.quit()
        pygame.quit()
        quit()


if __name__ == '__main__':
    game = MainClass()
    game.runGame()

