import pygame
import random

white = (255,255,255)
pad_width=1024
pad_height=512
background_width=1024
aircraft_width=0
aircraft_height=0
gamepad=None
aircraft=None
clock=None
background1=None
background2=None
bat=None
fires=None
bullet=None
aircraft_height=None
aircraft_width=None

def drawObject(obj, x, y):
    # global gamepad
    gamepad.blit(obj, (x,y))


def runGame():
    # global gamepad, clock, aircraft, background1, background2
    # global bat, fires, bullet

    bullet_xy=[]

    x = pad_width * 0.05
    y = pad_height * 0.4
    y_change = 0

    crashed= False
    background1_x=0
    background2_x = background_width

    bat_x=pad_width
    bat_y=random.randrange(0, pad_height)

    fire_x = pad_width
    fire_y=random.randrange(0, pad_height)
    random.shuffle(fires)
    fire = fires[0]


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
                if event.key==pygame.K_UP or event.key==pygame.K_DOWN:
                    y_change=0

        # get key status (continous)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            y_change = -5
        elif keys[pygame.K_DOWN]:
            y_change = 5

        if bshot:
            bshot=False
            bullet_x = x + aircraft_width
            bullet_y = y + aircraft_height/2
            bullet_xy.append([bullet_x, bullet_y])

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
        if fire==None:
            fire_x -=30
        else:
            fire_x -= 15
        if fire_x<=0:
            random.shuffle(fires)
            fire=fires[0]
            fire_x = pad_width
            if fire!=None:
                fire_y = random.randrange(0, pad_height-fire.get_height())

        drawObject(bat, bat_x, bat_y)
        if fire!=None:
            drawObject(fire, fire_x, fire_y)


        # player
        y+=y_change
        y = max(y, 0)
        y = min(y, pad_height-55)
        drawObject(aircraft, x,y)

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



        # update
        pygame.display.update()
        clock.tick(60)
    pygame.quit()
    quit()


def initGame():
    global gamepad, aircraft, clock, background1, background2
    global bat, fires, bullet, aircraft_height, aircraft_width

    fires=[]

    pygame.init()
    gamepad = pygame.display.set_mode((pad_width, pad_height))
    pygame.display.set_caption('PyFlying')

    aircraft = pygame.image.load('res/plane.png')

    aircraft_width = aircraft.get_width()
    aircraft_height = aircraft.get_height()
    print('aircraft size=', aircraft_width, aircraft_height)

    background1 = pygame.image.load('res/back2.png')
    background2 = background1.copy()

    bat = pygame.image.load('res/bat.png')

    fires.append(pygame.image.load('res/fireball.png'))
    fires.append(pygame.image.load('res/fireball2.png'))
    for i in range(5):
        fires.append(None)

    bullet = pygame.image.load('res/bullet.png')

    clock = pygame.time.Clock()
    runGame()


if __name__=='__main__':
    initGame()

