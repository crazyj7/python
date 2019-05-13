import pygame
import random
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
aircraft_width=0
aircraft_height=0
aircraft_height=None
aircraft_width=None
bat_height=None
bat_width=None
fire_height=0
fire_width=0

#sound
sound_shot=None
sound_explosion=None

# gap. smaller box
def checkInBox(pos, box, gap):
    if box[0]+gap<=pos[0] and box[2]-gap>=pos[0] and \
        box[1]+gap<=pos[1] and box[3]-gap>=pos[1]:
        return True
    return False

def checkCollision(list1, list2, gap):
    box = [ list2[0][0], list2[0][1], list2[3][0], list2[3][1]]
    for pos in list1:
        if checkInBox(pos, box, gap):
            return True
    box = [ list1[0][0], list1[0][1], list1[3][0], list1[3][1]]
    for pos in list2:
        if checkInBox(pos, box, gap):
            return True
    return False




def drawObject(obj, x, y):
    # global gamepad
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
            pygame.mixer.Sound.play(sound_shot)

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
        y = min(y, pad_height-aircraft_height)
        drawObject(aircraft, x,y)
        # collision
        finc=False
        air_pos=[[x,y], [x+aircraft_width, y], [x, y+aircraft_height], [x+aircraft_width, y+aircraft_height]]
        if fire!=None:
            obj_pos = [[fire_x, fire_y], [fire_x+fire_width, fire_y],
                    [fire_x, fire_y+fire_height], [fire_x+fire_width, fire_y+fire_height]]
            if checkCollision(air_pos, obj_pos, 10):
                # game over
                drawObject(boom, x, y)
                pygame.mixer.Sound.play(sound_explosion)
                return gameover()

        obj_pos = [[bat_x, bat_y],[bat_x + bat_width, bat_y],
                   [bat_x, bat_y + bat_height], [bat_x + bat_width, bat_y + bat_height]]
        finc = False
        if checkCollision(air_pos, obj_pos, 10):
            # game over
            drawObject(boom, x, y)
            pygame.mixer.Sound.play(sound_explosion)
            return gameover()

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
        clock.tick(60)
    pygame.quit()
    quit()


def initGame():
    global gamepad, aircraft, clock, background1, background2
    global bat, fires, bullet, aircraft_height, aircraft_width
    global bat_width, bat_height, boom
    global sound_shot, sound_explosion

    fires=[]

    pygame.init()

    sound_shot = pygame.mixer.Sound('shot.wav')
    sound_explosion = pygame.mixer.Sound('explosion.wav')

    gamepad = pygame.display.set_mode((pad_width, pad_height))
    pygame.display.set_caption('PyFlying')

    aircraft = pygame.image.load('plane.png')

    aircraft_width = aircraft.get_width()
    aircraft_height = aircraft.get_height()
    print('aircraft size=', aircraft_width, aircraft_height)

    background1 = pygame.image.load('back2.png')
    background2 = background1.copy()

    bat = pygame.image.load('bat.png')
    bat_height = bat.get_height()
    bat_width = bat.get_width()

    fire1 = pygame.image.load('fireball.png')
    fire_width = fire1.get_width()
    fire_height = fire1.get_height()
    fires.append(fire1)
    fires.append(pygame.image.load('fireball2.png'))

    for i in range(5):
        fires.append(None)

    bullet = pygame.image.load('bullet.png')

    boom = pygame.image.load('boom.png')

    # background music
    # pygame.mixer.music.load('mybgm.wav')
    # pygame.mixer.music.play(-1)


    clock = pygame.time.Clock()
    runGame()


if __name__=='__main__':
    initGame()

