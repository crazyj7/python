
'''


'''

import pygame

WHITE = (255,255,255)
pad_width = 1024
pad_height = 512
clock = None

gamepad = None
aircraft = None
background = None
background_width = 1024



def back(x, y):
    global gamepad, background
    gamepad.blit(background, (x,y))
    gamepad.blit(background, (x+background_width, y))

def airplane(x, y):
    global gamepad, aircraft
    gamepad.blit(aircraft, (x,y))


def runGame():
    global gamepad, aircraft, background
    global clock

    x=pad_width * 0.05
    y = pad_height*0.8
    y_change=0

    back_x = 0
    speed = 5

    crashed = False
    while not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    crashed = True
                if event.key == pygame.K_UP:
                    y_change = -5
                elif event.key == pygame.K_DOWN:
                    y_change = 5
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP or event.key==pygame.K_DOWN:
                    y_change = 0
        y += y_change
        back_x -= speed
        if back_x < -background_width:
            back_x+=background_width

        gamepad.fill(WHITE)
        back(back_x, 0)
        airplane(x,y)

        pygame.display.update()
        clock.tick(60)      # FPS 60
    pygame.quit()


def initGame():
    global gamepad, clock, aircraft, background

    pygame.init()
    gamepad = pygame.display.set_mode((pad_width, pad_height))
    pygame.display.set_caption('shoot')
    aircraft = pygame.image.load('resource/rocket2.png')        # 50x100
    background = pygame.image.load('resource/back.png')         # 1024x512
    ship1 = pygame.image.load('resource/ship1_r.png')


    clock = pygame.time.Clock()
    runGame()


initGame()

