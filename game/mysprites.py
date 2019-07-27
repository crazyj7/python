
import os
import pygame


class MySprites():
    def __init__(self):
        self.sprites = []
        self.width=0
        self.height=0
        self.filepath=''

    def load(self, path):
        self.filepath = path
        self.image = pygame.image.load(self.filepath).convert_alpha()



pygame.init()
pygame .display.set_caption('test')
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()

def main():
    while True:
        for event in pygame.event.get():
            if event==pygame.QUIT:
                break
