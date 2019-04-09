import os, sys

class MyUtil:
    def __init__(self):
        return

    @staticmethod
    def set_cwd():
        d = os.path.dirname( os.path.realpath(__file__) )
        print('dir=',d)
        os.chdir(d)

