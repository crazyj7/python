__author__ = 'crazyj'

from gym import envs

# print( envs.registry.all())

for node in envs.registry.all():
    print( node )

