'''

'''
import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc

fig, ax = plt.subplots()
ax.set_xlim((-100, 300))
ax.set_ylim((-100, 300))
line, = ax.plot([], [], lw=2)

def animate(i):
    N=10    # count of points
    center = (100,100)
    xlist = []
    ylist = []
    r=i
    print('r=', r)
    for i in range(N):
        theta = i*2*math.pi / N
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        xlist.append( x+center[0] )
        ylist.append( y+center[1] )

    line.set_data(xlist, ylist)
    return (line,)

def init():
    line.set_data([],[])
    return (line,)

anim = animation.FuncAnimation( fig, animate, init_func = init,
                                frames=100, interval=20, blit=True)

plt.show()

