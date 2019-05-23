'''

원형 모양으로 퍼지는 형태 (방사형)

'''

import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc
import random

fig, ax = plt.subplots()
ax.set_xlim((-100, 300))
ax.set_ylim((-100, 300))
line, = ax.plot([], [], lw=2)

b=0

def animate(i, mode, N):
    global b
    if i==50:
        b = random.random() * (2 * math.pi / N)  # bias
        print('b=', b)
    print('mode=', mode)
    center = (100,100)
    xlist = []
    ylist = []
    r=i
    print('r=', r)
    for i in range(N):
        theta = i*2*math.pi / N + b
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        xlist.append( x+center[0] )
        ylist.append( y+center[1] )

    # line.set_data(xlist, ylist)
    # return (line,)
    ax.clear()
    ax.set_xlim((-100, 300))
    ax.set_ylim((-100, 300))
    return ax.scatter(xlist, ylist, c='red', lw=3)

def init():
    line.set_data([],[])
    return (line,)

mode = 1
N = 10
# anim = animation.FuncAnimation( fig, animate, init_func = init, fargs=(mode),
#                                 frames=np.arange(10,100), interval=20, blit=False, repeat=False)
# anim = animation.FuncAnimation( fig, animate, init_func = init,
#                                 frames=100, interval=20, blit=False, repeat=False)
anim = animation.FuncAnimation( fig, animate, init_func = init,
                                frames=np.arange(50, 200, 5), interval=1, blit=False, repeat=True,
                                fargs=(mode, N))

plt.show()

