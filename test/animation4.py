
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import math

xlist = np.arange(-3*np.pi, 3*np.pi, 0.01)

def ffs(x, n):
    a_0 = 1/2
    val = a_0
    if n==0:
        return val
    for i in range(1, n+1):
        an = math.sin( i*np.pi) / (i*np.pi)
        bn = (1-math.cos(i*np.pi)) / (i*np.pi)
        val += an* math.cos(i*x) + bn*math.sin(i*x)
    return val

n = 1
ylist = [ffs(x, n) for x in xlist]
fig, ax = plt.subplots()
line, = ax.plot( xlist, ylist)
plt.grid(True)

def animate(i):
    n = i
    ax.set_title('n={}'.format(n))
    ylist = [ffs(x, n) for x in xlist]
    line.set_ydata( ylist)
    return line,

# 50ms 단위로 갱신하는 animate 함수 호출. 파라미터는 1~50까지 차례로 준다. 종료시 반복하지 않음.
ani = animation.FuncAnimation(fig, animate, np.arange(1, 51), interval=50, blit=False, repeat=False)
plt.show()
