'''
animation plot
가장 단순한 그래프를 천천히 그리기
x축 도메인을 조금씩 늘려서 그린다.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

# 0~2pi까지의 도메인을 만든다.
x = np.arange(0, 2*np.pi, 0.01)
xcnt=len(x)
xcur = 1

print(xcnt)

ax.set_xlim([0, 2*np.pi])
ax.set_ylim([-2, 2])

# 주어진 도메인으로 sin 곡선을 ax에 그린다.
line, = ax.plot(x[:xcur], np.sin(x[:xcur]))

def animate(i):
    # 화면을 업데이트한다. y값 변경
    xcur = int(xcnt*i/100)
    line.set_xdata(x[:xcur])
    line.set_ydata(np.sin(x[:xcur]))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 100),
                              interval=10, blit=False, repeat=True)
plt.show()













