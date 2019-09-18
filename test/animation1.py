'''
animation plot
가장 단순한 그래프를 x축이 이동하는 것처럼 그리기
x축 도메인을 변경하면서, y값을 계산한다.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

# 0~2pi까지의 도메인을 만든다.
x = np.arange(0, 2*np.pi, 0.01)
# 주어진 도메인으로 sin 곡선을 ax에 그린다.
line, = ax.plot(x, np.sin(x))

def animate(i):
    # 화면을 업데이트한다. y값 변경
    line.set_ydata(np.sin(x+i/10.0))
    return line,

# 애니매이션 파라미터 설정. 업데이트콜백 함수, 입력 파라미터, 초기화함수, 업데이트 주기.
ani = animation.FuncAnimation(fig, animate, np.arange(1, 100),
                              interval=25, blit=True, repeat=True)
plt.show()













