'''
fourier transform

입력 신호를 보고 숨겨진 주파수를 찾아라.
sin으로 구성된 신호, 주파수와 진폭을 어떻게 구하는가.
'''

import numpy as np
import matplotlib.pyplot as plt

fmax=1000  # 찾을 최대 주파수
dt = 1/fmax     # 샘플링 시간
N = 2000      # 샘플 개수

# 샘플 시각
t = np.arange(0, N)*dt
# original signal ; 60Hz, 120Hz 사용.
s = 0.8*np.sin(2*np.pi*80*t)-0.4*np.sin(2*np.pi*200*t)+0.3*np.sin(2*np.pi*250*t)

# detect signal ; noise added  노이즈가 추가된 신호. x
# x = s + 0.8*np.random.randn(N)
x = s + 0.1*np.random.randn(N)

plt.subplot(3,1,1)
plt.plot(t[0:200], s[0:200], label='signal')
plt.plot(t[0:200], x[0:200], label='x')
plt.legend()
plt.xlabel('time'), plt.ylabel('x(t)'), plt.grid()
# plt.show()

# fourier spectrum...
df = fmax/N
f = np.arange(0,N)*df   # 1Hz 2Hz, ... ,fmax Hz.
xf = np.fft.fft(x)*dt

plt.subplot(3,1,2)
cnt = fmax
plt.plot(f[0:cnt], np.abs(xf[0:cnt]))
plt.xlabel('freq(Hz)'), plt.ylabel('abs(xf)'), plt.grid()
# plt.show()

# 퓨리에 스펙트럼에서 peak를 치는 곳의 주파수 성분들을 사용하면 된다.

# get Top N
print('f=', f)
xf2 = np.abs(xf[0:cnt])
# print('xf2=', xf2)
print('xf mean=', np.mean(xf2))
# print('sort=', -np.sort(-xf2))
rankidx = np.argsort(-xf2)
# print(rankidx)
for i in range(3):
    # print(i, rank[i])
    print(f[rankidx[i]], xf2[rankidx[i]])

# draw by fft top.
plt.subplot(3,1,3)
# t 0~200
yf = np.zeros((N,))
for i in range(3):
    freq = f[rankidx[i]]
    amp = xf2[rankidx[i]]
    print('freq, and amp = ', freq, amp)
    print(xf[rankidx[i]])
    if xf[rankidx[i]].imag > 0 :
        amp = amp * -1
    yf += amp*np.sin(2*np.pi*freq*t)
plt.plot(t[0:200], s[0:200], label='signal')
plt.plot(t[0:200], yf[0:200], label='fft')
plt.legend()
plt.show()

