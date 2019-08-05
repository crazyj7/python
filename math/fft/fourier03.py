'''
fourier transform

dt = sampling term.
forward ; X = fft(x) dt
backward ; x = ifft(X) / dt

ex) 60Hz, 120Hz sin curve
x(t) = 0.7sin(120 pi t)+sin(240pi t)
noise = N(0, 2) 노이즈가 포함된 신호 분석 예제
sampling = 100Hz.
data count=1500 count.


'''

import numpy as np
import matplotlib.pyplot as plt

fmax=1000   # sampling max frequency 100 Hz = f = 100cycle/1sec
dt = 1/ 1000   # sampling period
N=1500  # sample count

t = np.arange(0, N)*dt
print('sample t=', t)

# original signal ; 60Hz, 120Hz 사용.
s = 0.7*np.sin(2*np.pi*60*t)+np.sin(2*np.pi*120*t)

# detect signal ; noise added  노이즈가 추가된 신호. x
x = s + 0.8*np.random.randn(N)

plt.subplot(2,1,1)
plt.plot(t[0:200], s[0:200], label='signal')
plt.plot(t[0:200], x[0:200], label='x')
plt.legend()
plt.xlabel('time'), plt.ylabel('x(t)'), plt.grid()
# plt.show()

# fourier spectrum...
df = fmax/N
f = np.arange(0,N)*df   # 1Hz 2Hz, ... ,fmax Hz.
xf = np.fft.fft(x)*dt

plt.subplot(2,1,2)
cnt = int(N/2+1)
plt.plot(f[0:cnt], np.abs(xf[0:cnt]))
plt.xlabel('freq(Hz)'), plt.ylabel('abs(xf)'), plt.grid()
plt.show()

# 퓨리에 스펙트럼에서 peak를 치는 곳의 주파수 성분들을 사용하면 된다.




