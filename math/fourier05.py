'''
fast fft
scipy fftpack
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

N = 1024
T = 1.0 / 44100.0
f1 = 697
f2 = 1209
t = np.linspace(0.0, N*T, N)
y1 = 1.1 * np.sin(2 * np.pi * f1 * t)
y2 = 0.9 * np.sin(2 * np.pi * f2 * t)
y = y1 + y2

if False:
    plt.subplot(311)
    plt.plot(t, y1)
    plt.title(r"$1.1\cdot\sin(2\pi\cdot 697t)$")
    plt.subplot(312)
    plt.plot(t, y2)
    plt.title(r"$0.9\cdot\sin(2\pi\cdot 1209t)$")
    plt.subplot(313)
    plt.plot(t, y)
    plt.title(r"$1.1\cdot\sin(2\pi\cdot 697t) + 0.9\cdot\sin(2\pi\cdot 1209t)$")
    plt.tight_layout()
    plt.show()

# 고속 퓨리에 변환 . Fast FFT
# 길이가 2^N 인 시계열에만 적용 가능. python의 np.fft는 크기가 2^N이 아니어도 처리 가능. 내부적으로 처리하는 듯.
# 계산량이 일반은 O(N^2) 이지만 Fast FFT는 O(Nlog_2 N)
# 시계열의 시작과 끝부분의 오차가 크면 에러신호가 나타난다. (깁스 현상)
if False:
    y2 = np.hstack([y, y, y])

    plt.subplot(211)
    plt.plot(y2)
    plt.axvspan(N, N * 2, alpha=0.3, color='green')
    plt.xlim(0, 3 * N)

    plt.subplot(212)
    plt.plot(y2)
    plt.axvspan(N, N * 2, alpha=0.3, color='green')
    plt.xlim(900, 1270)

    plt.show()


# 주파수 성분 분석
if True:
    yf = fft(y, N)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    plt.stem(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.xlim(0, 3000)

    plt.show()





