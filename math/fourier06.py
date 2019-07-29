'''
drawing to fft

'''


import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splrep, spleval, splev

# fmax=dt*cnt  # 찾을 최대 주파수
fmax= 1024
maxfcnt = 10


def make_drawing():
    dt = 2*np.pi/100
    x=[]
    y=[]
    r=10
    for theta in np.arange(0, (5/2)*np.pi, dt):
        tx=r*np.cos(theta)
        ty=r*np.sin(theta)
        x.append(tx)
        y.append(ty)
        r -= 0.01
    for theta in np.arange(0, 20, dt):
        ty -= dt
        x.append(tx)
        y.append(ty)
    x.append(x[0])
    y.append(y[0])

    cnt = len(x)
    x=np.array(x)
    y=np.array(y)
    noise = np.random.randn(cnt)*0.08
    noise2 = np.random.randn(cnt)*0.08
    x+=noise
    y+=noise2

    # interpolation
    t = np.linspace(0, 1, len(x))
    t1 = np.linspace(0, 1, fmax)

    spl = splrep(t, x)
    x1 = splev(t1, spl)

    spl = splrep(t, y)
    y1 = splev(t1, spl)

    # normalization
    x1 /= np.max(x1)
    y1 /= np.max(y1)
    return t1, x1, y1

t, x, y = make_drawing()


# 그림.
if False:
    plt.figure()
    # plt.plot(t, x)
    plt.plot(x, y)
    plt.show()

# x축만 분리하여 그래프.
if False:
    plt.figure()
    plt.plot(t, x)
    plt.show()


def predict(t, x):
    cnt = t.shape[0]
    dt = t[-1]/cnt

    print('sample count=', cnt)
    print('dt=', dt, 'maxtime=', t[-1])

    # FFT 분석
    print('fmax=', fmax)
    dt2 = 1/fmax     # 샘플링 시간
    N = cnt      # 샘플 개수

    if True:
        plt.subplot(3,1,1)
        plt.plot(t, x, label='x')
        plt.legend()
        plt.xlabel('time'), plt.ylabel('x(t)'), plt.grid()
        # plt.show()

    # fourier spectrum...
    df = fmax/N
    print('df =', df)
    f = np.arange(0,N)*df   # 1Hz 2Hz, ... ,fmax Hz.
    print('f:', f[0], f[-1])
    xf = np.fft.fft(x)*dt * 2           ## ????????????????????????????????????????
    print('dt=', dt)

    if True:
        plt.subplot(3,1,2)
        plt.plot(f, np.abs(xf))
        plt.xlabel('freq(Hz)'), plt.ylabel('abs(xf)'), plt.grid()
        # plt.show()

    # 퓨리에 스펙트럼에서 peak를 치는 곳의 주파수 성분들을 사용하면 된다.

    # get Top N
    # print('f=', f)
    xf2 = np.abs(xf[0:int(cnt/2)])
    # print('xf2=', xf2)
    print('xf mean=', np.mean(xf2))
    # print('sort=', -np.sort(-xf2))
    rankidx = np.argsort(-xf2)


    # print(rankidx)
    for i in range(maxfcnt):
        # print(i, rank[i])
        print('freq=', f[rankidx[i]], 'ampl abs=', xf2[rankidx[i]])

    # draw by fft top.
    plt.subplot(3,1,3)
    yf = np.zeros((cnt,))
    for i in range(maxfcnt):
        freq = f[rankidx[i]]
        amp = xf2[rankidx[i]]
        print('freq, and amp = ', freq, amp, xf[rankidx[i]])
        amp1 = xf[rankidx[i]].real
        amp2 = xf[rankidx[i]].imag
        amp2 = amp2 * (-1)
        yf += amp1*np.cos(2*np.pi*freq*t) + amp2*np.sin(2*np.pi*freq*t)
    plt.plot(t, x, label='signal')
    plt.plot(t, yf, label='fft')
    plt.legend()
    plt.show()
    return yf

px = predict(t,x)
py = predict(t,y)


# 그림.
if True:
    plt.subplot(2,1,1)
    plt.plot(x, y)
    plt.subplot(2,1,2)
    plt.plot(px, py)
    plt.show()


