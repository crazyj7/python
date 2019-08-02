'''
fourier transform

입력 신호를 보고 숨겨진 주파수를 찾아라.
sin, cos으로 구성된 신호, 주파수와 진폭을 어떻게 구하는가.

샘플 개수와, 최대 주파수의 관계. fft 사용법.

authors: crazyj7
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splrep, splev

# 원본 시그널. 1초간 발생. 그 동안 수집된 샘플 개수는 2000개.
ST=1
SN=2000  # 샘플 개수
sdt = ST/SN
t = np.linspace(0, ST, SN)
# original signal ; 60Hz, 120Hz 사용.
freq1=80
freq2=250
s = 0.8*np.sin(2*np.pi*freq1*t)-0.4*np.cos(2*np.pi*freq1*t)\
    +0.3*np.sin(2*np.pi*freq2*t)+0.3*np.cos(2*np.pi*freq2*t)
# detect signal ; noise added  노이즈가 추가된 신호. x
x = s + 0.1*np.random.randn(SN)

# 원본 모양
ph=4
pw=1
pidx=0
pidx+=1
plt.subplot(ph, pw, pidx)
plt.subplots_adjust(hspace=0.8)
plt.title('t-x')
plt.plot(t[:600], x[:600], label='signal')      # 데이터가 길기 때문에 일부만 출력!
plt.legend()
plt.xlabel('time'), plt.ylabel('x(t)'), plt.grid()
print('sample time:', ST, 'sample cnt=', SN, 'sample term=', sdt)
print('sample min:', np.min(x), 'max=', np.max(x))

# 샘플로 스플라인을 만들어 원하는 추정치 데이터를 산출한다.
spl = splrep(t, x)

### 분석
fmax=1000       # 찾을 최대 주파수
N = 500        # 샘플 개수.
dt = 1/fmax     # 샘플링 주기. 0.001 sec.

print('analyze max freq=', fmax, ' sample cnt=', N, 'dt=', dt)
print('analyze time:', dt*N)
# 샘플 시간. 0~1초를 N 구간으로 균일하게 나눈다.
t = np.arange(N)*dt  # 샘플링 주기시간을 0 단위부터 N(샘플개수) 까지. 즉 시간은 0 ~ 분석시간.
x2 = splev(t, spl)

if True:
    pidx+=1
    plt.subplot(ph, pw, pidx)
    plt.title('t-x')
    plt.plot(t, x2, label='signal new')      # 분석할 샘플을 추정하여 재구축
    plt.legend()
    plt.xlabel('time'), plt.ylabel('x(t)'), plt.grid()
    # plt.show()


# 푸리에 분석 스펙트럼.
# fourier spectrum...
df = fmax/N         # 주파수 증가분. (최대 주파수 / 샘플수).
print('df=', df)    # 주파수 증가분
# df 값이 작아야 촘촘한 주파수 검색이 가능. 즉 N(샘플수)가 많을 수록 좋다.

f = np.arange(0,N)*df   # 1Hz 2Hz, ... ,fmax Hz.
xf = np.fft.fft(x2)/N*2   # 신호를 입력으로 주고 fft 분석함.
# 출력결과에 2/N을 곱함!!! (half size만 봐서 에너지가 반으로 줄어듬. 전체 도메인에서 peak를 찾아 더한 거랑 결국 같다.)
# 스펙트럼 그래프. 가로축이 주파수인데, 중앙을 기준으로 좌우 대칭이다.

# 따라서 최대 fmax/2 만큼 주파수에 대해서 분석된다.
# 주파수값은 0~fmax/2 까지만 유효하다. 뒤에는 버림. 출력된 주파수 개수는 샘플개수(N)와 동일함.
pidx+=1
plt.subplot(ph, pw, pidx)
plt.plot(f, np.abs(xf))
plt.xlabel('freq(Hz)'), plt.ylabel('abs(xf)'), plt.grid()
# plt.show()

# 왼쪽 반만 사용한다.
halfcnt = int(N/2)
f0 = f[:halfcnt]
xf0 = xf[:halfcnt]

# 퓨리에 스펙트럼에서 peak를 치는 곳의 주파수 성분들을 사용하면 된다.
# get Top N
amp = np.abs(xf0)   # 벡터 norm으로 크기를 측정한다.
# print('amp=', amp)

# 크기가 큰 n개만 가져온다. 푸리에 계수 ; 주파수와 계수(cos, -sin)
topn = 10
rankidx = np.argsort(-amp)
for i in range(topn):
    print('frequency=', f0[rankidx[i]], 'coefficient(cos,-sin)=', xf0[rankidx[i]])

# 푸리에 역변환.  원본 신호를 복구해 본다.
pidx+=1
plt.subplot(ph, pw, pidx)
# t 0~200
x3 = np.zeros((N,))
for i in range(topn):
    freq=f0[rankidx[i]]
    ampt = xf0[rankidx[i]]
    print('freq, and ampt = ', freq, ampt)
    amp1 = ampt.real
    amp2 = ampt.imag * (-1)
    # real은 cos,  imagine은 sin 인데, 부호는 반대로 한다!
    x3 += amp1*np.cos(2*np.pi*freq*t)+amp2*np.sin(2*np.pi*freq*t)

plt.plot(t, x2, label='signal')
plt.plot(t, x3, label='fft')
plt.legend()
plt.show()

