'''
우도 ;  표본으로 모수를 추정했을 때 모수가 그럴 듯 할 확률.
표본=x, 모수=q
우도 = L(q|x) = 비례 P ( x | q )

동전 던지기 ; 100번 던져서 56번 성공(앞면)함.
모수 ; 이 동전은 정상이다. q=1/2
우도 = L(q | x )
'''

import numpy as np
import math
import matplotlib.pyplot as plt


import math

def nCr(n,r):
    f=math.factorial
    return f(n) / f(r) / f(n-r)

def nPr(n,r):
    f=math.factorial
    return f(n) / f(n-r)



unit=0.02
qlst = np.arange(0.1, 1, unit)
print ('qlst=', qlst)

# 각 q에 따른 우도는? L(q|x) ~ P(x|q)
def Lq(q):
    return nCr(100, 56) * math.pow(q, 56) * math.pow(1-q, 100-56)

print('q=0.5', Lq(0.5))

l = np.array(list(map(Lq, qlst)))
print('qlist=', qlst)
print('Likelyhood=', l)

idx = np.argmax( l )
print('maximum likelyhood = ', np.max(l), ' q=', 0.1+idx*unit)

plt.figure()
plt.plot(qlst, l)
plt.show()

