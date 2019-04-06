
import matplotlib.pyplot as plt
import numpy as np
import math

Fs=2000
T = 1/Fs
te=0.5    # time of elapsed
t = np.arange(0, te, T)


# sum of 50 Hz
noise = np.random.normal(0, 0.05, len(t))

x = 0.6*np.cos( 2*np.pi*60*t +np.pi/2) + np.cos(2*np.pi*120*t )
y = x

plt.figure(num=1, dpi=100, facecolor='white')
plt.plot(t, y, 'r')
plt.xlim(0, 0.05)
plt.xlabel('time($sec$)')
plt.ylabel('y')
plt.show()



