import matplotlib.pyplot as plt
import math
import numpy as np


'''
r = cos (t)
'''

t = np.arange(0, math.pi*2, 0.01)
r = np.cos(t)
print(t)
print(r)

## change to x, y
x = r * np.cos(t)
y = r * np.sin(t)

'''
r2=1-cos(t)
'''
r2 = 1-np.cos(t)
x2 = r2 * np.cos(t)
y2 = r2 * np.sin(t)


plt.figure()
plt.title('graph')
plt.plot(x,y, c='r', label='r=cos(t)')
plt.plot(x2,y2, c='b', label='r=1-cos(t)')
plt.axhline()
plt.axvline()
plt.legend()
plt.show()


