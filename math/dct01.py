'''
DCT ; discrete cosine transform

'''

from scipy.fftpack import dct
import matplotlib.pyplot as plt
import numpy as np

N = 1024
T = 1.0 / 44100.0
f1 = 697
f2 = 1209
t = np.linspace(0.0, N*T, N)
y1 = 1.1 * np.sin(2 * np.pi * f1 * t)
y2 = 0.9 * np.sin(2 * np.pi * f2 * t)
y = y1 + y2



dct_type=2
yf2 = dct(y, dct_type, N)
#
# plt.subplot(311)
# plt.stem(np.real(yf))
# plt.title("DFT 실수부")
#
# plt.subplot(312)
# plt.stem(np.imag(yf))
# plt.title("DFT 허수부")

plt.subplot(313)
plt.stem(np.abs(yf2))
plt.title("DCT")

plt.tight_layout()
plt.show()




