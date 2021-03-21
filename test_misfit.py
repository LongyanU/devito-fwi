import numpy as np
import matplotlib.pyplot as plt
from bfm import bfm
from misfit import qWasserstein

def wavelet(dt, n, freq, delay):
	t = np.arange(0, n) * dt
	delay = delay * dt
	t = t - delay
	tmp = np.pi * np.pi * freq * freq * t * t
	y = (1. - 2.*tmp) * np.exp(-tmp)

	return y.reshape(n, 1)

w1d = qWasserstein(trans_type='linear', gamma=1.01, method='1d')
w2d = qWasserstein(trans_type='linear', gamma=1.01, method='2d')

dt = 0.001
nt = 1000
ntr = 100
freq = 5
f = wavelet(dt, nt, freq, 200)
g = wavelet(dt, nt, freq, 500)

loss, grad = w1d(f, g)
print(loss)
plt.plot(grad)
plt.show()

f2 = np.tile(f, (1, ntr))
g2 = np.tile(g, (1, ntr))

loss2, grad2 = w2d(f2, g2)
print(loss2)
plt.plot(grad, 'b--')
plt.plot(grad2[:, int(ntr/2)], 'r')
plt.show()