from seismic import demo_model, AcquisitionGeometry, Receiver
from seismic import plot_shotrecord, plot_velocity, plot_image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fwi import fm_single
from misfit import qWasserstein
from bfm import bfm

def wavelet(dt, n, freq, delay):
	t = np.arange(0, n) * dt
	delay = delay * dt
	t = t - delay
	tmp = np.pi * np.pi * freq * freq * t * t
	y = (1. - 2.*tmp) * np.exp(-tmp)

	return y.reshape(n, 1)

w1d = qWasserstein(trans_type='linear', gamma=1.01, method='1d')
bfm_solver = bfm(num_steps=10, step_scale=8., verbose=True)
w2d = qWasserstein(trans_type='linear', gamma=1.01, method='2d', bfm_solver=bfm_solver)

dt = 0.001
nt = 1000
ntr = 100
freq = 5
f = wavelet(dt, nt, freq, 200)
g = wavelet(dt, nt, freq, 500)

loss, grad = w1d(f, g)
print(loss)
# plt.plot(grad)
# plt.show()

f2 = np.tile(f, (1, ntr))
g2 = np.tile(g, (1, ntr))

# loss2, grad2 = w2d(f2, g2)
# print(loss2)

# plt.imshow(grad2, aspect=ntr/nt)
# plt.show()

# plt.plot(grad, 'b--')
# plt.plot(grad2[:, int(ntr/2)], 'r')
# plt.show()

shape = (410, 101)
data1 = np.fromfile('./syn1', dtype=np.float32).reshape(shape)
data2 = np.fromfile('./obs1', dtype=np.float32).reshape(shape)

plt.imshow(data1, aspect=shape[1]/shape[0])
plt.show()

mu, nu, _ = w2d._transform(data1, data2)
print(mu.min(), nu.min())

loss, grad = w2d(data1, data2)
plt.imshow(grad, aspect=grad.shape[1]/grad.shape[0])
plt.show()

n1 = 512   # x axis
n2 = 512   # y axis


x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1), np.linspace(0.5/n2,1-0.5/n1,n2))

# Initialize densities
mu = np.zeros((n2, n1))
nu = np.zeros((n2, n1))
r = .25
mu[(x-.25)**2+(y-.25)**2<r**2] = 1
nu[(x-.75)**2+(y-.75)**2<r**2] = 1

# loss, grad = w2d(mu, nu)

