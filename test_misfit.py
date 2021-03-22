from seismic import demo_model, AcquisitionGeometry, Receiver
import numpy as np
import matplotlib.pyplot as plt

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
bfm_solver = bfm(num_steps=10, step_scale=1., verbose=True)
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

loss2, grad2 = w2d(f2, g2)
print(loss2)

# plt.imshow(grad2, aspect=ntr/nt)
# plt.show()

# plt.plot(grad, 'b--')
# plt.plot(grad2[:, int(ntr/2)], 'r')
# plt.show()

# Set up velocity model
shape = (101, 101)      # Number of grid points (nx, nz).
spacing = (10., 10.)    # Grid spacing in m. The domain size is now 1km by 1km.
origin = (0, 0)         # Need origin to define relative source and receiver locations.
nbl = 40

# True model
model1 = demo_model('circle-isotropic', vp_circle=3.0, vp_background=2.5,
    origin=origin, shape=shape, spacing=spacing, nbl=nbl)
print(model1.critical_dt)
# Initial model
model0 = demo_model('circle-isotropic', vp_circle=2.5, vp_background=2.5,
    origin=origin, shape=shape, spacing=spacing, nbl=nbl, grid = model1.grid)
print(model0.critical_dt)
#model0._dt = model1.critical_dt
# Set up acquisiton geometry
t0 = 0.
tn = 1000. 
f0 = 0.010

# Set up source geometry, but define 5 sources instead of just one.
nsources = 1
src_coordinates = np.empty((1, 2))
src_coordinates[:, 1] = model1.domain_size[0]/2
src_coordinates[:, 0] = 20.  # Source depth is 20m

# Initialize receivers for synthetic and imaging data
nreceivers = 101
rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 1] = np.linspace(spacing[0], model1.domain_size[0] - spacing[0], num=nreceivers)
rec_coordinates[:, 0] = 980.    # Receiver depth
# Set up geometry objects for observed and predicted data
geometry1 = AcquisitionGeometry(model1, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
geometry0 = AcquisitionGeometry(model0, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')

# client = Client(processes=False)
stride = 1
obs = fm_single(geometry1, save=False, dt=stride)[0]

syn = fm_single(geometry0, save=False, dt=geometry0.dt)[0]

data1 = obs.resample(geometry0.dt).data[:][0:syn.data.shape[0], :]

data2 = syn.data

loss, grad = w2d(data2, data1)

plt.imshow(grad, aspect=grad.shape[1]/grad.shape[0])
plt.show()