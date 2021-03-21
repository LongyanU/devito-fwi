from seismic import demo_model, AcquisitionGeometry, Receiver
from seismic import plot_shotrecord, plot_image
import numpy as np
from scipy import optimize
from distributed import Client, wait, LocalCluster
from scipy import optimize
import matplotlib.pyplot as plt

from fwi import fm_multi, fwi_obj_multi, fwi_loss
from misfit import least_square, qWasserstein
from bfm import bfm
# Set up velocity model
shape = (101, 101)      # Number of grid points (nx, nz).
spacing = (10., 10.)    # Grid spacing in m. The domain size is now 1km by 1km.
origin = (0, 0)         # Need origin to define relative source and receiver locations.
nbl = 40

# True model
model1 = demo_model('circle-isotropic', vp_circle=3.0, vp_background=2.5,
    origin=origin, shape=shape, spacing=spacing, nbl=nbl)

# Initial model
model0 = demo_model('circle-isotropic', vp_circle=2.5, vp_background=2.5,
    origin=origin, shape=shape, spacing=spacing, nbl=nbl, grid = model1.grid)
model0._dt = model1.critical_dt
# Set up acquisiton geometry
t0 = 0.
tn = 1000. 
f0 = 0.010

# Set up source geometry, but define 5 sources instead of just one.
nsources = 5
src_coordinates = np.empty((nsources, 2))
src_coordinates[:, 1] = np.linspace(0, model1.domain_size[0], num=nsources)
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
obs = fm_multi(geometry1, save=False, dt=stride)

#plot_shotrecord(obs[2].data, model1, t0, tn)
qWmetric1d = qWasserstein(gamma=1.1, method='1d')
bfm_solver = bfm(shape=[obs[0].data.shape[1], obs[0].data.shape[0]], 
				num_steps=10, step_scale=8.)
qWmetric2d = qWasserstein(gamma=1.1, method='2d', bfm_solver=bfm_solver)

#misfit_func = least_square
misfit_func = qWmetric1d

f, g = fwi_obj_multi(geometry0, obs, misfit_func)

plot_image(g.reshape(model1.shape), cmap='cividis')

model_err = []
def fwi_callback(xk):
	nbl = model1.nbl
	v = model1.vp.data[nbl:-nbl, nbl:-nbl]
	m = 1. / (v.reshape(-1).astype(np.float64))**2
	model_err.append(np.linalg.norm((xk-m)/m))

# Box contraints
vmin = 1.4    # do not allow velocities slower than water
vmax = 4.0
bounds = [(1.0/vmax**2, 1.0/vmin**2) for _ in range(np.prod(model0.shape))]    # in [s^2/km^2]

# Initial guess
v0 = model0.vp.data[model0.nbl:-model0.nbl, model0.nbl:-model0.nbl]
m0 = 1.0 / (v0.reshape(-1).astype(np.float64))**2

# FWI with L-BFGS
ftol = 0.1
maxiter = 10
result = optimize.minimize(fwi_loss, m0, args=(geometry0, obs, misfit_func), method='L-BFGS-B', jac=True, 
    callback=fwi_callback, bounds=bounds, options={'ftol':ftol, 'maxiter':maxiter, 'disp':True})

# Plot FWI result
vp = 1.0/np.sqrt(result['x'].reshape(model1.shape))
plot_image(vp, vmin=2.5, vmax=3, cmap="cividis")
