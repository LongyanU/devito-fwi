from seismic import Model, AcquisitionGeometry, Receiver
from seismic import plot_shotrecord, plot_velocity, plot_image
import numpy as np
from scipy import optimize
from distributed import Client, wait, LocalCluster
from scipy import optimize
import matplotlib.pyplot as plt

from fwi import Filter, fm_multi, fwi_obj_multi, fwi_loss
from misfit import least_square, qWasserstein
from bfm import bfm

# Setup velocity model
shape = (300, 106)      # Number of grid points (nx, nz).
spacing = (30., 30.)    # Grid spacing in m. The domain size is now 1km by 1km.
origin = (0, 0)         # Need origin to define relative source and receiver locations.
space_order = 6
nbl = 40
free_surface = False
dt = 2.
precond = False

true_vp = np.fromfile("./model_data/SMARMN/vp.true", dtype=np.float32).reshape(shape)/1000
smooth_vp = np.fromfile("./model_data/SMARMN/vp.smooth1", dtype=np.float32).reshape(shape)/1000
bathy_mask = np.ones(shape, dtype=np.float32)
bathy_mask[:, :6] = 0

true_model = Model(origin=origin, spacing=spacing, 
				shape=shape, space_order=space_order, vp=true_vp, 
				nbl=nbl, fs=free_surface, dt=dt)
init_model = Model(origin=origin, spacing=spacing, 
				shape=shape, space_order=space_order, vp=smooth_vp, 
				nbl=nbl, fs=free_surface, dt=dt)

# Set up acquisiton geometry
t0 = 0.
tn = 4500. 
f0 = 0.005
resample_dt = 5
# Set up source geometry, but define 5 sources instead of just one.
nsources = 31
src_coordinates = np.empty((nsources, 2))
src_coordinates[:, 0] = np.linspace(0, true_model.domain_size[0], num=nsources)
src_coordinates[:, -1] = 30.  # Source depth is 20m

# Initialize receivers for synthetic and imaging data
nreceivers = shape[0]
rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 0] = np.linspace(spacing[0], true_model.domain_size[0] - spacing[0], num=nreceivers)
rec_coordinates[:, 1] = 30.    # Receiver depth

# Set up geometry objects for observed and predicted data

geometry1 = AcquisitionGeometry(true_model, rec_coordinates, src_coordinates, t0, tn, 
				f0=f0, src_type='Ricker')
geometry0 = AcquisitionGeometry(init_model, rec_coordinates, src_coordinates, t0, tn, 
				f0=f0, src_type='Ricker')
geometry1.resample(resample_dt)
geometry0.resample(resample_dt)
#plot_velocity(true_model, source=geometry1.src_positions, receiver=geometry1.rec_positions[::4, :])

obs = fm_multi(geometry1, save=False)
#print(obs[2].data.shape)
#plot_shotrecord(obs[2].data, true_model, t0, tn)

filt_func = Filter(filter_type='highpass', freqmin=2, df=1000/resample_dt, axis=-2)

#filted_obs = filt_func(obs[1].data)
#plot_shotrecord(filted_obs, true_model, t0, tn)

qWmetric1d = qWasserstein(gamma=1.01, method='1d')
bfm_solver = bfm(num_steps=10, step_scale=1.)
qWmetric2d = qWasserstein(gamma=1.01, method='2d', bfm_solver=bfm_solver)

misfit_func = least_square
#misfit_func = qWmetric1d
#misfit_func = qWmetric2d

# Gradient check
# f, g = fwi_obj_multi(geometry0, obs, misfit_func, 
# 				filt_func)
# plot_image(g.reshape(shape), cmap='cividis')

model_err = []
def fwi_callback(xk):
	m = 1. / (true_vp.reshape(-1).astype(np.float64))**2
	model_err.append(np.linalg.norm((xk-m)/m))

# Box contraints
vmin = 1.4    # do not allow velocities slower than water
vmax = 5.0
bounds = [(1.0/vmax**2, 1.0/vmin**2) for _ in range(np.prod(shape))]    # in [s^2/km^2]

m0 = 1./(smooth_vp.reshape(-1).astype(np.float64))**2

# FWI with L-BFGS
ftol = 1e-20
maxiter = 10
maxls = 10
gtol = 1e-5
stepsize = 0.01
result = optimize.minimize(fwi_loss, m0, 
			args=(geometry0, obs, misfit_func, filt_func, bathy_mask, precond), 
			method='L-BFGS-B', jac=True, 
    		callback=fwi_callback, bounds=bounds, 
    		options={'ftol':ftol, 'maxiter':maxiter, 'disp':True,
    				'eps':stepsize,
    				'maxls':5, 'gtol':gtol, 'iprint':1,
    		})

# Plot FWI result
vp = 1.0/np.sqrt(result['x'].reshape(shape))
plot_image(vp, vmin=vmin, vmax=vmax, cmap="cividis")