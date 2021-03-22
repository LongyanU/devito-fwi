from seismic import demo_model, AcquisitionGeometry, Receiver
from seismic import plot_shotrecord, plot_velocity, plot_image
import numpy as np
from scipy import optimize
from distributed import Client, wait, LocalCluster
from scipy import optimize
import matplotlib.pyplot as plt

from fwi import Filter, fm_multi, fwi_obj_multi, fwi_loss
from misfit import least_square, qWasserstein
from bfm import bfm
# Set up velocity model
shape = (101, 101)      # Number of grid points (nx, nz).
spacing = (10., 10.)    # Grid spacing in m. The domain size is now 1km by 1km.
origin = (0, 0)         # Need origin to define relative source and receiver locations.
nbl = 40
dt = 1.
precond = True
# True model
true_model = demo_model('circle-isotropic', vp_circle=3.6, vp_background=3,
    origin=origin, shape=shape, spacing=spacing, nbl=nbl, dt=dt)

# Initial model
init_model = demo_model('circle-isotropic', vp_circle=3, vp_background=3,
    origin=origin, shape=shape, spacing=spacing, nbl=nbl, dt=dt)

bathy_mask = np.ones(shape, dtype=np.float32)
#init_model._dt = true_model.critical_dt
# Set up acquisiton geometry
t0 = 0.
tn = 1000. 
f0 = 0.010
resample_dt = 5
# Set up source geometry, but define 5 sources instead of just one.
nsources = 11
src_coordinates = np.empty((nsources, 2))
src_coordinates[:, 1] = np.linspace(0, true_model.domain_size[0], num=nsources)
src_coordinates[:, 0] = 20.  # Source depth is 20m

# Initialize receivers for synthetic and imaging data
nreceivers = shape[0]
rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 1] = np.linspace(spacing[0], true_model.domain_size[0] - spacing[0], num=nreceivers)
rec_coordinates[:, 0] = 980.    # Receiver depth
# Set up geometry objects for observed and predicted data
geometry1 = AcquisitionGeometry(true_model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
geometry0 = AcquisitionGeometry(init_model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
geometry1.resample(resample_dt)
geometry0.resample(resample_dt)
# client = Client(processes=False)
obs = fm_multi(geometry1, save=False)

filt_func = Filter(filter_type='highpass', freqmin=2, 
				corners=10, df=1000/resample_dt, axis=-2)

# syn = fm_multi(geometry0, save=False)
# obs_data = obs[int(nsources/2)].data
# syn_data = syn[int(nsources/2)].data
# plot_shotrecord(obs_data, true_model, t0, tn)
# plot_shotrecord(filt_func(obs_data), true_model, t0, tn)
# plot_shotrecord(syn_data-obs_data, true_model, t0, tn)
# plot_shotrecord(filt_func(syn_data)-filt_func(obs_data), true_model, t0, tn)

qWmetric1d = qWasserstein(gamma=1.01, method='1d')
bfm_solver = bfm(num_steps=10, step_scale=1.)
qWmetric2d = qWasserstein(gamma=1.01, method='2d', bfm_solver=bfm_solver)

misfit_func = least_square
#misfit_func = qWmetric1d
#misfit_func = qWmetric2d

# # Gradient check
f, g = fwi_obj_multi(geometry0, obs, misfit_func, 
				filt_func)
plot_image(g.reshape(shape), cmap='cividis')

model_err = []
def fwi_callback(xk):
	nbl = true_model.nbl
	v = true_model.vp.data[nbl:-nbl, nbl:-nbl]
	m = 1. / (v.reshape(-1).astype(np.float64))**2
	model_err.append(np.linalg.norm((xk-m)/m))

# Box contraints
vmin = 2.5    # do not allow velocities slower than water
vmax = 4.0
bounds = [(1.0/vmax**2, 1.0/vmin**2) for _ in range(np.prod(init_model.shape))]    # in [s^2/km^2]

# Initial guess
v0 = init_model.vp.data[init_model.nbl:-init_model.nbl, init_model.nbl:-init_model.nbl]
m0 = 1.0 / (v0.reshape(-1).astype(np.float64))**2

# FWI with L-BFGS
ftol = 2e-9 # converge when ftol <= _factor * EPSMCH
maxiter = 50
maxls = 5
gtol = 1e-6
stepsize = 1e-8 # minimize default step size
"""
scipy.optimize.minimize(fun, x0, args=(), method='L-BFGS-B', jac=None, 
	bounds=None, tol=None, callback=None, 
	options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 
	'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 
	'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
"""
result = optimize.minimize(fwi_loss, m0, 
			args=(geometry0, obs, misfit_func, filt_func, bathy_mask, precond), 
			method='L-BFGS-B', jac=True, 
    		callback=fwi_callback, bounds=bounds, 
    		options={'ftol':ftol, 'maxiter':maxiter, 'disp':True,
    				'eps':stepsize,
    				'maxls':maxls, 'gtol':gtol, 'iprint':1,
    		})
# Plot FWI result
vp = 1.0/np.sqrt(result['x'].reshape(true_model.shape))
plot_image(vp, vmin=vmin, vmax=vmax, cmap="cividis")
