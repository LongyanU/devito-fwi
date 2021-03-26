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

import argparse, os, shutil
from time import time
from sympy import Min, Max
from devito import Eq, Operator

parser = argparse.ArgumentParser(description='Full waveform inversion')
parser.add_argument('--misfit', type=int, default=0, choices=[0, 1, 2], 
			help='misfit function type:0=least square/1=1d W2/2=2d W2')
parser.add_argument('--precond', type=int, default=1, help='apply precondition')
parser.add_argument('--odir', type=str, default='./result', 
			help='directory to output result')
parser.add_argument('--bathy', type=int, default=1, help='apply bathy mask')
parser.add_argument('--check-gradient', type=int, default=1, 
			help='check the gradient at 1st iteration')
parser.add_argument('--filter', type=int, default=0, help='filtering data')
parser.add_argument('--check-filter', type=int, default=1,
			help='check the filtered data')
parser.add_argument('--resample', type=float, default=5., help='resample dt')
if __name__=='__main__':
	# Parse argument
	args = parser.parse_args()

	result_dir = args.odir
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	misfit_type = args.misfit
	precond = args.precond
	use_bathy = args.bathy
	check_gradient = args.check_gradient
	use_filter = args.filter
	check_filter = args.check_filter
	resample_dt = args.resample
	# Setup velocity model
	shape = (300, 106)      # Number of grid points (nx, nz).
	spacing = (30., 30.)    # Grid spacing in m. The domain size is now 1km by 1km.
	origin = (0, 0)         # Need origin to define relative source and receiver locations.
	space_order = 6
	nbl = 40
	free_surface = False
	dt = 2.

	true_vp = np.fromfile("./model_data/SMARMN/vp.true", dtype=np.float32).reshape(shape)/1000
	smooth_vp = np.fromfile("./model_data/SMARMN/vp.smooth_5", dtype=np.float32).reshape(shape)/1000
	bathy_mask = np.ones(shape, dtype=np.float32)
	bathy_mask[:, :7] = 0
	if not use_bathy:
		bathy_mask = None

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
	# Set up source geometry, but define 5 sources instead of just one.
	nsources = 21
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

	plot_shotrecord(obs[int(nsources/2)].data, true_model, t0, tn, show=False)
	plt.savefig(os.path.join(result_dir, 'marmousi_data'+'.png'), 
				bbox_inches='tight')
	plt.clf()

	filt_func = None
	if use_filter:
		filt_func = Filter(filter_type='highpass', freqmin=2, 
					corners=6, df=1000/resample_dt, axis=-2)

		if check_filter:
			filted_obs = filt_func(obs[int(nsources/2)].data)
			plot_shotrecord(filted_obs, true_model, t0, tn, show=False)
			plt.savefig(os.path.join(result_dir, 
				'marmousi_filtered_data'+'.png'), 
				bbox_inches='tight')			
			plt.clf()
	qWmetric1d = qWasserstein(gamma=1.01, method='1d')
	bfm_solver = bfm(num_steps=10, step_scale=1.)
	qWmetric2d = qWasserstein(gamma=1.01, method='2d', bfm_solver=bfm_solver)

	if misfit_type == 0:
		misfit_func = least_square
	if misfit_type == 1:
		misfit_func = qWmetric1d
	if misfit_type == 2:
		misfit_func = qWmetric2d

	model_err = []
	def fwi_callback(xk):
		m = 1. / (true_vp.reshape(-1).astype(np.float64))**2
		model_err.append(np.linalg.norm((xk-m)/m))

	# Box contraints
	vmin = 1.5    # do not allow velocities slower than water
	vmax = 5.5
	bounds = [(1.0/vmax**2, 1.0/vmin**2) for _ in range(np.prod(shape))]    # in [s^2/km^2]

	maxiter = 10
	history = np.zeros((maxiter, 1))
	vp = smooth_vp
	for i in range(0, maxiter):
		loss, grad = fwi_obj_multi(geometry0, obs, misfit_func, 
					filt_func, bathy_mask, precond)

		history[i] = loss
		alpha = .05 / np.max(grad)
		vp += alpha * grad
		geometry0.model.update('vp', vp)

		print('Iter: %d, Objective value %f' %(i, loss))