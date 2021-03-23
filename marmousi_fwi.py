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
parser.add_argument('--check-filter', type=int, default=0,
			help='check the filtered data')
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

	# Setup velocity model
	shape = (300, 106)      # Number of grid points (nx, nz).
	spacing = (30., 30.)    # Grid spacing in m. The domain size is now 1km by 1km.
	origin = (0, 0)         # Need origin to define relative source and receiver locations.
	space_order = 6
	nbl = 40
	free_surface = False
	dt = 3.

	true_vp = np.fromfile("./model_data/SMARMN/vp.true", dtype=np.float32).reshape(shape)/1000
	smooth_vp = np.fromfile("./model_data/SMARMN/vp.smooth1", dtype=np.float32).reshape(shape)/1000
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
	resample_dt = 10
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

	# Gradient check
	if check_gradient:
		f, g = fwi_obj_multi(geometry0, obs, misfit_func, 
						filt_func, bathy_mask, precond)
		g.tofile(os.path.join(result_dir, 'marmousi_1st_grad_'+str(misfit_type)))
		plot_image(g.reshape(shape), cmap='bwr', show=False)
		plt.savefig(os.path.join(result_dir, 
				'marmousi_1st_grad_'+str(misfit_type)+'.png'), bbox_inches='tight')
		plt.savefig(os.path.join(result_dir, 
				'marmousi_1st_grad_'+str(misfit_type)+'.eps'), bbox_inches='tight')

		plt.clf()
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
	ftol = 2e-10 # converge when ftol <= _factor * EPSMCH
	maxiter = 300
	maxls = 5
	gtol = 1e-9
	stepsize = 1e-8 # minimize default step size
	L = 10
	"""
	scipy.optimize.minimize(fun, x0, args=(), method='L-BFGS-B', jac=None, 
		bounds=None, tol=None, callback=None, 
		options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 
		'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 
		'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
	"""
	tic = time()
	result = optimize.minimize(fwi_loss, m0, 
				args=(geometry0, obs, misfit_func, filt_func, bathy_mask, precond), 
				method='L-BFGS-B', jac=True, 
	    		callback=fwi_callback, bounds=bounds, 
	    		options={'ftol':ftol, 'maxiter':maxiter, 'disp':True,
	    				'eps':stepsize, 'maxcor': L,
	    				'maxls':maxls, 'gtol':gtol, 'iprint':1,
	    		})
	toc = time()
	print(f'\n Elapsed time: {toc-tic:.2f}s')
	# Plot FWI result
	vp = 1.0/np.sqrt(result['x'].reshape(shape))

	vp.tofile(os.path.join(result_dir, "marmousi_result_misfit_"+str(misfit_type)))
	file = open(os.path.join(result_dir, "marmousi_model_err_info_"+str(misfit_type)+'.txt'), "w")
	for item in model_err:
		if item is not None:
			file.write("%s\n" % str(item))
	file.close()
	try:
		useful_info = []
		with open('./nohup.out', 'r') as file:
			for line in file:
				if line.find('Operator') < 0:
					useful_info.append(line)
		with open(os.path.join(result_dir, 'marmousi_optim_info_'+str(misfit_type)+'.txt')) as file:
			for item in useful_info:
				f.write("%s\n" % item)
		nohup_file = 'marmousi_nohup_'+str(misfit_type)+'.out'
		os.rename('./nohup.out', nohup_file)
		shutil.move(nohup_file, os.path.join(result_dir, nohup_file))
	except:
		pass
	plot_image(vp, vmin=vmin, vmax=vmax, cmap="jet", show=False)
	plt.savefig(os.path.join(result_dir, 
			'marmousi_inverted_'+str(misfit_type)+'.png'), bbox_inches='tight')
	plt.savefig(os.path.join(result_dir, 
			'marmousi_inverted_'+str(misfit_type)+'.eps'), bbox_inches='tight')
	plt.clf()