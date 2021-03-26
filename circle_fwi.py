from seismic import demo_model, AcquisitionGeometry, Receiver
from seismic import plot_shotrecord, plot_velocity, plot_image
from seismic.wavelet import Ricker, Gabor, DGauss
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
parser.add_argument('--resample', type=float, default=5., help='resample dt')
parser.add_argument('--ftol', type=float, default=1e-2, help='Optimizing loss tolerance')
parser.add_argument('--gtol', type=float, default=1e-4, help='Optimizing gradient norm tolerance')

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
	resample_dt = args.resample
	ftol = args.ftol
	gtol = args.gtol	
	# Set up velocity model
	shape = (201, 201)      # Number of grid points (nx, nz).
	spacing = (10., 10.)    # Grid spacing in m. The domain size is now 1km by 1km.
	origin = (0, 0)         # Need origin to define relative source and receiver locations.
	space_order = 6
	nbl = 40
	dt = 1.
	radius = 60
	# True model
	true_model = demo_model('circle-isotropic', vp_circle=3.6, vp_background=3, r=radius,
	    origin=origin, shape=shape, spacing=spacing, space_order=space_order, nbl=nbl, dt=dt)

	# Initial model
	init_model = demo_model('circle-isotropic', vp_circle=3, vp_background=3, r=radius,
	    origin=origin, shape=shape, spacing=spacing, space_order=space_order, nbl=nbl, dt=dt)

	bathy_mask = np.ones(shape, dtype=np.float32)
	if not use_bathy:
		bathy_mask = None	

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
	rec_coordinates[:, 0] = 1980.    # Receiver depth

	# set up source 
	src_data = Ricker(t0, tn, dt, f0)
	if use_filter:
		filt_func = Filter(filter_type='highpass', freqmin=2, 
					corners=6, df=1000/dt)
		src_data = filt_func(src_data)
	# Set up geometry objects for observed and predicted data
	geometry1 = AcquisitionGeometry(true_model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_data=src_data)
	geometry0 = AcquisitionGeometry(init_model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_data=src_data)
	geometry1.resample(resample_dt)
	geometry0.resample(resample_dt)
	# client = Client(processes=False)
	obs = fm_multi(geometry1, save=False)

	plot_shotrecord(obs[int(nsources/2)].data, true_model, t0, tn, show=False)
	plt.savefig(os.path.join(result_dir, 'circle_data'+'.png'), 
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
						None, bathy_mask, precond)
		g.tofile(os.path.join(result_dir, 'circle_1st_grad_'+str(misfit_type)))		
		plot_image(g.reshape(shape), cmap='bwr', show=False)
		plt.savefig(os.path.join(result_dir, 
				'circle_1st_grad_'+str(misfit_type)+'.png'), bbox_inches='tight')
		plt.savefig(os.path.join(result_dir, 
				'circle_1st_grad_'+str(misfit_type)+'.eps'), bbox_inches='tight')
		plt.clf()
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
	# ftol = 2e-2 converge when |fk - fkp1|/max(|fk|, |fkp1|, 1) < ftol
	# gtol = 1e-4	
	# for Wasserstein loss, it is always very small (~1e-6) depending on problems

	maxiter = 50
	maxls = 5
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
				args=(geometry0, obs, misfit_func, None, bathy_mask, precond), 
				method='L-BFGS-B', jac=True, 
	    		callback=fwi_callback, bounds=bounds, 
	    		options={'ftol':ftol, 'maxiter':maxiter, 'disp':True,
	    				'maxcor': L, 'maxls':maxls, 'gtol':gtol, 'iprint':1,
	    		})
	toc = time()
	print(f'\n Elapsed time: {toc-tic:.2f}s')	
	# Plot FWI result
	vp = 1.0/np.sqrt(result['x'].reshape(true_model.shape))

	vp.tofile(os.path.join(result_dir, "circle_result_misfit_"+str(misfit_type)))

	file = open(os.path.join(result_dir, "circle_model_err_info_"+str(misfit_type)+'.txt'), "w")
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
		file = open(os.path.join(result_dir, "circle_optim_info_"+str(misfit_type)+'.txt'), "w")
		for item in useful_info:
			file.write("%s\n" % item)
		file.close()
		nohup_file = 'circle_nohup_'+str(misfit_type)+'.out'
		os.rename('./nohup.out', nohup_file)
		shutil.move(nohup_file, os.path.join(result_dir, nohup_file))
	except:
		pass
	plot_image(vp, vmin=vmin, vmax=vmax, cmap="jet", show=False)
	plt.savefig(os.path.join(result_dir, 
			'circle_inverted_'+str(misfit_type)+'.png'), bbox_inches='tight')
	plt.savefig(os.path.join(result_dir, 
			'circle_inverted_'+str(misfit_type)+'.eps'), bbox_inches='tight')
	plt.clf()
