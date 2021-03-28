from seismic import Model, AcquisitionGeometry, Receiver
from seismic import plot_shotrecord, plot_velocity, plot_image
import numpy as np
from scipy import optimize
from distributed import Client, wait, LocalCluster
from scipy import optimize
import matplotlib.pyplot as plt

from fwi import Filter, fm_multi, fwi_obj_multi, fwi_loss
from misfit import least_square, qWasserstein
from optimize import LBFGS, NLCG, SteepestDescent
from minimize import minimize

import argparse, os, shutil
from time import time

parser = argparse.ArgumentParser(description='Full waveform inversion')
parser.add_argument('--misfit', type=int, default=0, choices=[0, 1, 2], 
			help='misfit function type:0=least square/1=1d W2/2=2d W2')
parser.add_argument('--precond', type=int, default=1, help='apply precondition')
parser.add_argument('--odir', type=str, default='./result/SMARMN', 
			help='directory to output result')
parser.add_argument('--bathy', type=int, default=1, help='apply bathy mask')
parser.add_argument('--check-gradient', type=int, default=0, 
			help='check the gradient at 1st iteration')
parser.add_argument('--filter', type=int, default=0, help='filtering data')
parser.add_argument('--resample', type=float, default=0., help='resample dt, default 0 will not resample')
parser.add_argument('--ftol', type=float, default=1e-2, help='Optimizing loss tolerance')
parser.add_argument('--gtol', type=float, default=1e-4, help='Optimizing gradient norm tolerance')
parser.add_argument('--nsrc', type=int, default=21, help='number of shots')
parser.add_argument('--maxiter', type=int, default=500, help='FWI iteration')
parser.add_argument('--steplen', type=float, default=0.05, help='initial step length for line search')
parser.add_argument('--maxls', type=int, default=5, help='max number of line search in each iteration')


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
	nsources = args.nsrc
	maxiter = args.maxiter
	print('---------------- Parameter Setting ------------\n',
		'\t Result dir: %s \t Misfit function: %d \t Precondition: %d\n'%(result_dir, misfit_type, precond), 
		'\t Use mask: %d \t Filtering source: %d \t Resample rate: %f\n'%(use_bathy, use_filter, resample_dt),
		'\t ftol: %e \t gtol: %e \t nsrc: %d\n'%(ftol, gtol, nsources),
		'\t maxiter:%d \t maxls: %d \t init step length: %.3f\n'%(maxiter, args.maxls, args.steplen),
		'-------------------------------------------------'
		)

	# Setup velocity model
	shape = (300, 106)      # Number of grid points (nx, nz).
	spacing = (25., 25.)    # Grid spacing in m. The domain size is now 1km by 1km.
	origin = (0, 0)         # Need origin to define relative source and receiver locations.
	space_order = 8
	nbl = 40
	free_surface = False
	dt = 2.

	true_vp = np.fromfile("./model_data/SMARMN/vp.true", dtype=np.float32).reshape(shape)/1000
	smooth_vp = np.fromfile("./model_data/SMARMN/vp.smooth_20", dtype=np.float32).reshape(shape)/1000
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
	tn = 4000. 
	f0 = 0.007
	# Set up source geometry, but define 5 sources instead of just one.
	src_coordinates = np.empty((nsources, 2))
	src_coordinates[:, 0] = np.linspace(0, true_model.domain_size[0], num=nsources)
	src_coordinates[:, -1] = 2*spacing[0]  # Source depth

	# Initialize receivers for synthetic and imaging data
	nreceivers = shape[0]
	rec_coordinates = np.empty((nreceivers, 2))
	rec_coordinates[:, 0] = np.linspace(spacing[0], true_model.domain_size[0] - spacing[0], num=nreceivers)
	rec_coordinates[:, 1] = 2*spacing[0]    # Receiver depth

	filt_func = None
	if use_filter:
		filt_func = Filter(filter_type='highpass', freqmin=3, 
					corners=6, df=1000/dt)		
	# Set up geometry objects for observed and predicted data
	geometry1 = AcquisitionGeometry(true_model, rec_coordinates, src_coordinates, t0, tn, 
					f0=f0, src_type='Ricker', filter=filt_func)
	geometry0 = AcquisitionGeometry(init_model, rec_coordinates, src_coordinates, t0, tn, 
					f0=f0, src_type='Ricker', filter=filt_func)
	if resample_dt == 0:
		resample_dt = dt
	geometry1.resample(resample_dt)
	geometry0.resample(resample_dt)
	#plot_velocity(true_model, source=geometry1.src_positions, receiver=geometry1.rec_positions[::4, :])

	obs = fm_multi(geometry1, save=False)

	plot_shotrecord(obs[int(nsources/2)].data, true_model, t0, tn, show=False)
	plt.savefig(os.path.join(result_dir, 
				'marmousi_data'+('_filtered' if use_filter else '')+'.png'), 
				bbox_inches='tight')
	plt.clf()

	qWmetric1d = qWasserstein(gamma=1.01, method='1d')
	qWmetric2d = qWasserstein(gamma=1.01, method='2d', 
							num_steps=10, step_scale=1.)

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
		g.tofile(os.path.join(result_dir, 
				'marmousi_1st_grad_'+str(misfit_type)+('_filtered' if use_filter else '')))
		plot_image(g.reshape(shape), cmap='bwr', show=False)
		plt.savefig(os.path.join(result_dir, 
				'marmousi_1st_grad_'+str(misfit_type)+('_filtered' if use_filter else '')+'.png'), 
				bbox_inches='tight')
		plt.clf()

	model_err = []
	def fwi_callback(xk):
		m = 1. / (true_vp.reshape(-1).astype(np.float64))**2
		k = len(model_err)
		if k%10==0:
			v = np.sqrt(1./xk)
			v.tofile(os.path.join(result_dir, 
				'marmousi_iter'+str(k)+'_'+str(misfit_type)+('_filtered' if use_filter else '')))
			plot_image(v.reshape(shape), cmap='jet', show=False)
			plt.savefig(os.path.join(result_dir, 
					'marmousi_iter'+str(k)+'_'+str(misfit_type)+('_filtered' if use_filter else '')+'.png'), 
					bbox_inches='tight')
			plt.clf()
		model_err.append(np.linalg.norm((xk-m)/m))	
	# Box contraints
	vmin = 1.5    # do not allow velocities slower than water
	vmax = 5.2
	bounds = [1.0/vmax**2, 1.0/vmin**2]    # in [s^2/km^2]

	m0 = 1./(smooth_vp.reshape(-1).astype(np.float64))**2

	# FWI with L-BFGS
	# ftol = 2e-2 converge when |fk - fkp1|/max(|fk|, |fkp1|, 1) < ftol
	# gtol = 1e-4	
	# for Wasserstein loss, it is always very small (~1e-6) depending on problems

	"""
	scipy.optimize.minimize(fun, x0, args=(), method='L-BFGS-B', jac=None, 
		bounds=None, tol=None, callback=None, 
		options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 
		'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 
		'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
	"""
	tic = time()
	optimizer = LBFGS(memory=10, ls_method='Bracket', step_len_init=args.steplen, max_ls=args.maxls, 
					log_path=os.path.join(result_dir, 'log'))
	minimizer = minimize(optimizer, maxIter=maxiter, ftol=ftol, gtol=gtol, 
					log_path=os.path.join(result_dir, 'log'))

	m = minimizer.run(m0, geometry0, obs, misfit_func, None, precond, bathy_mask, bounds)

	toc = time()
	print(f'\n Elapsed time: {toc-tic:.2f}s')
	# Plot FWI result
	vp = 1.0/np.sqrt(m.reshape(shape))

	vp.tofile(os.path.join(result_dir, 
				"marmousi_result_misfit_"+str(misfit_type)+('_filtered' if use_filter else '')))
	file = open(os.path.join(result_dir, 
				"marmousi_model_err_info_"+str(misfit_type)+('_filtered' if use_filter else '')+'.txt'), "w")
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
		file = open(os.path.join(result_dir, 
			"marmousi_optim_info_"+str(misfit_type)+('_filtered' if use_filter else '')+'.txt'), "w")
		for item in useful_info:
			file.write("%s\n" % item)
		file.close()
		nohup_file = 'marmousi_nohup_'+str(misfit_type)+('_filtered' if use_filter else '')+'.out'
		os.rename('./nohup.out', nohup_file)
		shutil.move(nohup_file, os.path.join(result_dir, nohup_file))
	except:
		pass
	plot_image(vp, vmin=vmin, vmax=vmax, cmap="jet", show=False)
	plt.savefig(os.path.join(result_dir, 
			'marmousi_inverted_'+str(misfit_type)+('_filtered' if use_filter else '')+'.png'), 
			bbox_inches='tight')
	plt.clf()