from seismic import Model, AcquisitionGeometry, Receiver
from seismic import plot_shotrecord, plot_velocity, plot_image
import numpy as np
import matplotlib.pyplot as plt

from fwi import Filter, fm_multi

import argparse, os, shutil
from time import time

parser = argparse.ArgumentParser(description='Full waveform inversion')
parser.add_argument('--misfit', type=int, default=0, choices=[0, 1, 2], 
			help='misfit function type:0=least square/1=1d W2/2=2d W2')
parser.add_argument('--precond', type=int, default=1, help='apply precondition')
parser.add_argument('--odir', type=str, default='./result/SMARM2', 
			help='directory to output result')
parser.add_argument('--bathy', type=int, default=1, help='apply bathy mask')
parser.add_argument('--check-gradient', type=int, default=0, 
			help='check the gradient at 1st iteration')
parser.add_argument('--filter', type=int, default=0, help='filtering data')
parser.add_argument('--resample', type=float, default=0., help='resample dt, default 0 will not resample')
parser.add_argument('--ftol', type=float, default=1e-5, help='Optimizing loss tolerance')
parser.add_argument('--gtol', type=float, default=1e-10, help='Optimizing gradient norm tolerance')
parser.add_argument('--nsrc', type=int, default=31, help='number of shots')
parser.add_argument('--maxiter', type=int, default=200, help='FWI iteration')
parser.add_argument('--steplen', type=float, default=0.1, help='initial step length for line search')
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
		'\t Use mask: %d \t Filtering source: %d \t Resample rate: %.2f\n'%(use_bathy, use_filter, resample_dt),
		'\t ftol: %e \t gtol: %e \t nsrc: %d\n'%(ftol, gtol, nsources),
		'\t maxiter:%d \t maxls: %d \t init step length: %.3f\n'%(maxiter, args.maxls, args.steplen),	
		'-------------------------------------------------'
		)

	# Setup velocity model
	shape = (340, 140)      # Number of grid points (nx, nz).
	spacing = (30., 30.)    # Grid spacing in m. The domain size is now 1km by 1km.
	origin = (0, 0)         # Need origin to define relative source and receiver locations.
	space_order = 8
	nbl = 40
	free_surface = False
	dt = 3.

	true_vp = np.fromfile("./model_data/SMARM2/vp.true", dtype=np.float32).reshape(shape)/1000
	smooth_vp = np.fromfile("./model_data/SMARM2/vp.smooth_20", dtype=np.float32).reshape(shape)/1000

	# constant water model
	constant_vp = np.ones(shape) * 1.5

	bathy_mask = np.ones(shape, dtype=np.float32)
	bathy_mask[:, :5] = 0
	if not use_bathy:
		bathy_mask = None

	true_model = Model(origin=origin, spacing=spacing, 
					shape=shape, space_order=space_order, vp=true_vp, 
					nbl=nbl, fs=free_surface, dt=dt)
	init_model = Model(origin=origin, spacing=spacing, 
					shape=shape, space_order=space_order, vp=smooth_vp, 
					nbl=nbl, fs=free_surface, dt=dt)
	constant_model = Model(origin=origin, spacing=spacing, 
					shape=shape, space_order=space_order, vp=constant_vp, 
					nbl=nbl, fs=free_surface, dt=dt)
	# Set up acquisiton geometry
	t0 = 0.
	tn = 4500. 
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
	geometry2 = AcquisitionGeometry(constant_model, rec_coordinates, src_coordinates, t0, tn, 
					f0=f0, src_type='Ricker', filter=filt_func)	

	#plot_velocity(true_model, source=geometry1.src_positions, receiver=geometry1.rec_positions[::4, :])

	obs = fm_multi(geometry1, save=False)
	syn = fm_multi(geometry0, save=False)
	direct_wave = fm_multi(geometry2, save=False)
	print(obs[0].data.shape)
	for i in range(nsources):
		obs[i].data[:].astype(np.float32).tofile(os.path.join(result_dir, 'data/obs'+str(i)))
		syn[i].data[:].astype(np.float32).tofile(os.path.join(result_dir, 'data/syn'+str(i)))
		direct_wave[i].data[:].astype(np.float32).tofile(os.path.join(result_dir, 'data/dw'+str(i)))