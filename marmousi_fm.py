from seismic import Model, AcquisitionGeometry, Receiver
from seismic import plot_shotrecord, plot_velocity, plot_image
import numpy as np

import matplotlib.pyplot as plt

from fwi import Filter, fm_multi


import argparse, os, shutil
from time import time
import copy

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
parser.add_argument('--resample', type=float, default=6., help='resample dt')
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
	# Setup velocity model
	shape = (300, 106)      # Number of grid points (nx, nz).
	spacing = (30., 30.)    # Grid spacing in m. The domain size is now 1km by 1km.
	origin = (0, 0)         # Need origin to define relative source and receiver locations.
	space_order = 8
	nbl = 40
	free_surface = False
	dt = 2.95

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
	tn = 4500. 
	f0 = 0.007
	# Set up source geometry, but define 5 sources instead of just one.
	nsources = 21
	src_coordinates = np.empty((nsources, 2))
	src_coordinates[:, 0] = np.linspace(0, true_model.domain_size[0], num=nsources)
	src_coordinates[:, -1] = 2*spacing[0]  # Source depth is 20m

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
	geometry1.resample(resample_dt)
	geometry0.resample(resample_dt)


	obs = fm_multi(geometry1, save=False)
	syn = fm_multi(geometry0, save=False)

	for i in range(nsources):
		obs[i].data[:].astype(np.float32).tofile(os.path.join(result_dir, 'data/obs'+str(i)))
		syn[i].data[:].astype(np.float32).tofile(os.path.join(result_dir, 'data/syn'+str(i)))
