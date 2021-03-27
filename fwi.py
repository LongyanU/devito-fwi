from devito import Function
from seismic import Model, Receiver, AcquisitionGeometry
from seismic.acoustic import AcousticWaveSolver
import numpy as np
from scipy import interpolate
from distributed import wait
from seismic.filter import bandpass, lowpass, highpass
from copy import deepcopy

def seismic_filter(data, filter_type: str, freqmin=None, freqmax=None, 
				df=None, corners=16, zerophase=False, axis=-1):
	assert filter_type.lower() in ['bandpass', 'lowpass', 'highpass']

	if filter_type == 'bandpass':
		if freqmin and freqmax and df:
			filt_data = bandpass(data, freqmin, freqmax, df, corners, zerophase, axis)
		else:
			raise ValueError
	if filter_type == 'lowpass':
		if freqmax and df:
			filt_data = lowpass(data, freqmax, df, corners, zerophase, axis)
		else:
			raise ValueError
	if filter_type == 'highpass':
		if freqmin and df:
			filt_data = highpass(data, freqmin, df, corners, zerophase, axis)
		else:
			raise ValueError
	return filt_data

class Filter(object):
	def __init__(self, filter_type: str, freqmin=None, freqmax=None, 
				df=None, corners=10, zerophase=False, axis=-1):
		self.filter_type = filter_type
		self.freqmin = freqmin
		self.freqmax = freqmax
		self.df = df
		self.corners = corners
		self.zerophase = zerophase
		self.axis = axis

	def __call__(self, data):
		return seismic_filter(data, self.filter_type, self.freqmin, 
			self.freqmax, self.df, self.corners, self.zerophase, self.axis)


def resample(x, t, t0, order=3):
	dt = t[1] - t[0]
	dt0 = t0[1] - t0[0]
	if np.isclose(dt, dt0):
		return x
	nsamples, ntraces = x.shape
	new_x = np.zeros((t.size, ntraces), dtype=np.float32)
	for i in range(ntraces):
		tck = interpolate.splrep(t0, x[:, i], k=order)
		new_x[:, i] = interpolate.splev(t, tck)
	return new_x

def fm_single(geometry, save=False):
	"""Modeling function for acoustic function
	"""
	solver = AcousticWaveSolver(geometry.model, geometry, 
				space_order=geometry.model.space_order)
	data, u = solver.forward(vp=geometry.model.vp, save=save)[0:2]
	return data, u

def fm_multi(geometry, save=False):
	"""modeling function for acoustic equation
	"""
	shots = []
	for i in range(geometry.nsrc):
		# Geometry for current shot
		geom_i = AcquisitionGeometry(geometry.model, geometry.rec_positions, 
					geometry.src_positions[i, :], geometry.t0, geometry.tn, 
					f0=geometry.f0, src_type=geometry.src_type, 
					filter=geometry._filter)
		# Call modeling function
		shot = fm_single(geom_i, save)[0]
		shots.append(shot)

	return shots

def fm_multi_parallel(client, geometry, save=False):
	"""Parallel modeling function for acoustic equation
	"""
	futures = []
	for i in range(geometry.nsrc):
		# Geometry for current shot
		geom_i = AcquisitionGeometry(geometry.model, geometry.rec_positions, 
					geometry.src_positions[i, :], geometry.t0, geometry.tn, 
					f0=geometry.f0, src_type=geometry.src_type, 
					filter=geometry._filter)
		# Call modeling function
		futures.append(client.submit(fm_single, geom_i, save))

	# Wait for all workers to finish and collect shots
	wait(futures)
	shots = []
	for i in range(geometry.nsrc):
		shots.append((futures[i].result()[0]))

	return shots

def fix_source_illumination(geometry, g):
	if geometry.src_positions.shape[0] > 1:
		raise ValueError("Only single source valid.")
	dx, dz = geometry.model.spacing
	src_pos = geometry.src_positions
	sx, sz = src_pos[0][0], src_pos[0][1]
	nx, nz = geometry.model.shape
	if g.shape != (nx, nz):
		raise ValueError("Shape does not match!")
	x = np.arange(0, nx) * dx
	z = np.arange(0, nz) * dz
	xx, zz = np.meshgrid(z, x)
	sigma = dx + dz

	# first source
	mask = np.exp( -.5*( (xx-sx)**2 + (zz-sz)**2 )/(sigma**2) )
	g = g * (1. - mask)
	# then receiver
	nr = geometry.rec_positions.shape[0]
	for i in range(nr):
		rec_pos = geometry.rec_positions[i, :]
		rx, rz = rec_pos[0], rec_pos[1]
		mask = np.exp( -.5*( (xx-rx)**2 + (zz-rz)**2 )/(sigma**2) )
		g = g * (1. - mask)

	return g

def fwi_obj_single(geometry, obs, misfit_func, 
			filter_func=None, resample_dt=None, calc_grad=False):

	solver = AcousticWaveSolver(geometry.model, geometry, 
					space_order=geometry.model.space_order)
	# predicted data and residual
	pred, wfd = solver.forward(vp=geometry.model.vp, save=calc_grad)[0:2]

	if resample_dt is None:
		resample_dt = geometry.dt
	if resample_dt is not None:
		obs = deepcopy(obs).resample(resample_dt) # Important: use deepcopy to avoid changing the orignal data
		pred = pred.resample(resample_dt)	
	if filter_func is not None:
		syn_data = filter_func(pred.data)
		obs_data = filter_func(obs.data) 
	else:
		syn_data = pred.data
		obs_data = obs.data
	fval, residual_data = misfit_func(syn_data, obs_data)

	residual = Receiver(name="rec", grid=geometry.model.grid, 
				time_range=geometry.time_axis, 
				coordinates=geometry.rec_positions)

	residual.data[:] = resample(residual_data, 
					geometry.time_axis.time_values,
					pred.time_values)[:]
	illum, calc_grad = None, None
	if calc_grad:
		grad = Function(name="grad", grid=geometry.model.grid)
		solver.gradient(rec=residual, u=wfd, vp=geometry.model.vp, grad=grad)

		# Convert to numpy array and remove absorbing boundaries
		nbl = geometry.model.nbl
		crop_grad = np.array(grad.data[:])[nbl:-nbl, nbl:-nbl]
		crop_grad = fix_source_illumination(geometry, crop_grad)

		illum = (wfd.data * wfd.data).sum(axis=0)[nbl:-nbl, nbl:-nbl]
		illum = fix_source_illumination(geometry, illum)

	return fval, crop_grad, residual, illum

def fwi_obj_multi(geometry, obs, misfit_func, 
			filter_func=None, mask=None, precond=True, 
			calc_grad=False):
	fval = .0
	grad = np.zeros(geometry.model.shape)
	illum = np.zeros(geometry.model.shape)
	nbl = geometry.model.nbl
	for i in range(geometry.nsrc):
		# Geometry for current shot
		geom_i = AcquisitionGeometry(geometry.model, geometry.rec_positions, 
					geometry.src_positions[i, :], geometry.t0, geometry.tn, 
					f0=geometry.f0, src_type=geometry.src_type, 
					filter=geometry._filter)
		fval_, grad_, _, illum_ = fwi_obj_single(geom_i, obs[i], misfit_func, 
							filter_func, geometry.dt, calc_grad)
		fval += fval_
		if calc_grad:
			grad += grad_
			illum += illum_
	if calc_grad:
		if precond:
			grad  /= np.sqrt(illum + 1e-30)
		if mask is not None:
			grad *= mask 
	return fval, grad

def fwi_obj_multi_parallel(client, geometry, obs, misfit_func, 
			filter_func=None, mask=None, precond=True, calc_grad=False):
	futures = []
	for i in range(geometry.nsrc):
		# Geometry for current shot
		geom_i = AcquisitionGeometry(geometry.model, geometry.rec_positions, 
					geometry.src_positions[i, :], geometry.t0, geometry.tn, 
					f0=geometry.f0, src_type=geometry.src_type, 
					filter=geometry._filter)
		futures.append(client.submit(fwi_obj_single, geom_i, obs[i], 
						misfit_func, geometry.dt, calc_grad))
	wait(futures)
	fval = .0
	grad = np.zeros(geometry.model.shape)
	illum = np.zeros(geometry.model.shape)
	nbl = geometry.model.nbl	
	for i in range(geometry.nsrc):
		fval += futures[i].result()[0]
		if calc_grad:
			grad += futures[i].result()[1]
			illum += futures[i].result()[3]
	if calc_grad:		
		if precond:
			grad  /= np.sqrt(illum + 1e-30)	
		if mask is not None:
			grad *= mask

	return fval, grad

def fwi_loss(x, geometry, obs, misfit_func, 
		filter_func=None, mask=None, precond=True,
		calc_grad=True):
	# Convert x to velocity
	v = 1. / np.sqrt(x.reshape(geometry.model.shape))
	geometry.model.update('vp', v.reshape(geometry.model.shape))
	
	fval, grad = fwi_obj_multi(geometry, obs, misfit_func, 
						filter_func, mask, precond, calc_grad)

	print("Loss: %f"%fval)

	return fval, grad.flatten().astype(np.float64)

