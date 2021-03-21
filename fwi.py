from devito import Function
from seismic import Model, Receiver, AcquisitionGeometry
from seismic.acoustic import AcousticWaveSolver
import numpy as np
from scipy import optimize
from distributed import wait

def _loss(x, geometry, y, obj_func):

	# Convert x to velocity
	v = 1./np.sqrt(x.reshape(geometry.model.shape))
	# Overwrite current velocity in geometry (don't update boundary region)
	geometry.model.update('vp', v.reshape(geometry.model.shape))
	# Evaluate objective function
	fval, grad = obj_func()

def fm_single(geometry, save=False, dt=1.):
	"""Modeling function for acoustic function
	"""
	solver = AcousticWaveSolver(geometry.model, geometry, space_order=4)
	data, u = solver.forward(vp=geometry.model.vp, save=save)[0:2]
	return data.resample(dt), u

def fm_multi(geometry, save=False, dt=1.):
	"""modeling function for acoustic equation
	"""
	shots = []
	for i in range(geometry.nsrc):
		# Geometry for current shot
		geom_i = AcquisitionGeometry(geometry.model, geometry.rec_positions, 
					geometry.src_positions[i, :], geometry.t0, geometry.tn, 
					f0=geometry.f0, src_type=geometry.src_type)
		# Call modeling function
		shot = fm_single(geom_i, save, dt)[0]
		shots.append(shot)

	return shots

def fm_multi_parallel(client, geometry, save=False, dt=1.):
	"""Parallel modeling function for acoustic equation
	"""
	futures = []
	for i in range(geometry.nsrc):
		# Geometry for current shot
		geom_i = AcquisitionGeometry(geometry.model, geometry.rec_positions, 
					geometry.src_positions[i, :], geometry.t0, geometry.tn, 
					f0=geometry.f0, src_type=geometry.src_type)
		# Call modeling function
		futures.append(client.submit(fm_single, geom_i, save=save, dt=dt))

	# Wait for all workers to finish and collect shots
	wait(futures)
	shots = []
	for i in range(geometry.nsrc):
		shots.append(futures[i].result()[0])

	return shots

def fwi_obj_single(geometry, obs, misfit_func):
	grad = Function(name="grad", grid=geometry.model.grid)
	residual = Receiver(name="rec", grid=geometry.model.grid, 
				time_range=geometry.time_axis, 
				coordinates=geometry.rec_positions)
	solver = AcousticWaveSolver(geometry.model, geometry, space_order=4)
	dt = geometry.dt
	# predicted data and residual
	pred, wfd = solver.forward(vp=geometry.model.vp, save=True)[0:2]
	fval, residual_data = misfit_func(pred.data, obs.resample(dt).data[:][0:pred.data.shape[0], :])
	residual.data[:] = residual_data[:]

	solver.gradient(rec=residual, u=wfd, vp=geometry.model.vp, grad=grad)

	# Convert to numpy array and remove absorbing boundaries
	nbl = geometry.model.nbl
	crop_grad = np.array(grad.data[:])[nbl:-nbl, nbl:-nbl]

	return fval, crop_grad

def fwi_obj_multi(geometry, obs, misfit_func):
	fval = .0
	grad = np.zeros(geometry.model.shape)
	for i in range(geometry.nsrc):
		# Geometry for current shot
		geom_i = AcquisitionGeometry(geometry.model, geometry.rec_positions, 
					geometry.src_positions[i, :], geometry.t0, geometry.tn, 
					f0=geometry.f0, src_type=geometry.src_type)
		fval_, grad_ = fwi_obj_single(geom_i, obs[i], misfit_func)
		fval += fval_
		grad += grad_

	return fval, grad

def fwi_obj_multi_parallel(client, geometry, obs, misfit_func):
	futures = []
	for i in range(geometry.nsrc):
		# Geometry for current shot
		geom_i = AcquisitionGeometry(geometry.model, geometry.rec_positions, 
					geometry.src_positions[i, :], geometry.t0, geometry.tn, 
					f0=geometry.f0, src_type=geometry.src_type)
		futures.append(client.submit(fwi_obj_single, geom_i, obs[i], misfit_func))
	wait(futures)
	fval = .0
	grad = np.zeros(geometry.model.shape)
	for i in range(geometry.nsrc):
		fval += futures[i].result()[0]
		grad += futures[i].result()[1]	

	return fval, grad	

def fwi_loss(x, geometry, obs, misfit_func):
	# Convert x to velocity
	v = 1. / np.sqrt(x.reshape(geometry.model.shape))
	geometry.model.update('vp', v.reshape(geometry.model.shape))
	
	fval, grad = fwi_obj_multi(geometry, obs, misfit_func)

	return fval, grad.flatten().astype(np.float64)

class FWI(object):
	def __init__(self, true_model, misfit_func, maxIter=10, ftol=0.01):

		self.true_model = true_model
		self.geometry = geometry
		self.optim_methd = 'L-BFGS-B'
		self.ftol = ftol
		self.misfit = misfit_func
		self.maxIter = maxIter
		self.model_err = []

	def _fwi_callback(self, x):
		nbl = self.true_model.nbl
		v = self.true_model.vp.data[nbl:-nbl, nbl:-nbl]
		m = 1. / (v.reshape(-1).astype(np.float64))**2
		self.model_err.append(np.linalg.norm((x-m)/m))