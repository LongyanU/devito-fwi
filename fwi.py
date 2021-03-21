from devito import Function
from seismic import Model, Receiver, AcquisitionGeometry
from seismic.acoustic import AcousticWaveSolver
import numpy as np
from scipy import optimize

from distributed import Client, wait

def _loss(x, geometry, y, obj_func):

	# Convert x to velocity
	v = 1./np.sqrt(x.reshape(geometry.model.shape))
	# Overwrite current velocity in geometry (don't update boundary region)
	geometry.model.update('vp', v.reshape(geometry.model.shape))
	# Evaluate objective function
	fval, grad = obj_func()

def fm_single(geometry, save=False, stride=1.):
	"""Modeling function for acoustic function
	"""
	solver = AcousticWaveSolver(geometry.model, geometry, space_order=4)
	data, u = solver.forward(vp=geometry.model.vp, save=save)[0:2]
	return data.resample(stride), u

def fm_multi(client, geometry, save=False, stride=1.):
	"""Parallel modeling function for acoustic equation
	"""
	futures = []
	for i in range(geometry.nsrc):
		# Geometry for current shot
		geom_i = AcquisitionGeometry(geometry.model, geometry.rec_positions, 
					geometry.src_positions[i, :], geometry.t0, geometry.tn, 
					f0=geometry.f0, src_type=geometry.src_type)
		# Call modeling function
		futures.append(client.submit(fm_single, geom_i, save=save, stride=stride))

	# Wait for all workers to finish and collect shots
	wait(futures)
	shots = []
	for i in range(geometry.nsrc):
		shots.append(futures[i].result()[0])

	return shots

def fwi_obj_single(geometry, obs_data, misfit_func):
	grad = Function(name="grad", grid=geometry.model.grid)
	residual = Receiver(name="rec", grid=geometry.model.grid, 
				time_range=geometry.time_axis, 
				coordinates=geometry.rec_positions)
	solver = AcousticWaveSolver(geometry.model, geometry, space_order=4)
	# predicted data and residual
	pred_data, wfd = solver.forward(vp=geometry.model.vp, save=True)[0:2]
	fval, residual_data = misfit_func(pred_data.data, obs_data.data)
	residual.data[:] = residual_data[:]

	solver.gradient(rec=residual, u=wfd, vp=geometry.model.vp, grad=grad)

	# Convert to numpy array and remove absorbing boundaries


class FWI(object):
	def __init__(self, model, geometry, misfit_func):

		self.model = model
		self.geometry = geometry
		self.optim_methd = 'L-BFGS-B'
		self.ftol = 0.01
		self.misfit = misfit_func


