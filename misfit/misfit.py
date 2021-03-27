from .bfm import bfmx as bfm_solver
import numpy as np


def least_square(x, y):
	residal = x - y
	fval = .5 * np.linalg.norm(residal.flatten())**2

	return fval, residal

class qWasserstein(object):
	def __init__(self, trans_type='linear', gamma=1.0, method='1d', 
				num_steps=10, step_scale=1.):
		self.gamma = gamma
		assert method in ['1d', '2d']
		self.method = method
		self.bfm = bfm_solver(num_steps=num_steps, step_scale=step_scale)
		self.trans_type = trans_type
		
	def _transform(self, f, g):
		c = 0
		min_value = min(f.min(), g.min())
		if self.trans_type == 'linear':
			mu, nu = f, g
			c = -min_value if min_value<0 else 0
			c = c * self.gamma
			d = np.ones(f.shape)
		elif self.trans_type == 'square':
			mu = f * f
			nu = g * g
			d = 2 * f
		elif self.trans_type == 'exp':
			mu = np.exp(self.gamma*f)
			nu = np.exp(self.gamma*g)
			d = self.gamma * mu
		elif self.trans_type == 'softplus':
			mu = np.log(np.exp(self.gamma*f) + 1)
			nu = np.log(np.exp(self.gamma*g) + 1)
			d = gamma / np.exp(-self.gamma*f)
		else:
			mu, nu = f, g
			d = np.ones(f.shape)
		mu = mu + c
		nu = nu + c
		return mu, nu, d

	def _1d_calculator(self, f, g):
		"""Only works for 1d signal
		"""
		shape = f.shape
		f = np.squeeze(f)
		g = np.squeeze(g)
		mass = f.sum()
		mu = f / f.sum()
		nu = g / g.sum()

		t = np.linspace(0, 1, mu.size)
		# Cumulative
		F = np.cumsum(mu)
		G = np.cumsum(nu)
		T = np.interp(F, G, t)
		# W2
		loss = .5 * ((t - T)**2 * mu).sum()
		# gradient of W2 w.r.t. f
		grad = np.cumsum(t - T) - (t - T).sum()
		grad = (grad - (grad * mu).sum()) / mass
		return loss, grad.reshape(shape)

	def _2d_calculator(self, f, g):
		"""The bfm has built-in normalizing method, so we dont have to
		call normalize again 
		"""
		mass = f.sum() / f.size
		if self.bfm is None:
			raise ValueError("self.bfm has not been set up. Call method \"_set_bfm\" \
				to set it up.")
		loss, grad = self.bfm.gradient(f, g)

		return loss, grad/mass

	def __call__(self, f, g):
		shape = f.shape
		if(len(shape)==1):
			ntr = 1
		else:
			ntr = shape[1]
		if self.method == '2d' and ntr <= 1:
			raise ValueError("Can not use 2d method for 1D input.")

		# First transform signal to being positive
		mu, nu, d = self._transform(f, g)
		loss = 0
		grad = np.zeros(shape)
		if self.method == '1d':
			if ntr > 1:
				for i in range(ntr):
					value, grad[:, i] = self._1d_calculator(mu[:, i], nu[:, i])
					loss += value
			else:
				loss, grad = self._1d_calculator(mu, nu)
		elif self.method == '2d':
			loss, grad = self._2d_calculator(mu, nu)

		return loss, grad * d

class Misfit(object):
	def __init__(self, operator):
		self.operator = operator

	def __call__(self, x, y):
		return self.operator(x, y)

