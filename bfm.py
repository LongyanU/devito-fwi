from w2 import BFM
from time import time
import numpy as np
import numpy.ma as ma
from scipy.fftpack import dctn, idctn

def dct2(x):
	return dctn(x, norm='ortho')

def idct2(x):
	return idctn(x, norm='ortho')

class bfm(object):
	"""Solve quadratic cost optimal transport using the back-and-forth method
	"""
	def __init__(self, shape, num_steps=10, step_scale=8.):
		self.shape = shape
		self.num_steps = num_steps
		self.step_scale = step_scale

		self.kernel = self._init_poisson_solver_kernel();
		self.mu = np.zeros(self.shape)
		self.nu = np.zeros(self.shape)

		self.upper = .75
		self.lower = .25
		self.scale_down = .8
		self.scale_up = 1./self.scale_down

		self.x, self.y = np.meshgrid(np.linspace(.5/self.shape[0], 1-.5/self.shape[0], self.shape[0]), 
							np.linspace(.5/self.shape[1], 1-.5/self.shape[1], self.shape[1]))

	def _init_poisson_solver_kernel(self):
		xx, yy = np.meshgrid(np.linspace(0, np.pi, self.shape[0], False), 
							np.linspace(0, np.pi, self.shape[1], False))
		kernel = 2*self.shape[0]**2*(1 - np.cos(xx)) + 2*self.shape[1]**2*(1 - np.cos(yy))
		kernel[0, 0] = 1
		return kernel

	def setup(self, f, g):
		self.mu = f
		self.nu = g

	def set_param(self, upper, lower, scale_down):
		self.upper = upper
		self.lower = lower
		self.scale_down = scale_down
		self.scale_up = 1./ scale_down

	def _normalize(self):
		self.mu *= self.mu.size / np.sum(self.mu)
		self.nu *= self.nu.size / np.sum(self.nu)

	def _update_potential(self, phi, rho, f, sigma):
		rho -= f
		workspace = dct2(rho) / self.kernel
		workspace[0, 0] = 0
		workspace = idct2(workspace)

		phi += sigma * workspace
		h1 = np.sum(workspace * rho) / rho.size
		return h1

	def _calc_obj(self, phi, psi):
		return np.sum(.5*(self.x**2 + self.y**2)*(self.mu + self.nu) - 
				self.nu*phi - self.mu*psi) / phi.size

	def _update_stepsize(self, sigma, value, old_value, grad_sq):
		diff = value - old_value
		if diff > grad_sq * sigma * self.upper:
			return sigma * self.scale_up
		elif diff < grad_sq * sigma * self.lower:
			return sigma * self.scale_down
		return sigma

	def _check(self):
		if np.sum(self.mu)<= 0 or np.sum(self.nu)<=0:
			raise ValueError("Non-positive density funcion unsupported!") 

	def solve(self, verbose=False):
		tic = time()
		self._check()
		self._normalize()

		sigma = self.step_scale / np.maximum(self.mu.max(), self.nu.max())

		phi = .5 * (self.x**2 + self.y**2)
		psi = .5 * (self.x**2 + self.y**2)

		bf = BFM(self.shape[0], self.shape[1], self.mu)

		rho = np.copy(self.mu)
		old_value = self._calc_obj(phi, psi)

		for k in range(self.num_steps+1):
			grad_sq = self._update_potential(phi, rho, self.nu, sigma)
			bf.ctransform(psi, phi)
			bf.ctransform(phi, psi)

			value = self._calc_obj(phi, psi)
			sigma = self._update_stepsize(sigma, value, old_value, grad_sq)
			old_value = value

			bf.pushforward(rho, phi, self.nu)

			grad_sq = self._update_potential(psi, rho, self.mu, sigma)
			bf.ctransform(phi, psi)
			bf.ctransform(psi, phi)
			bf.pushforward(rho, psi, self.mu)

			value = self._calc_obj(phi, psi)
			sigma = self._update_stepsize(sigma, value, old_value, grad_sq)
			old_value = value

			if verbose:
				if k%5 == 0:
					print(f'\t#iter {k:4d}, \t W2 value: {value:.6e}, H1 err: {grad_sq:.2e}')
		toc = time()
		if verbose:
			print(f'\n Elapsed time: {toc-tic:.2f}s')
		psi = .5*(self.x **2 + self.y **2) - psi
		phi = .5*(self.x **2 + self.y **2) - phi

		return value, phi, psi

	def gradient(self, f, g):
		self.setup(f, g)
		value, _, psi = self.solve()
		grad = psi - (psi * self.mu).sum()/psi.size

		return value, grad