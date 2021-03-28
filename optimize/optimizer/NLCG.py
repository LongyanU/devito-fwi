
import os
import numpy as np
from ..math import dot
from .base import Base


class nlcg(Base):
	"""Nonliear conjugate gradient method
	"""
	def __init__(self, beta_type='FR', max_call=np.inf, thresh=0.):
		assert beta_type in ['FR', 'PR', 'HS', 'DY']
		self.beta_type = beta_type

		self.g_old = None
		self.g_new = None
		self.p_old = None
		self.p_new = None
		self.thresh = thresh
		self.call_count = 0
		self.max_call = max_call

	def compute_direction(self, m, g):
		self.g_old = self.g_new
		self.p_old = self.p_new
		self.g_new = g
		self.call_count += 1
		if self.call_count == 1:
			self.p_new = -g
			return -g, 0
		elif self.call_count > self.max_call:
			print('restarting NLCG... [periodic restart]')
			self.restart()
			return -g, 1

		if self.beta_type == 'FR':
			beta = fletcher_reeves(self.g_new, self.g_old)
		if self.beta_type == 'PR':
			beta = pollak_ribere(self.g_new, self.g_old)
		if self.beta_type == 'HS':
			beta = hestenes_stiefel(self.g_new, self.g_old, self.p_old)
		if self.beta_type == 'DY':
			beta = dai_yuan(self.g_new, self.g_old, self.p_old)

		self.p_new = -self.g_new + beta * self.p_old

		if check_conjugacy(self.g_new, self.g_old) > self.thresh:
			print('Restaring NLCG... [loss of conjugacy]')
			self.restart()
			return -g, 1
		elif check_descent(self.p_new, self.g_new) > 0.:
			print('Restaring NLCG... [not a descent direction]')
			self.restart()
			return -g, 1
		else:
			return self.p_new, 0

	def restart(self):
		"""Restart
		"""
		self.call_count = 0

def fletcher_reeves(g_new, g_old):
	num = dot(g_new, g_new)
	den = dot(g_old, g_old)
	beta = num/den if den!=0 else 0
	return beta

def pollak_ribere(g_new, g_old):
	num = dot(g_new, g_new-g_old)
	den = dot(g_old, g_old)
	beta = num/den if den!=0 else 0
	if beta < 0:
		beta = 0
	return beta

def hestenes_stiefel(g_new, g_old, p_old):
	num = -dot(g_new, g_new-g_old)
	den = dot(p_old, g_new-g_old)
	beta = num/den if den!=0 else 0
	return beta

def dai_yuan(g_new, g_old, p_old):
	num = -dot(g_new, g_new)
	den = dot(p_old, g_new-g_old)
	beta = num/den if den!=0 else 0
	return beta

def check_conjugacy(g_new, g_old):
	return abs(dot(g_new, g_old) / dot(g_new, g_new))

def check_descent(p_new, g_new):
	return dot(p_new, g_new) / dot(g_new, g_new)

