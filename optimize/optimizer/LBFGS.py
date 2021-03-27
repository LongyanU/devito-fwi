
import numpy as np
from ..math import angle
from .base import Base

class lbfgs(Base):
	""" Limited-memory BFGS algorithm

	Includes optional safeguards: periodic restarting and descent
	conditions.
	"""

	def __init__(self, memory=10, thresh=0., max_call=np.inf):
		self.memory = memory
		self.max_call = max_call
		self.thresh = thresh

		self.call_count = 0
		self.memory_used = 0

		self.g = None
		self.m = None

	def compute_direction(self, m, g):
		""" Returns L-BFGS search direction
		"""	
		if self.call_count == 0:
			self.g = g
			self.m = m
			return -g
		elif self.call_count > self.max_call:
			print('Restaring LBFGS... [periodic restart]')
			self.restart()
			return -g

		S, Y = self.update(m, g)
		q = self.apply(g, S, Y)

		self.g = g
		self.m = m

		status = self.check_status(g, q)
		if status != 0:
			self.restart()
			return -g
		else:
			return -q

	def update(self, m, g):
		""" Update L-BFGS algorithm history
		"""
		s = m - self.m
		y = g = self.g
		m = len(s)
		n = self.memory

		if self.memory_used == 0:
			S = np.memmap('LBFGS/S', mode='w+', dtype='float32', shape=(m, n))
			Y = np.memmap('LBFGS/Y', mode='w+', dtype='float32', shape=(m, n))
			S[:, 0] = s
			Y[:, 0] = y
			self.memory_used = 1
		else:
			S = np.memmap('LBFGS/S', mode='r+', dtype='float32', shape=(m, n))
			Y = np.memmap('LBFGS/Y', mode='r+', dtype='float32', shape=(m, n))
			S[:, 1:] = S[:, :-1]
			Y[:, 1:] = Y[:, :-1]
			S[:, 0] = s
			Y[:, 0] = y
			if self.memory_used < self.memory:
				self.memory_used += 1
		return S, Y

	def apply(self, q, S=[], Y=[]):
		""" Applies L-BFGS inverse Hessian to given vector
		"""
		if S==[] or Y==[]:
			m = len(q)
			n = self.memory
			S = np.memmap('LBFGS/S', mode='w+', dtype='float32', shape=(m, n))
			Y = np.memmap('LBFGS/Y', mode='w+', dtype='float32', shape=(m, n))
		# first matrix product
		kk = self.memory_used
		rh = np.zeros(kk)
		al = np.zeros(kk)
		for ii in range(kk):
			rh[ii] = 1/np.dot(Y[:,ii], S[:,ii])
			al[ii] = rh[ii]*np.dot(S[:,ii], q)
			q = q - al[ii]*Y[:,ii]

		r = q
		# use scaling M3 proposed by Liu and Nocedal 1989
		sty = np.dot(Y[:,0], S[:,0])
		yty = np.dot(Y[:,0], Y[:,0])
		r *= sty/yty
		# second matrix product
		for ii in range(kk-1, -1, -1):
			be = rh[ii]*np.dot(Y[:,ii], r)
			r = r + S[:,ii]*(al[ii] - be)
		return r

	def restart(self):
		""" Discards history and resets counters
		"""
		self.call_count = 0
		self.memory_used = 0

		S = np.memmap('LBFGS/S', mode='r+')
		Y = np.memmap('LBFGS/Y', mode='r+')
		S[:] = 0.
		Y[:] = 0.

	def check_status(self, g, r):
		theta = 180.*np.pi**-1*angle(g, r)
		if not 0. < theta < 90.:
			print('restarting LBFGS... [not a descent direction]')
			return 1
		elif theta > 90. - self.thresh:
			print('restarting LBFGS... [practical safeguard]')
			return 1
		else:
			return 0		