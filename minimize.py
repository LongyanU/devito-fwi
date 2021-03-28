
import os
import numpy as np
from fwi import fwi_loss

def divides(i, j):
	"""True if j divides i"""
	if j is 0:
		return False
	elif i % j:
		return False
	else:
		return True

class minimize(object):
	def __init__(self, optimizer,  maxIter=10, ftol=1e-2, gtol=1e-3, 
				log_path = './log',
				save_model_freq=10, 
				save_grad_freq=10):

		assert optimizer.name in ['LBFGS', 'NLCG', 'SteepestDescent']

		self.optimizer = optimizer
		self.ftol = ftol
		self.gtol = gtol
		self.maxIter = maxIter
		self.log_path = log_path
		self.save_model_freq = save_model_freq
		self.save_grad_freq = save_grad_freq

		self.optimizer.setup()
		self.check_path()

	def run(self, m, geometry, obs_data, misfit_func, filter_func, 
			precond=True, mask=None, bounds=None):
		iter_count = 0
		while iter_count < self.maxIter:
			print('Starting iteration', iter_count+1)
			# compute gradient
			print('\t Computing gradient')			
			fval, g = fwi_loss(m, geometry, obs_data, misfit_func, filter_func, mask, 
						precond)
			if iter_count == 0:
				self.f0 = fval 
			self.save_misfit(fval, g)
			# compute search direction
			print('\t Computing search direction')
			p = self.optimizer.compute_direction(m, g)
			# line search step size
			print('\t Computing step length')

			do_line_search = True
			while do_line_search:
				alpha = self.optimizer.initialize_search(m, g, p, fval)
				while True:
					print(" trial step", 
						self.optimizer.line_search.step_count+1)

					m_temp = self.apply_bounds(m + alpha*p, bounds)

					fval_try, _ = fwi_loss(m_temp, geometry, obs_data, 
								misfit_func, filter_func, mask, precond, calc_grad=False)
					print('\t fval_try: %10.3e' % fval_try)
					alpha, status = self.optimizer.update_search(alpha, fval_try)
				
					if status > 0:
						self.optimizer.finalize_search(g, p)
						do_line_search = False
						break
					elif status == 0:
						continue
					elif status < 0:
						if self.optimizer.retry_status(g, p):
							print(' Line search failed\n\n Retrying...')
							self.optimizer.restart()
							break
						else:
							print(' Line search failed\n\n Aborting...')
							do_line_search = False
							return m
			iter_count += 1
			# update the model
			m = self.apply_bounds(m + alpha*p, bounds)

			stop = self.finalize(m, g, fval, fval_try, iter_count)
			print('')
			if stop:
				return m
		return m

	def apply_bounds(self, x, bounds):
		if bounds is not None:
			if len(bounds) != 2:
				raise ValueError('The bounds should only have two values')
			x[x<bounds[0]] = bounds[0]
			x[x>bounds[1]] = bounds[1]
			return x
		return x

	def finalize(self, m, g, fk, fkp1, iter_count):
		self.write_count()
		if divides(iter_count, self.save_model_freq):
			self.save_model(m, iter_count)
		if divides(iter_count, self.save_grad_freq):
			self.save_gradient(g, iter_count)
		
		status = self.check_stopping_criteria(fk, fkp1, g)
		return status

	def check_stopping_criteria(self, fk, fkp1, g):
		"""Stop when |fk - fkp1|/max(|fk|, |fkp1|) < ftol
		or |g|_\\infty < gtol
		"""
		fr = abs(fk - fkp1)/max(abs(fk), abs(fkp1))
		gr = np.max(np.abs(g))

		# if fr < self.ftol or gr < self.gtol:
		# 	print("Stopping crieria met. Done!")
		# 	return 1
		# else:
		# 	return 0
		if fkp1/self.f0 < self.ftol:
			return 1
		else:
			return 0

	def save_model(self, m, k):
		v = 1. / np.sqrt(m)
		path = os.path.join(self.log_path, 'model_est')
		if not os.path.exists(path):
			os.makedirs(path)
		v.astype(np.float32).tofile(os.path.join(path, 'v_'+str(k)))

	def save_gradient(self, g, k):
		path = os.path.join(self.log_path, 'gradient')
		if not os.path.exists(path):
			os.makedirs(path)
		g.astype(np.float32).tofile(os.path.join(path, 'g_'+str(k)))

	def save_misfit(self, fval, g):
		file = os.path.join(self.log_path, 'misfit')
		norm_g = np.max(np.abs(g))
		with open(file, 'a') as f:
			fmt = '%10.3e  %10.3e\n'
			f.write(fmt % (fval, norm_g))			
		print('\t\t f: %10.3e \t |g|: %10.3e'%(fval, norm_g))	

	def check_path(self):
		if not os.path.exists(self.log_path):
			os.makedirs(self.log_path)
		file = os.path.join(self.log_path, 'misfit')
		if os.path.exists(file):
			os.remove(file)

	def write_count(self):
		count = 0
		# first order methods (3 + NLS) * nsrc
		if self.optimizer.name in ['SteepestDescent', 'NLCG']:
			count = 3 + self.optimizer.line_search.step_count
		# Quasi-Newton and second-order (3 + NLS)
		elif self.optimizer.name in ['LBFGS']:
			if self.optimizer.call_count == 1:
				count = 3 + self.optimizer.line_search.step_count
			else:
				count = 2 + self.optimizer.line_search.step_count

		self.optimizer.writer('sim_count', count)