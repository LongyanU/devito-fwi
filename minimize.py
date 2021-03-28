import numpy as np
from fwi import fwi_loss
import optimize

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

		assert optimizer.name in ['LBFGS', 'NLCG', 'steepest descent']

		self.optimizer = optimizer
		self.line_searcher = line_searcher
		self.step_len = step_len
		self.ftol = ftol
		self.gtol = gtol
		self.maxIter = maxIter

		self.const_step_len = False
		if self.line_searcher is None:
			self.const_step_len = True

	def run(self, m, geometry, obs_data, misfit_func, 
			precond=True, mask=None, bounds=None):
		iter_count = 0
		while iter_count <= self.maxIter:
			print('Starting iteration', iter_count)
			# compute gradient
			print('\t Computing gradient')			
			fval, g = fwi_loss(m, geometry, obs_data, misfit_func,
						precond, mask)
			if iter_count == 0:
				self.f0 = fval 
			self.save_misfit(fval)
			# compute search direction
			print('\t Computing search direction')
			p, _ = self.optimizer.compute_direction(m, g)
			# line search step size
			print('\t Computing step length')

			do_line_search = True
			while do_line_search:
				alpha = self.optimizer.initialize_search(m, g, p, fval)
				while True:
					print(" trial step", 
						self.optimizer.line_search.step_count+1)
					m_temp = m + alpha * p

					fval_try, _ = fwi_loss(m_temp, geometry, obs_data, 
								misfit_func, precond, mask, calc_grad=False)
					alpha, status = self.optimizer.update_search(fval_try)
				
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
							sys.exit(-1)
			iter_count += 1
			# update the model
			m = m + alpha * p
			self.finalize(m, g, fval, fval_try, iter_count)
			print('')

		return m

	def finalize(self, m, g, fk, fkp1, iter_count):
		if divides(iter_count, self.save_model_freq):
			self.save_model(m, iter_count)
		if divides(iter_count, self.save_grad_freq):
			self.save_gradient(g, iter_count)
		if iter_count > 1:
			self.check_stopping_criteria(fk, fkp1, g)

	def check_stopping_crieria(self, fk, fkp1, g):
		"""Stop when |fk - fkp1|/max(|fk|, |fkp1|) < ftol
		or |g|_\\infty < gtol
		"""
		fr = abs(fk - fkp1)/max(abs(fk), abs(fkp1))
		gr = np.max(np.abs(g))

		if fr < self.ftol or gr < self.gtol:
			print("Stopping crieria met. Done!")
			sys.exit(0)

	def save_model(self, m, k):
		v = v = 1. / np.sqrt(m)
		path = os.path.join(self.log_path, 'model_est')
		if not os.path.exists(path):
			os.makedirs(path)
		v.astype(np.float32).tofile(os.path.join(path, 'v_'+str(k)))

	def save_gradient(self, g, k):
		path = os.path.join(self.log_path, 'gradient')
		if not os.path.exists(path):
			os.makedirs(path)
		g.astype(np.float32).tofile(os.path.join(path, 'g_'+str(k)))

	def save_misfit(self, fval):
		file = os.path.join(self.log_path, 'misfit')
		with open(file, 'ab') as f:
			np.savetxt(f, [fval], '%11.6e')