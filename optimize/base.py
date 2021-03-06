
import numpy as np
import os
from . import line_search 
from .math import angle

class base(object):
	""" Nonlinear optimization abstract base class

	Base class on top of which steepest descent, nonlinear conjugate, quasi-
	Newton and Newton methods can be implemented.  Includes methods for
	both search direction and line search.

	Variables
		m_new - current model
		m_old - previous model
		m_try - line search model
		f_new - current objective function value
		f_old - previous objective function value
		f_try - line search function value
		g_new - current gradient direction
		g_old - previous gradient direction
		p_new - current search direction
		p_old - previous search direction	
	"""	
	def __init__(self, line_search_method='Bracket', max_ls=10, 
					step_len_init=None, step_len_max=None, 
					log_path='.', verbose=1):
		assert line_search_method in ['Backtrack', 'Bracket']
		self.line_search_method = line_search_method
		self.max_ls = max_ls
		self.log_path = log_path
		self.step_len_init = step_len_init
		self.step_len_max = step_len_max
		self.verbose = verbose
		self.restarted = 0
		
	@property
	def name(self):
		raise NotImplementedError("")

	@property
	def call_count(self):
		raise NotImplementedError("")

	def setup(self):
		""" Set up nonlinear optimization machinery
		"""
		self.writer = Writer(self.log_path)		
		# prepare line search machinery
		self.line_search = getattr(line_search, self.line_search_method)(
				step_count_max=self.max_ls, path=self.log_path
			)
		self.check_path()

	def compute_direction(self, m, g):

		return -g

	def initialize_search(self, m, g, p, fval):
		norm_m = np.abs(m).max()
		norm_p = np.abs(p).max()
		gtg = dot(g, g)
		gtp = dot(g, p)

		if self.restarted:
			self.line_search.clear_history()
		# optional step length safeguard
		if self.step_len_max:
			self.line_search.step_len_max = \
					self.step_len_max*norm_m/norm_p
		# determine initial step length
		alpha, _ = self.line_search.initialize(0., fval, gtg, gtp)

		# optional initial step length override
		if self.step_len_init and len(self.line_search.step_lens)<=1:
			alpha = self.step_len_init * norm_m/norm_p

		return alpha

	def update_search(self, alpha, fval):
		""" Updates line search status and step length

			Status codes
				status > 0  : finished
				status == 0 : not finished
				status < 0  : failed
		"""		
		alpha, status = self.line_search.update(alpha, fval)

		return alpha, status

	def finalize_search(self, g, p):
		x = self.line_search.search_history()[0]
		f = self.line_search.search_history()[1]

		# output latest statistics
		self.writer('factor', -dot(g, g)**-0.5 * (f[1]-f[0])/(x[1]-x[0]))
		self.writer('gradient_norm_L1', np.linalg.norm(g, 1))
		self.writer('gradient_norm_L2', np.linalg.norm(g, 2))
		self.writer('fval', f[0])
		self.writer('restarted', self.restarted)
		self.writer('slope', (f[1]-f[0])/(x[1]-x[0]))
		self.writer('step_count', self.line_search.step_count)
		self.writer('step_length', x[f.argmin()])
		self.writer('theta', 180.*np.pi**-1*angle(p, -g))

		self.line_search.writer.newline()


	def check_path(self):
		factor_file = os.path.join(self.log_path, 'factor')
		if os.path.exists(factor_file):
			os.remove(factor_file)
		gradient_norm_file1 = os.path.join(self.log_path, 'gradient_norm_L1')
		if os.path.exists(gradient_norm_file1):
			os.remove(gradient_norm_file1)
		gradient_norm_file2 = os.path.join(self.log_path, 'gradient_norm_L2')
		if os.path.exists(gradient_norm_file2):
			os.remove(gradient_norm_file2)
		fval_file = os.path.join(self.log_path, 'fval')
		if os.path.exists(fval_file):
			os.remove(fval_file)
		restart_file = os.path.join(self.log_path, 'restarted')
		if os.path.exists(restart_file):
			os.remove(restart_file)
		slope_file = os.path.join(self.log_path, 'slope')
		if os.path.exists(slope_file):
			os.remove(slope_file)		
		step_count_file = os.path.join(self.log_path, 'step_count')
		if os.path.exists(step_count_file):
			os.remove(step_count_file)
		sim_count_file = os.path.join(self.log_path, 'sim_count')
		if os.path.exists(sim_count_file):
			os.remove(sim_count_file)				
		step_length_file = os.path.join(self.log_path, 'step_length')
		if os.path.exists(step_length_file):
			os.remove(step_length_file)	
		theta_file = os.path.join(self.log_path, 'theta')
		if os.path.exists(theta_file):
			os.remove(theta_file)

	def retry_status(self, g, p):
		""" Determines if restart is worthwhile
		After failed line search, determines if restart is worthwhile by 
		checking, in effect, if search direction was the same as gradient
		direction
		"""
		theta = angle(p, -g)
		if self.verbose >= 2:
			print('\t theta: %.3f' % theta)
		thresh = 1e-3
		if abs(theta) < thresh:
			return 0
		else:
			return 1


	def restart(self):
		""" Restarts nonlinear optimization algorithm
		Keeps current position in model space, but discards history of
		nonlinear optimization algorithm in an attempt to recover from
		numerical stagnation 
		"""		
		self.line_search.clear_history()
		self.restarted = 1
		self.line_search.writer.iter -= 1
		self.line_search.writer.newline()


def dot(x, y):
	""" Computes inner product between vectors
	"""
	return np.dot(np.squeeze(x.flatten()), 
				np.squeeze(y.flatten()))

class Writer(object):
	""" Utility for appending values to text files
	"""
	def __init__(self, path='.'):
		self.path = path
		if not os.path.exists(path):
			os.makedirs(path)

		self.__call__('step_count', 0)

	def __call__(self, filename, val):
		fullfile = os.path.join(self.path, filename)
		with open(fullfile, 'a') as f:
			f.write('%e\n' % val)
