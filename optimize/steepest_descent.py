
from .optimizer import steepest_descent
from .base import base

class SteepestDescent(base):
	"""Steepest descent algorithm
	"""
	def __init__(self, ls_method='Bracket', max_ls=5, 
				step_len_init=0.05, step_len_max=0.5, 
				log_path='.', verbose=1):
		super().__init__(line_search_method=ls_method, max_ls=max_ls, 
					step_len_init=step_len_init, step_len_max=step_len_max, 
					log_path=log_path, verbose=verbose)
	@property
	def name(self):
		return 'steepest descent'

	@property
	def call_count(self):
		return self.sd.call_count

	def setup(self):
		super(SteepestDescent, self).setup()

		self.sd = steepest_descent()

	def compute_direction(self, m, g):
		p, self.restarted = self.sd.compute_direction(m, g)
		return p

	def restart(self):
		# steepest descent never requires restarts
		pass