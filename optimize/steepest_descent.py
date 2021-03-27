
from optimizer import SteepestDescent
from .base import base

class steepest_descent(base):
	"""Steepest descent algorithm
	"""
	def __init__(self, ls_method='Bracket', max_ls=5, 
				step_len_init=0.05, step_len_max=0.5, 
				log_path='.', verbose=1):
		super().__init__(line_search_method=ls_method, max_ls=max_ls, 
					step_len_init=step_len_init, step_len_max=step_len_max, 
					log_path=log_path, verbose=verbose)


	def setup(self):
		super(SteepestDescent, self).setup()

	def compute_direction(self, m, g):
		super(SteepestDescent, self).compute_direction(m, g)

	def restart(self):
		# steepest descent never requires restarts
		pass