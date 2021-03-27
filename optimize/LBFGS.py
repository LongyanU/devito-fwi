
from optimizer import lbfgs
from .base import base

class LBFGS(base):
	"""Limited memory BFGS algorithm
	"""
	def __init__(self, memory=5, max_call=np.inf, thresh=0,
				ls_method='Bracket', max_ls=5, 
				step_len_init=0.05, step_len_max=0.5, 
				log_path='.', verbose=1):
		super().__init__(line_search_method=ls_method, max_ls=max_ls, 
					step_len_init=step_len_init, step_len_max=step_len_max, 
					log_path=log_path, verbose=verbose)

		self.memory = memory
		self.max_call = max_call
		self.thresh = thresh

	def setup(self):
		super(LBFGS, self).setup()

		self.lbfgs = lbfgs(memory=self.memory, 
					max_call=self.max_call,
					thresh=self.thresh)

	def compute_direction(self, m_new, g_new):
		p_new, self.restarted = self.lbfgs.compute_direction(m_new, g_new)

	def restart(self):
		super(LBFGS, self).restart()

		self.lbfgs.restart()