
from .base import base
from optimizer import nlcg

class NLCG(base):
	"""Non-linear congugate gradient algorithm
	"""
	def __init__(self, max_call=np.inf, thresh=0,
			ls_method='Bracket', max_ls=5, 
			step_len_init=0.05, step_len_max=0.5, 
			log_path='.', verbose=1):
		super().__init__(line_search_method=ls_method, max_ls=max_ls, 
					step_len_init=step_len_init, step_len_max=step_len_max, 
					log_path=log_path, verbose=verbose)

		self.max_call = max_call
		self.thresh = thresh

	def name(self):
		return 'NLCG'

	def setup(self):
		super(NLCG, self).setup()

		self.nlcg = nlcg( max_call=self.max_call, thresh=self.thresh)

	def compute_direction(self, m, g):
		p, self.restarted = self.nlcg.compute_direction(m, g)
		return p

	def restart(self):
		super(NLCG, self).restart()

		self.nlcg.restart()	