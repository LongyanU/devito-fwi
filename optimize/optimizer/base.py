

class Base(object):
	"""
	"""

	def __init__(self):
		self.call_count = 0

	def compute_direction(self, m, g):

		return -g, 0

steepest_descent = Base