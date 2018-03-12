import time

import tensorflow as tf

def print_time(fn, name):
	def fn_with_print_time(*args, **kwargs):
		start = time.time()
		result = fn(*args, **kwargs)
		elapsed = time.time() - start
		print("{0:s} took {1:d} seconds".format(name, int(elapsed)))
		return result
	return fn_with_print_time

def tensor_eval(tensor):
	"""Evaluate a tensor.

	Args:
	    tensor (tf.Tensor): The tensor.

	Returns:
	    (np.ndarray) The value of the tensor.
	"""
	with tf.Session().as_default():
		return tensor.eval()
