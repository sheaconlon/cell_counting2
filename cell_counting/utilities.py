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

def prompt(message, converter):
	"""Prompt for some information at the command line.

	Args:
	    message (str): A message indicating what information is desired. Will be
	    	followed by a prompt on the next line.
	    converter (func(str)->tuple(bool, *)): A function that converts the
	    	response to some desired type or format. Returns a pair. If the
	    	response is valid, the first element is ``True`` and the second
	    	element is the converted response. If the response is not valid,
	    	the first element is ``False`` and the second element is an error
	    	message to display before the prompt is repeated.
	"""
	print("=" * 10)
	while True:
		print(message)
		success, value = converter(input("Enter here: "))
		if success:
			break
		else:
			err_message = value
			print(err_message)
	return value
