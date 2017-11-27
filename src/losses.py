def make_class_difference_loss(class_fn=lambda x: x, exp=1):
	"""Make a class-difference loss function.

	The cvlass difference loss is this:
	\sum_{(c,y)} \sum_{i=0}^7 y_i|class_fn(i) - class_fn(c)|^exp

	c is the correct class for an example
	y is the softmax of the model's output for an example (a vector)

	Args:
		class_fn (func(int) -> int): A function that takes a class number and
			returns a number. Applied to the class numbers before they are used
			in the calculation. Default is the identity function.
		exp (float): The exponent on the class difference. Default is 1.
	"""
	def class_difference_loss(pred, actual):
		pred_cls = np.argmax(pred, axis=1)
		actual_cls = np.argmax(actual, axis=1)
		for i in range(pred_clas.shape[0]):
			pred_cls[i] = class_fn(pred_cls[i])
			actual_cls[i] = class_fn(actual_cls[i])
		diffs = pred_cls - actual_cls
		diffs = np.absolute(diffs)
		diffs = np.power(diffs, exp)
		return diffs
	return class_difference_loss